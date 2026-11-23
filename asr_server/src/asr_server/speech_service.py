"""gRPC layer: speech.proto (SpeechStream.Recognize) on top of the recognition core.

AudioChunk  -> messages.StreamRequest   (config taken from the first chunk)
Transcript  -> RecognitionResult        (RTP ticks, language codes, seq, reason enum)

The task of the first chunk picks the pipeline: noise_reduction goes to
noise_service.NoiseReductionService when it is on; everything below is speech_to_text.

Contract enforced here (see README, "gRPC contract"):
- one RPC carries one (session_id, stream_id); the session fields of the first
  chunk must not change;
- pcm is signed 16-bit little-endian mono at sample_rate (16000, or 8000);
- chunk_seq, when provided (not -1), normally increases by 1; a gap means the gateway dropped
  chunks from its send queue: it is logged and processing goes on (closing the stream would
  make the gateway close the whole session);
- end_of_stream=true on a chunk: its audio is processed, then the stream is flushed
  and closed;
- ts_start / ts_end are RTP timestamps of the source clock, modulo 2^32. They follow the
  rtp_timestamp of the chunk that carried the audio: when a chunk's rtp_timestamp does not
  continue from the audio received so far, the timeline is re-anchored at that chunk.
"""
import logging

import grpc

from . import messages as m
from .proto import speech_pb2 as pb
from .proto import speech_pb2_grpc as pb_grpc
from .word_pacer import WordPacer, paced_transcripts

LOG = logging.getLogger('qwen3_asr.grpc.vad')

RTP_MODULO = 1 << 32
# RTP anchors older than this much audio are dropped; far more than the longest utterance
# plus the audio that may wait for ASR.
ANCHOR_HISTORY_MS = 60000

# speech.proto language code <-> language name used by the model
LANGUAGE_NAMES = {'vi': 'Vietnamese', 'en': 'English'}
LANGUAGE_CODES = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}
AUTO_CODE = 'auto'      # the model detects the language; never sent back to the client

SUPPORTED_TASKS = ('', 'speech_to_text')
# Served by noise_service.NoiseReductionService when noise_reduction.enabled is true.
NOISE_TASK = 'noise_reduction'
# Tasks the gateway may route here that this engine does not implement (or has off).
OTHER_GATEWAY_TASKS = ('realtime_translation', 'keyword', 'voice_controller_emoji', NOISE_TASK)

REASONS = {
    'vad_endpoint': pb.COMMIT_REASON_SILENCE,
    'semantic_pause': pb.COMMIT_REASON_SEMANTIC_PAUSE,
    'max_duration': pb.COMMIT_REASON_MAX_DURATION,
    'end_stream': pb.COMMIT_REASON_END_OF_STREAM,
    'stream_closed': pb.COMMIT_REASON_END_OF_STREAM,
    'end_utterance': pb.COMMIT_REASON_END_OF_STREAM,
}


class ProtocolError(Exception):
    """The client broke the contract; the RPC ends with `code`."""

    def __init__(self, code, details):
        super().__init__(details)
        self.code = code
        self.details = details


class _Stream:
    """What the first AudioChunk fixed for the whole RPC."""

    def __init__(self, chunk, language_code):
        self.session_id = chunk.session_id
        self.stream_id = chunk.stream_id
        self.sample_rate = chunk.sample_rate
        self.task = chunk.task
        self.language_code = language_code
        # speech.proto: rtp_clock_rate 0 = infer from sample_rate.
        self.declared_clock_rate = chunk.rtp_clock_rate
        self.clock_rate = chunk.rtp_clock_rate or chunk.sample_rate
        # RTP timeline: (index of the first sample, RTP timestamp of that sample), oldest first.
        # One entry while the sender's timestamps are continuous; one more per jump.
        self.anchors = [(0, chunk.rtp_timestamp)]
        self.samples_in = 0
        self.next_chunk_seq = None                  # None = sequence checking off
        self.chunk_seq_gaps = 0
        self.rtp_jumps = 0
        self.warned_languages = set()
        self.seq = 0

    def expected_rtp(self):
        """RTP timestamp the next sample has if the sender's timeline is continuous."""
        first, rtp = self.anchors[-1]
        return (rtp + (self.samples_in - first) * self.clock_rate // self.sample_rate) % RTP_MODULO

    def reanchor(self, rtp_timestamp):
        """The next sample carries `rtp_timestamp`, which does not continue the timeline."""
        if self.anchors[-1][0] == self.samples_in:      # no audio received on the last anchor
            self.anchors[-1] = (self.samples_in, rtp_timestamp)
        else:
            self.anchors.append((self.samples_in, rtp_timestamp))
        horizon = self.samples_in - ANCHOR_HISTORY_MS * self.sample_rate // 1000
        while len(self.anchors) > 1 and self.anchors[1][0] <= horizon:
            del self.anchors[0]

    def ticks(self, media_ms, end=False):
        """Media time since the first sample -> RTP timestamp, modulo 2^32.

        A position exactly on a jump belongs to the new anchor when it starts an interval
        and to the old one when it ends an interval (`end=True`): [ts_start, ts_end) is
        half-open, so the end of audio received before the jump stays on the old timeline.
        """
        position = media_ms * self.sample_rate / 1000.0
        first, rtp = self.anchors[0]
        for anchor in self.anchors[1:]:
            if anchor[0] < position or (anchor[0] == position and not end):
                first, rtp = anchor
            else:
                break
        return (rtp + round((position - first) * self.clock_rate / self.sample_rate)) % RTP_MODULO


class SpeechStreamService(pb_grpc.SpeechStreamServicer):
    def __init__(self, core, *, use_language_source=False, word_reveal=None, noise=None):
        """`core`: the speech_to_text core (server.ASRService); None = asr.enabled is false.
        `word_reveal`: (window_ms, min_interval_ms) to reveal PARTIALs word by word; None = off.
        `noise`: noise_service.NoiseReductionService; None = noise_reduction is off."""
        self.core = core
        self.use_language_source = bool(use_language_source)
        self.word_reveal = word_reveal
        self.noise = noise

    # ------------------------------------------------------------------ input
    def _language_code(self, chunk):
        code = chunk.language
        if self.use_language_source and chunk.language_source:
            code = chunk.language_source
        return code.strip().lower()

    def _open(self, chunk):
        if not chunk.session_id or not chunk.stream_id:
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                'session_id and stream_id are mandatory')
        if chunk.task not in SUPPORTED_TASKS:
            if chunk.task in OTHER_GATEWAY_TASKS:
                raise ProtocolError(grpc.StatusCode.UNIMPLEMENTED,
                                    f'task {chunk.task} is not supported by this engine')
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT, f'unknown task {chunk.task!r}')
        if chunk.sample_rate not in (8000, 16000):
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                f'sample_rate must be 16000 (or 8000), got {chunk.sample_rate}')
        if chunk.channels not in (0, 1):          # 0 = field not set
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                f'pcm must be mono, got channels={chunk.channels}')
        code = self._language_code(chunk)
        if code and code != AUTO_CODE and code not in LANGUAGE_NAMES:
            LOG.warning('unknown language code %r session=%s stream=%s; using the server default',
                        code, chunk.session_id, chunk.stream_id)
        stream = _Stream(chunk, code)
        if chunk.chunk_seq != -1:
            stream.next_chunk_seq = chunk.chunk_seq
        if not chunk.rtp_clock_rate:
            LOG.warning('rtp_clock_rate not set session=%s stream=%s; using sample_rate %d as the '
                        'RTP clock', chunk.session_id, chunk.stream_id, chunk.sample_rate)
        return stream

    @staticmethod
    def _check(stream, chunk):
        if (chunk.session_id != stream.session_id or chunk.stream_id != stream.stream_id
                or chunk.sample_rate != stream.sample_rate or chunk.task != stream.task
                or chunk.rtp_clock_rate != stream.declared_clock_rate):
            raise ProtocolError(
                grpc.StatusCode.INVALID_ARGUMENT,
                'session_id, stream_id, sample_rate, task and rtp_clock_rate must not change '
                'within one Recognize call')
        if chunk.channels not in (0, 1):
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                f'pcm must be mono, got channels={chunk.channels}')

    @staticmethod
    def _accept(stream, chunk):
        """Sequence, payload and RTP continuity of one chunk."""
        if stream.next_chunk_seq is not None:
            if chunk.chunk_seq != stream.next_chunk_seq:
                # The gateway numbers chunks before its send queue and drops from that queue
                # when it is full. The audio in between is lost; what arrives is still valid.
                stream.chunk_seq_gaps += 1
                if stream.chunk_seq_gaps <= 3 or stream.chunk_seq_gaps % 500 == 0:
                    LOG.warning('chunk_seq gap session=%s stream=%s: expected %d, got %d '
                                '(gap %d of this stream); processing goes on',
                                stream.session_id, stream.stream_id, stream.next_chunk_seq,
                                chunk.chunk_seq, stream.chunk_seq_gaps)
            stream.next_chunk_seq = chunk.chunk_seq + 1
        if len(chunk.pcm) % 2:
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                'pcm length must be a multiple of 2 (16-bit samples)')
        if chunk.pcm:
            expected = stream.expected_rtp()
            if chunk.rtp_timestamp != expected:
                if stream.samples_in:               # before any audio it is just the first anchor
                    stream.rtp_jumps += 1
                    if stream.rtp_jumps <= 3 or stream.rtp_jumps % 500 == 0:
                        LOG.warning('rtp_timestamp jump session=%s stream=%s chunk_seq=%d: expected %d, '
                                    'got %d (jump %d of this stream); ts_start/ts_end follow the new '
                                    'timestamp from this chunk on',
                                    stream.session_id, stream.stream_id, chunk.chunk_seq, expected,
                                    chunk.rtp_timestamp, stream.rtp_jumps)
                stream.reanchor(chunk.rtp_timestamp)
        stream.samples_in += len(chunk.pcm) // 2

    # ----------------------------------------------------------------- output
    @staticmethod
    def _language_out(stream, name):
        if not name:
            return ''
        code = LANGUAGE_CODES.get(name.strip().lower())
        if code is None:
            if name not in stream.warned_languages:
                stream.warned_languages.add(name)
                LOG.warning('model reported language %r session=%s stream=%s, which has no code '
                            'in speech.proto; language is left empty', name,
                            stream.session_id, stream.stream_id)
            return ''
        return code

    def _result(self, stream, t):
        stream.seq += 1
        return pb.RecognitionResult(
            session_id=stream.session_id,
            stream_id=stream.stream_id,
            text=t.text,
            is_final=t.is_final,
            ts_start=stream.ticks(t.start_ms),
            ts_end=stream.ticks(t.end_ms, end=True),
            confidence=0.0,                     # no calibrated confidence available
            language=self._language_out(stream, t.language),
            seq=stream.seq,
            utterance_id=t.utterance_index,
            revision=t.revision,
            raw_text=t.raw_text,
            reason=(REASONS.get(t.reason, pb.COMMIT_REASON_UNSPECIFIED) if t.is_final
                    else pb.COMMIT_REASON_UNSPECIFIED),
            status=pb.RESULT_STATUS_OK,
        )

    # -------------------------------------------------------------------- rpc
    async def Recognize(self, request_iterator, context):
        # The task of the first chunk picks the pipeline.
        first = await anext(request_iterator, None)
        if first is not None and first.task == NOISE_TASK and self.noise is not None:
            async for result in self.noise.Recognize(first, request_iterator, context):
                yield result
            return
        if self.core is None:
            await self._refuse_without_asr(first, context)
            return
        holder = {}

        async def chunks():
            if first is not None:
                yield first
                async for chunk in request_iterator:
                    yield chunk

        async def requests():
            stream = None
            sequence = 0
            async for chunk in chunks():
                if stream is None:
                    stream = holder['stream'] = self._open(chunk)
                    yield m.StreamRequest(config=m.StreamConfig(
                        call_id=stream.session_id, leg_id=stream.stream_id,
                        encoding=m.PCM_S16LE, sample_rate=stream.sample_rate,
                        language=(m.AUTO_LANGUAGE if stream.language_code == AUTO_CODE
                                  else LANGUAGE_NAMES.get(stream.language_code, ''))))
                else:
                    self._check(stream, chunk)
                self._accept(stream, chunk)
                if chunk.pcm:
                    sequence += 1
                    yield m.StreamRequest(audio=m.AudioChunk(data=chunk.pcm, sequence=sequence))
                if chunk.end_of_stream:
                    yield m.StreamRequest(end_stream=m.EndStream())
                    return

        transcripts = self.core.StreamRecognize(requests(), context)
        if self.word_reveal is not None:
            transcripts = paced_transcripts(transcripts, WordPacer(*self.word_reveal))
        try:
            async for transcript in transcripts:
                yield self._result(holder['stream'], transcript)
        except ProtocolError as exc:
            LOG.warning('protocol error: %s', exc.details)
            await context.abort(exc.code, exc.details)

    async def _refuse_without_asr(self, first, context):
        """asr.enabled is false: the same checks as speech_to_text, then UNIMPLEMENTED."""
        if first is None:
            return
        try:
            self._open(first)
        except ProtocolError as exc:
            LOG.warning('protocol error: %s', exc.details)
            await context.abort(exc.code, exc.details)
            return
        await context.abort(grpc.StatusCode.UNIMPLEMENTED,
                            'task speech_to_text is off on this engine (asr.enabled: false)')
