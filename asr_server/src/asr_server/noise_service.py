"""speech.proto task noise_reduction: AudioChunk -> DPDFNet-2 -> RecognitionResult with audio.

Design: docs/DPDFNet2_Task_Routing_Design_VI.md (audio-engine repo root), sections 4, 5 and 8.
The gateway plays the original audio for the rest of the call when a stream returns no audio
for noise_reduction.fallback_after_ms (500 ms) or ends with an error (section 5.5), so:
- audio goes back for every chunk, as soon as the model returns it;
- the output format is fixed by output_audio_format (field 14) of the first chunk: absent or
  LEGACY_ENCODED = G.711 PCMU 8 kHz (audio_pt 0), PCM_S16LE = PCM16 LE 16 kHz with
  audio_sample_rate 16000. Any other value, or a change within the stream, is INVALID_ARGUMENT;
- input is PCM16 LE mono 16 kHz only;
- ts_start = ts_end = rtp_timestamp of the latest chunk given to the model: the gateway looks
  that value up to measure the latency;
- is_final once, on an empty result after end_of_stream or the client's half-close and the
  model flush. A cancelled RPC gets no final;
- each stream borrows a denoiser from the pool; none free = RESOURCE_EXHAUSTED at once.
  A model error ends the stream with UNAVAILABLE;
- chunks are read as they arrive into a queue; when more than max_backlog_ms of audio waits
  there for the model (the CPU does not keep up), the stream ends with RESOURCE_EXHAUSTED
  instead of returning ever later audio that the gateway would still accept.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

import grpc
import numpy as np

from .denoise import FRAME_MS, OUTPUT_PCM16, OUTPUT_PCMU, NoiseStream
from .proto import speech_pb2 as pb
from .speech_service import NOISE_TASK, ProtocolError

LOG = logging.getLogger('qwen3_asr.grpc.noise')

SAMPLE_RATE = 16000
# speech.proto AudioFormat (PCM_S16LE = 1), not messages.PCM_S16LE (= 3) of the speech_to_text core.
OUTPUTS = {pb.AudioFormat.LEGACY_ENCODED: OUTPUT_PCMU, pb.AudioFormat.PCM_S16LE: OUTPUT_PCM16}
MAX_CHUNK_BYTES = 64000             # as speech_to_text (server.py): 2 s of audio
# Chunks read ahead at most, whatever max_backlog_ms says (0 = no limit): further chunks stay
# with gRPC, whose flow control then holds the client back.
MAX_QUEUED_CHUNKS = 64


class ModelFailure(Exception):
    """A model call failed; the stream ends with UNAVAILABLE.

    `broken`: the denoiser raised and is replaced; False when the service was closed."""

    def __init__(self, message, broken=True):
        super().__init__(message)
        self.broken = broken


class Backlog(Exception):
    """More audio waits for the model than noise_reduction.max_backlog_ms: RESOURCE_EXHAUSTED."""


_EOF = object()                             # the client half-closed the request stream


class _ReadError:
    """Reading the request stream raised; the stream ends with that error."""

    def __init__(self, exc):
        self.exc = exc


class _Histogram:
    """Milliseconds in 1 ms bins up to 1 s (the last bin takes the rest): same memory for any call."""

    def __init__(self):
        self.counts = np.zeros(1001, dtype=np.int64)
        self.max = 0.0

    def add(self, ms):
        self.counts[min(int(ms), 1000)] += 1
        self.max = max(self.max, ms)

    def percentile(self, q):
        total = int(self.counts.sum())
        if not total:
            return 0
        return int(np.searchsorted(np.cumsum(self.counts), q / 100 * total))


class _State:
    """What the first chunk fixed for the RPC, and the figures of the summary line."""

    def __init__(self, chunk):
        self.session_id = chunk.session_id
        self.stream_id = chunk.stream_id
        self.rtp_clock_rate = chunk.rtp_clock_rate
        self.output_audio_format = chunk.output_audio_format
        self.next_chunk_seq = None if chunk.chunk_seq == -1 else chunk.chunk_seq
        self.ts = chunk.rtp_timestamp       # rtp_timestamp of the latest chunk given to the model
        self.seq = 0
        self.job = None                     # concurrent future of the latest model call
        self.chunks = 0
        self.chunk_seq_gaps = 0
        self.started = time.monotonic()
        self.first_audio_ms = None
        self.model_ms = _Histogram()        # time in the model
        self.queue_ms = _Histogram()        # time waiting for a worker thread
        self.waiting_samples = 0            # audio received and not yet given to the model
        self.backlog_ms_max = 0
        self.backlog = None                 # set by the reader when max_backlog_ms is passed
        self.slots = asyncio.Semaphore(MAX_QUEUED_CHUNKS)


class NoiseReductionService:
    def __init__(self, pool, *, frame_ms=20, workers=2, max_backlog_ms=500):
        """`max_backlog_ms`: audio allowed to wait for the model per stream; 0 = no limit."""
        if frame_ms not in FRAME_MS:
            raise ValueError(f'frame_ms must be one of {FRAME_MS}, got {frame_ms}')
        self.pool = pool
        self.frame_ms = frame_ms
        self.max_backlog_ms = max_backlog_ms
        self.refused = 0                    # streams refused because no denoiser was free
        self.backlogged = 0                 # streams ended by max_backlog_ms
        # Model calls of all streams; those of one stream run one at a time (each is awaited).
        # Loading a denoiser in place of a dropped one runs here too.
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='denoise')

    def close(self):
        """Call after the gRPC server has stopped: later model calls fail with UNAVAILABLE."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ input
    @staticmethod
    def _open(chunk):
        if not chunk.session_id or not chunk.stream_id:
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT, 'session_id and stream_id are mandatory')
        if chunk.sample_rate != SAMPLE_RATE:
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                f'noise_reduction needs sample_rate 16000, got {chunk.sample_rate}')
        if chunk.output_audio_format not in OUTPUTS:
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                f'unknown output_audio_format {chunk.output_audio_format}')
        return _State(chunk)

    @staticmethod
    def _check(state, chunk):
        if (chunk.session_id != state.session_id or chunk.stream_id != state.stream_id
                or chunk.task != NOISE_TASK or chunk.sample_rate != SAMPLE_RATE
                or chunk.rtp_clock_rate != state.rtp_clock_rate
                or chunk.output_audio_format != state.output_audio_format):
            raise ProtocolError(
                grpc.StatusCode.INVALID_ARGUMENT,
                'session_id, stream_id, task, sample_rate, rtp_clock_rate and output_audio_format '
                'must not change within one Recognize call')
        if chunk.channels not in (0, 1):          # 0 = field not set
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                f'pcm must be mono, got channels={chunk.channels}')
        if len(chunk.pcm) % 2:
            raise ProtocolError(grpc.StatusCode.INVALID_ARGUMENT,
                                'pcm length must be a multiple of 2 (16-bit samples)')
        if len(chunk.pcm) > MAX_CHUNK_BYTES:
            raise ProtocolError(grpc.StatusCode.RESOURCE_EXHAUSTED, 'audio chunk above 64 KB')
        if state.next_chunk_seq is not None:
            if chunk.chunk_seq != state.next_chunk_seq:
                # Chunks dropped by the gateway's send queue: the audio is lost, the stream goes on.
                state.chunk_seq_gaps += 1
                if state.chunk_seq_gaps <= 3 or state.chunk_seq_gaps % 500 == 0:
                    LOG.warning('chunk_seq gap session=%s stream=%s: expected %d, got %d (gap %d of '
                                'this stream); processing goes on', state.session_id, state.stream_id,
                                state.next_chunk_seq, chunk.chunk_seq, state.chunk_seq_gaps)
            state.next_chunk_seq = chunk.chunk_seq + 1

    async def _read(self, request_iterator, queue, state):
        """Move the chunks into `queue` as they arrive, so that the audio waiting to be processed
        is known: unread, it would wait unseen in the gRPC buffers. Stops after end_of_stream, at
        max_backlog_ms (the stream then ends at its next chunk), and waits while
        MAX_QUEUED_CHUNKS are queued. Always leaves an item that ends the stream's wait."""
        try:
            async for chunk in request_iterator:
                await state.slots.acquire()
                state.waiting_samples += len(chunk.pcm) // 2
                waiting_ms = state.waiting_samples * 1000 // SAMPLE_RATE
                state.backlog_ms_max = max(state.backlog_ms_max, waiting_ms)
                queue.put_nowait(chunk)
                if self.max_backlog_ms and waiting_ms > self.max_backlog_ms:
                    state.backlog = (f'{waiting_ms} ms of audio waiting to be processed, above '
                                     f'noise_reduction.max_backlog_ms {self.max_backlog_ms}')
                    return
                if chunk.end_of_stream:
                    return
        except BaseException as exc:        # handed to the stream, which ends with it
            queue.put_nowait(_ReadError(exc))
            if isinstance(exc, asyncio.CancelledError):
                raise
            return
        queue.put_nowait(_EOF)

    @staticmethod
    def _take(state, item):
        """The next chunk out of the queue, or None at the half-close."""
        if state.backlog is not None:
            raise Backlog(state.backlog)
        if item is _EOF:
            return None
        state.slots.release()
        state.waiting_samples -= len(item.pcm) // 2
        return item

    # ----------------------------------------------------------------- output
    @staticmethod
    def _result(state, payload, final=False):
        state.seq += 1
        if payload and state.first_audio_ms is None:
            state.first_audio_ms = round(1000 * (time.monotonic() - state.started))
        result = pb.RecognitionResult(
            session_id=state.session_id, stream_id=state.stream_id, is_final=final,
            ts_start=state.ts, ts_end=state.ts, seq=state.seq, audio_payload=payload)
        if state.output_audio_format == pb.AudioFormat.PCM_S16LE:
            result.audio_format = pb.AudioFormat.PCM_S16LE
            result.audio_sample_rate = SAMPLE_RATE
        return result          # LEGACY_ENCODED: audio_format and audio_pt (PCMU) stay 0

    # -------------------------------------------------------------- model calls
    async def _call(self, state, fn, *args):
        """A blocking model call on the worker threads."""
        def timed():
            started = time.perf_counter()
            return fn(*args), time.perf_counter() - started
        submitted = time.perf_counter()
        try:
            state.job = self._executor.submit(timed)
        except RuntimeError as exc:         # close() was called: the server is stopping
            raise ModelFailure(f'service closed ({exc})', broken=False) from exc
        try:
            result, model_s = await asyncio.wrap_future(state.job)
        except Exception as exc:
            raise ModelFailure(f'{type(exc).__name__}: {exc}') from exc
        state.model_ms.add(1000 * model_s)
        state.queue_ms.add(1000 * (time.perf_counter() - submitted - model_s))
        return result

    def _release(self, state, denoiser, broken):
        """Give the denoiser back once no model call uses it: a cancelled RPC leaves its call running.

        A denoiser the pool drops (`broken`, or its reset failed) is replaced off the event loop."""
        def give_back():
            if not self.pool.release(denoiser, broken=broken):
                try:
                    self._executor.submit(self.pool.replace)
                except RuntimeError:        # service closed
                    pass
        job = state.job
        if job is None or job.done():
            give_back()
            return
        loop = asyncio.get_running_loop()

        def back(_):
            try:
                loop.call_soon_threadsafe(give_back)
            except RuntimeError:            # event loop closed: the server is shutting down
                pass
        job.add_done_callback(back)

    # -------------------------------------------------------------------- rpc
    async def Recognize(self, first, request_iterator, context):
        """`first`: the first AudioChunk, already read; `request_iterator` gives the rest."""
        try:
            # The whole first chunk is checked before a denoiser is borrowed: a bad stream gets
            # INVALID_ARGUMENT (not retried by the gateway) even when the pool is full.
            state = self._open(first)
            self._check(state, first)
        except ProtocolError as exc:
            LOG.warning('noise_reduction protocol error session=%s stream=%s: %s',
                        first.session_id, first.stream_id, exc.details)
            await context.abort(exc.code, exc.details)
            return
        denoiser = self.pool.acquire()
        if denoiser is None:
            self.refused += 1
            if self.refused <= 3 or self.refused % 100 == 0:
                LOG.warning('noise_reduction refused session=%s stream=%s: all %d denoisers in use '
                            '(%d streams refused since start)', state.session_id, state.stream_id,
                            self.pool.size, self.refused)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                f'noise_reduction: all {self.pool.size} denoisers are in use')
            return
        stream = None
        reader = None
        outcome = 'cancelled'
        broken = False
        try:
            stream = NoiseStream(denoiser, output=OUTPUTS[state.output_audio_format],
                                 frame_ms=self.frame_ms)
            queue = asyncio.Queue()
            if not first.end_of_stream:
                reader = asyncio.create_task(self._read(request_iterator, queue, state))
            chunk = first
            end_ts = None
            while True:
                state.chunks += 1
                if chunk.pcm:
                    state.ts = chunk.rtp_timestamp
                    for payload in await self._call(state, stream.process, chunk.pcm):
                        yield self._result(state, payload)
                if chunk.end_of_stream:
                    end_ts = chunk.rtp_timestamp
                    break
                item = await queue.get()
                if isinstance(item, _ReadError):
                    outcome = 'read_error'
                    raise item.exc
                chunk = self._take(state, item)
                if chunk is None:           # half-close
                    break
                self._check(state, chunk)
            for payload in await self._call(state, stream.finish):
                yield self._result(state, payload)
            if end_ts is not None:          # the final carries the end_of_stream chunk's timestamp
                state.ts = end_ts
            yield self._result(state, b'', final=True)
            outcome = 'final'
        except ProtocolError as exc:
            outcome = 'invalid'
            LOG.warning('noise_reduction protocol error session=%s stream=%s: %s',
                        state.session_id, state.stream_id, exc.details)
            await context.abort(exc.code, exc.details)
        except ModelFailure as exc:
            outcome = 'model_error'
            broken = exc.broken
            LOG.error('noise_reduction model failed session=%s stream=%s: %s',
                      state.session_id, state.stream_id, exc)
            await context.abort(grpc.StatusCode.UNAVAILABLE, 'noise_reduction model failed')
        except Backlog as exc:
            outcome = 'backlog'
            self.backlogged += 1
            if self.backlogged <= 3 or self.backlogged % 100 == 0:
                LOG.warning('noise_reduction backlog session=%s stream=%s: %s; stream closed '
                            '(%d streams closed since start)', state.session_id, state.stream_id,
                            exc, self.backlogged)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f'noise_reduction: {exc}')
        finally:
            try:
                if reader is not None:
                    reader.cancel()
            finally:
                try:
                    self._release(state, denoiser, broken)
                finally:
                    self._summary(state, stream, outcome)

    def _summary(self, state, stream, outcome):
        def count(name):
            return getattr(stream, name) if stream is not None else 0
        LOG.info('noise_reduction stream ended session=%s stream=%s outcome=%s output=%s chunks=%d '
                 'samples_in=%d samples_out=%d samples_sent=%d clipped=%d results=%d first_audio_ms=%s '
                 'model_ms_p50=%d model_ms_p95=%d model_ms_max=%.1f queue_ms_p95=%d queue_ms_max=%.1f '
                 'backlog_ms_max=%d chunk_seq_gaps=%d',
                 state.session_id, state.stream_id, outcome, OUTPUTS[state.output_audio_format],
                 state.chunks, count('samples_in'), count('samples_out'), count('samples_sent'),
                 count('clipped'), state.seq, state.first_audio_ms, state.model_ms.percentile(50),
                 state.model_ms.percentile(95), state.model_ms.max, state.queue_ms.percentile(95),
                 state.queue_ms.max, state.backlog_ms_max, state.chunk_seq_gaps)
