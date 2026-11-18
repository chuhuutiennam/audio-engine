"""Qwen3-ASR 1.7B vLLM recognition core with internal VAD and punctuation.

Per-stream endpointer, per-utterance ASR state. PARTIAL replaces the current
utterance display; COMMIT is immutable. Punctuation alone never commits.

ASRService is protocol-independent: it consumes messages.StreamRequest and yields
messages.Transcript. speech_service.SpeechStreamService maps speech.proto onto it.

ASR backends (config asr.backend):
- native_vllm: asr_backend.VLLMASRBackend, vllm.LLM.chat() on the accumulated
  utterance audio, language auto-detected; no qwen_asr package;
- qwen_asr:    Qwen3ASRModel.LLM streaming state of the qwen_asr package.
VAD backends (config vad.backend): silero (TorchScript, CPU) or webrtc.

Concurrency model:
- one reader task per stream only moves gRPC messages into a bounded queue;
- while audio of a stream is waiting in that queue, its PARTIAL inferences are skipped
  (the audio is kept for the next one), so overload thins out PARTIALs instead of
  growing the queue until the stream is closed; a COMMIT is always inferred;
- every model call of every stream runs on ONE worker thread (InferWorker);
- Silero VAD is loaded and run on a small VAD thread pool, off the event loop;
- punctuation runs on its own small thread pool with a deadline.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import logging
import os

import grpc
import numpy as np

from . import messages as pb
from .audio import PCM16kConverter, decode_audio
from .punctuation import HFPunctuator, IdentityPunctuator, PunctResult
from .text_clean import strip_asr_markers
from .vad import SileroEndpointer, WebRTCEndpointer

LOG = logging.getLogger('qwen3_asr.grpc.vad')

_EOF = object()         # request stream ended
_OVERFLOW = object()    # ingress queue limit exceeded


class AsrFailure(Exception):
    """A Qwen SDK call raised; the utterance is lost, the stream goes on."""


class InferWorker:
    """Runs every model call on one dedicated thread, in submission order.

    A cancelled RPC cannot make two model calls overlap: a job that already
    started keeps the only thread busy until it returns, and a job that has not
    started yet is dropped. Results never cross streams because each job works
    on the state object passed to it and resolves only its own future.
    """

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='asr-infer')

    async def run(self, fn, *args, alive=None):
        def job():
            if alive is not None and not alive():
                return False
            fn(*args)
            return True
        return await asyncio.get_running_loop().run_in_executor(self._pool, job)

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def _consume_exception(future):
    if not future.cancelled():
        future.exception()


def _accepts_language(restore):
    """True when punctuator.restore() can be called with language=..."""
    try:
        params = inspect.signature(restore).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == 'language' or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


class ASRService:
    def __init__(self, model, punctuator, *, chunk_sec=1.0, max_utterance_sec=15.0,
                 endpoint_ms=600, pause_ms=300, vad_mode=2, min_voiced_ms=160,
                 semantic_early_commit=False, semantic_confidence=0.90,
                 stable_updates=2, vad_factory=None,
                 vad_backend='silero', silero_model=None,
                 vad_threshold=0.5, vad_neg_threshold=None, vad_workers=2,
                 ingress_max_ms=3000, audio_timeout_ms=2000,
                 partial_max_backlog_ms=500,
                 max_consecutive_errors=3, punct_workers=1,
                 punct_partial_timeout_ms=300, punct_commit_timeout_ms=1000):
        if not 0.2 <= chunk_sec < max_utterance_sec:
            raise ValueError('require 0.2 <= chunk_sec < max_utterance_sec')
        self.asr = model
        self.punctuator = punctuator
        # Punctuators written before the language argument existed keep working.
        self._punct_takes_language = _accepts_language(punctuator.restore)
        self.chunk_sec = float(chunk_sec)
        self.chunk_samples = int(round(16000 * self.chunk_sec))
        self.max_utterance_samples = int(round(16000 * max_utterance_sec))
        self.endpoint_ms = endpoint_ms
        self.pause_ms = pause_ms
        self.vad_mode = vad_mode
        self.min_voiced_ms = min_voiced_ms
        self.semantic_early_commit = bool(semantic_early_commit)
        self.semantic_confidence = float(semantic_confidence)
        self.stable_updates = int(stable_updates)
        self.vad_factory = vad_factory
        if vad_backend not in ('silero', 'webrtc'):
            raise ValueError("vad_backend must be 'silero' or 'webrtc'")
        self.vad_backend = vad_backend
        self.silero_model = silero_model
        self.vad_threshold = vad_threshold
        self.vad_neg_threshold = vad_neg_threshold
        self.vad_pool = ThreadPoolExecutor(max_workers=max(1, int(vad_workers)),
                                           thread_name_prefix='vad')
        self.ingress_max_ms = int(ingress_max_ms)          # 0 = unlimited
        self.audio_timeout_ms = int(audio_timeout_ms)      # 0 = disabled
        self.partial_max_backlog_ms = int(partial_max_backlog_ms)   # 0 = never skip a PARTIAL
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.punct_partial_timeout_ms = int(punct_partial_timeout_ms)
        self.punct_commit_timeout_ms = int(punct_commit_timeout_ms)
        self.infer_worker = InferWorker()
        self.punct_pool = ThreadPoolExecutor(max_workers=max(1, int(punct_workers)),
                                             thread_name_prefix='punct')

    def close(self):
        self.infer_worker.shutdown()
        self.vad_pool.shutdown(wait=False, cancel_futures=True)
        self.punct_pool.shutdown(wait=False, cancel_futures=True)

    def build_vad(self, sample_rate):
        """One endpointer per stream. Silero is stateful and must never be shared."""
        if self.vad_factory:
            return self.vad_factory(sample_rate)
        if self.vad_backend == 'silero':
            return SileroEndpointer(
                sample_rate, model_path=self.silero_model,
                threshold=self.vad_threshold, neg_threshold=self.vad_neg_threshold,
                endpoint_ms=self.endpoint_ms, pause_ms=self.pause_ms,
                min_voiced_ms=self.min_voiced_ms)
        return WebRTCEndpointer(sample_rate, mode=self.vad_mode,
                                endpoint_ms=self.endpoint_ms, pause_ms=self.pause_ms,
                                min_voiced_ms=self.min_voiced_ms)

    def submit_punctuation(self, text, language=None):
        """Start punctuation on the punctuation pool; returns an asyncio future.

        `language` is the ASR language of the utterance (None when unknown).
        """
        if self._punct_takes_language:
            call = lambda: self.punctuator.restore(text, language=language or None)
        else:
            call = lambda: self.punctuator.restore(text)
        future = asyncio.get_running_loop().run_in_executor(self.punct_pool, call)
        future.add_done_callback(_consume_exception)
        return future

    async def await_punctuation(self, future, text, timeout_ms):
        """Wait for `future` up to timeout_ms; on timeout or error fall back to the raw text."""
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            LOG.warning('punctuation not ready after %d ms; using raw text', timeout_ms)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception('punctuation failed; using raw text')
        return PunctResult(text)

    async def punctuate(self, text, timeout_ms=None, language=None):
        if not text or not text.strip():
            return PunctResult(text or '')
        if timeout_ms is None:
            timeout_ms = self.punct_commit_timeout_ms
        return await self.await_punctuation(
            self.submit_punctuation(text, language), text, timeout_ms)

    async def StreamRecognize(self, request_iterator, context):
        cfg = None
        vad = None
        vad_offload = False           # run vad.push() on the VAD pool (Silero forward)
        loop = asyncio.get_running_loop()
        asr_state = None
        converter = None
        utterance_index = 1
        commit_id = 0
        revision = 0
        audio_samples = 0
        current_start_ms = 0
        last_text = ''
        last_preview = PunctResult('')
        stable_count = 0
        seq = 0
        pending = []                  # 16 kHz audio not yet handed to the model
        pending_samples = 0
        partial_sent = False          # a PARTIAL with text is on the client's screen
        skip_until_endpoint = False   # utterance failed: drop its remaining audio
        consecutive_errors = 0
        punct_pending = None          # PARTIAL punctuation that missed its deadline
        skipped_partials = 0          # PARTIAL inferences skipped in this utterance (backlog)
        deferred_chunks = 0           # whole chunks waiting in `pending` already counted as skipped
        backlog_warned = False        # one WARNING per stream
        alive = True
        queue = asyncio.Queue()
        queued_ms = 0.0

        def new_state(start_ms):
            nonlocal asr_state, converter, audio_samples, pending, pending_samples
            nonlocal current_start_ms, last_text, last_preview, stable_count, partial_sent
            nonlocal skipped_partials, deferred_chunks
            skipped_partials = 0
            deferred_chunks = 0
            asr_state = self.asr.init_streaming_state(
                language=cfg.language or 'Vietnamese',
                unfixed_chunk_num=2, unfixed_token_num=5,
                chunk_size_sec=self.chunk_sec)
            converter = PCM16kConverter(cfg.sample_rate)
            # `revision` is NOT reset here: it restarts only when utterance_index
            # advances (after a COMMIT), so it stays monotonic when an index is reused.
            audio_samples = 0
            pending = []
            pending_samples = 0
            current_start_ms = int(start_ms)
            last_text = ''
            last_preview = PunctResult('')
            stable_count = 0
            partial_sent = False

        def utterance_language():
            """Language of the open utterance: what the model reports, else what the client asked for."""
            return (asr_state.language if asr_state is not None else '') or cfg.language or ''

        def make_event(kind, text, raw_text, reason, end_ms):
            language = utterance_language()
            return pb.Transcript(
                call_id=cfg.call_id, leg_id=cfg.leg_id,
                utterance_index=utterance_index, text=text,
                is_final=(kind == pb.COMMIT), revision=revision,
                language=language,
                reason=reason, event=kind, raw_text=raw_text,
                commit_id=(commit_id if kind == pb.COMMIT else 0),
                start_ms=max(0, current_start_ms), end_ms=max(0, int(end_ms)),
            )

        def clear_partial(end_ms):
            """Empty PARTIAL: tells the client to drop the text shown for this utterance."""
            nonlocal revision, partial_sent
            revision += 1
            partial_sent = False
            return make_event(pb.PARTIAL, '', '', 'partial', end_ms)

        async def infer(fn, *args):
            """Run a model call on the inference worker; AsrFailure if the model raised."""
            nonlocal consecutive_errors
            try:
                await self.infer_worker.run(fn, *args, alive=lambda: alive)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_errors += 1
                raise AsrFailure(repr(exc)) from exc
            consecutive_errors = 0

        async def partial_preview(text):
            """Punctuate a PARTIAL without holding it back longer than the partial deadline."""
            nonlocal punct_pending
            if not text.strip():
                return PunctResult(text)
            if punct_pending is not None and not punct_pending.done():
                return PunctResult(text)      # previous preview still running: stay on raw text
            future = self.submit_punctuation(text, utterance_language())
            result = await self.await_punctuation(future, text, self.punct_partial_timeout_ms)
            punct_pending = None if future.done() else future
            return result

        async def feed(samples, end_ms):
            nonlocal audio_samples, revision, last_text, last_preview, stable_count
            nonlocal pending, pending_samples, partial_sent
            nonlocal skipped_partials, deferred_chunks, backlog_warned
            if samples.size == 0:
                return None
            if asr_state is None:
                raise RuntimeError('ASR utterance state not created')
            samples16 = converter.push(samples)
            if samples16.size == 0:
                return None
            audio_samples += int(samples16.size)
            pending.append(samples16)
            pending_samples += int(samples16.size)
            if pending_samples < self.chunk_samples:
                return None
            # The model is behind: more audio of this stream is already waiting. A PARTIAL
            # inferred now would be stale on arrival and the wait for it would let the queue
            # grow until the stream is closed. Keep the audio and infer once caught up.
            if self.partial_max_backlog_ms and queued_ms > self.partial_max_backlog_ms:
                waiting_chunks = pending_samples // self.chunk_samples
                if waiting_chunks > deferred_chunks:
                    deferred_chunks = waiting_chunks
                    skipped_partials += 1
                if not backlog_warned:
                    backlog_warned = True
                    LOG.warning('asr backlog call=%s leg=%s utterance=%s: %d ms of audio waiting; '
                                'PARTIAL inference is skipped until the stream has caught up',
                                cfg.call_id, cfg.leg_id, utterance_index, int(queued_ms))
                return None
            deferred_chunks = 0
            # Hand whole chunks to the model; keep the remainder for the next step.
            block = np.concatenate(pending)
            take = (block.size // self.chunk_samples) * self.chunk_samples
            pending = [block[take:]] if take < block.size else []
            pending_samples = int(block.size - take)
            await infer(self.asr.streaming_transcribe, block[:take], asr_state)
            new_text = strip_asr_markers(asr_state.text or '')
            stable_count = stable_count + 1 if new_text == last_text else 1
            if new_text == last_text:
                return None
            # No PARTIAL until the utterance has enough voiced audio to survive the
            # VAD's discard rule, so a PARTIAL can never belong to a discarded utterance.
            if vad.voiced_frames < vad.min_voiced_frames:
                return None
            # Qwen can revise previous words; never append a PARTIAL as a delta.
            last_text = new_text
            last_preview = await partial_preview(new_text)
            revision += 1
            partial_sent = bool(last_preview.text.strip())
            return make_event(pb.PARTIAL, last_preview.text, new_text, 'partial', end_ms)

        async def finish(reason, end_ms):
            nonlocal asr_state, utterance_index, revision, commit_id
            nonlocal pending, pending_samples, partial_sent
            if asr_state is None:
                return None
            state = asr_state
            tail = converter.flush()
            parts = pending + ([tail] if tail.size else [])
            rest = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
            pending = []
            pending_samples = 0

            def flush_and_finish():
                if rest.size:
                    self.asr.streaming_transcribe(rest, state)
                self.asr.finish_streaming_transcribe(state)

            await infer(flush_and_finish)
            raw = strip_asr_markers(state.text or '')
            # Do not commit an empty transcript. It is commonly silence/noise.
            if not raw.strip():
                msg = None
                if partial_sent:
                    LOG.debug('empty final text call=%s leg=%s utterance=%s; clearing partial',
                              cfg.call_id, cfg.leg_id, utterance_index)
                    msg = clear_partial(end_ms)
                asr_state = None
                return msg
            formatted = await self.punctuate(raw, self.punct_commit_timeout_ms, utterance_language())
            revision += 1
            commit_id += 1
            msg = make_event(pb.COMMIT, formatted.text, raw, reason, end_ms)
            LOG.info('commit call=%s leg=%s utterance=%s reason=%s words=%s skipped_partials=%d',
                     cfg.call_id, cfg.leg_id, utterance_index, reason, len(raw.split()),
                     skipped_partials)
            utterance_index += 1
            revision = 0
            partial_sent = False
            asr_state = None
            return msg

        async def fail_utterance(exc, end_ms):
            """Model call failed: log, clear the client's partial, keep the stream."""
            nonlocal asr_state, skip_until_endpoint
            LOG.error('asr error call=%s leg=%s utterance=%s consecutive=%d: %s',
                      cfg.call_id, cfg.leg_id, utterance_index, consecutive_errors, exc)
            out = [clear_partial(end_ms)] if partial_sent else []
            asr_state = None
            skip_until_endpoint = bool(vad.active)
            if consecutive_errors >= self.max_consecutive_errors:
                await context.abort(grpc.StatusCode.UNAVAILABLE,
                                    f'ASR model failed {consecutive_errors} times in a row')
            return out

        async def process_event(ev, forced_reason=None):
            nonlocal asr_state, skip_until_endpoint
            if ev.kind == 'audio':
                if skip_until_endpoint:
                    return []
                if asr_state is None:
                    new_state(ev.start_ms)
                out = []
                try:
                    result = await feed(ev.samples, ev.end_ms)
                    if result is not None:
                        out.append(result)
                    if audio_samples >= self.max_utterance_samples:
                        msg = await finish('max_duration', ev.end_ms)
                        # The VAD utterance goes on: what follows starts at the cut.
                        vad.rebase_start(ev.end_ms)
                        if msg is not None:
                            out.append(msg)
                except AsrFailure as exc:
                    out.extend(await fail_utterance(exc, ev.end_ms))
                return out
            if ev.kind == 'pause_candidate':
                if (self.semantic_early_commit and asr_state is not None
                        and vad.voiced_frames * vad.frame_ms >= self.min_voiced_ms
                        and stable_count >= self.stable_updates
                        and last_preview.last_mark in ('.', '?', '!')
                        and last_preview.last_mark_confidence >= self.semantic_confidence
                        and len(last_text.split()) >= 3):
                    try:
                        msg = await finish('semantic_pause', ev.end_ms)
                        out = [msg] if msg is not None else []
                    except AsrFailure as exc:
                        out = await fail_utterance(exc, ev.end_ms)
                    vad.reset_activity()
                    skip_until_endpoint = False
                    return out
                return []
            if ev.kind == 'endpoint':
                if skip_until_endpoint:
                    skip_until_endpoint = False
                    asr_state = None
                    return []
                try:
                    msg = await finish(forced_reason or 'vad_endpoint', ev.end_ms)
                except AsrFailure as exc:
                    return await fail_utterance(exc, ev.end_ms)
                return [msg] if msg is not None else []
            if ev.kind == 'discard':
                # Drop short VAD triggers; do not infer or emit a transcript.
                out = []
                if asr_state is not None and partial_sent:      # cannot happen with the voiced gate
                    out.append(clear_partial(ev.end_ms))
                LOG.debug('discard call=%s leg=%s utterance=%s', cfg.call_id, cfg.leg_id,
                          utterance_index)
                skip_until_endpoint = False
                asr_state = None
                return out
            raise RuntimeError('unknown VAD event')

        async def reader():
            """Only moves messages off the wire, so reading never waits for the model."""
            nonlocal queued_ms
            bytes_per_ms = 0.0
            try:
                async for req in request_iterator:
                    what = req.WhichOneof('payload')
                    duration = 0.0
                    if what == 'config' and req.config.sample_rate > 0:
                        width = 1 if req.config.encoding in (pb.PCMA, pb.PCMU) else 2
                        bytes_per_ms = req.config.sample_rate * width / 1000.0
                    elif what == 'audio' and bytes_per_ms:
                        duration = len(req.audio.data) / bytes_per_ms
                    if self.ingress_max_ms and duration and queued_ms + duration > self.ingress_max_ms:
                        queue.put_nowait(_OVERFLOW)
                        return
                    queued_ms += duration
                    queue.put_nowait((req, duration))
                queue.put_nowait(_EOF)
            except asyncio.CancelledError:
                raise
            except Exception as exc:        # transport error while reading
                queue.put_nowait(exc)

        reader_task = asyncio.ensure_future(reader())
        get_task = None
        try:
            while True:
                if get_task is None:
                    get_task = asyncio.ensure_future(queue.get())
                waiting_for_audio = (self.audio_timeout_ms > 0 and vad is not None and vad.active)
                done, _ = await asyncio.wait(
                    {get_task}, timeout=self.audio_timeout_ms / 1000.0 if waiting_for_audio else None)
                if not done:
                    # An utterance is open and the sender went silent without sending frames.
                    LOG.warning('audio_timeout call=%s leg=%s utterance=%s: no audio for %d ms, committing',
                                cfg.call_id, cfg.leg_id, utterance_index, self.audio_timeout_ms)
                    for event in vad.finish():
                        for response in await process_event(event, forced_reason='vad_endpoint'):
                            yield response
                    continue
                item = get_task.result()
                get_task = None
                if item is _EOF:
                    break
                if item is _OVERFLOW:
                    LOG.error('ingress overflow call=%s leg=%s: more than %d ms of audio waiting',
                              cfg.call_id if cfg else '', cfg.leg_id if cfg else '', self.ingress_max_ms)
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                        f'more than {self.ingress_max_ms} ms of audio waiting for ASR')
                if isinstance(item, Exception):
                    raise item
                req, duration = item
                queued_ms -= duration

                what = req.WhichOneof('payload')
                if cfg is None:
                    if what != 'config':
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                            'first message must contain config')
                    cfg = req.config
                    if not cfg.call_id or not cfg.leg_id:
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                            'call_id and leg_id are mandatory')
                    if cfg.encoding not in (pb.PCMA, pb.PCMU, pb.PCM_S16LE):
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'unsupported codec')
                    if cfg.sample_rate not in (8000, 16000):
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'expected 8k or 16k')
                    if cfg.encoding in (pb.PCMA, pb.PCMU) and cfg.sample_rate != 8000:
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'G.711 requires 8k')
                    if self.vad_factory:
                        vad = self.build_vad(cfg.sample_rate)
                    else:
                        # Loading the Silero checkpoint takes tens of ms: not on the event loop.
                        vad = await loop.run_in_executor(self.vad_pool, self.build_vad,
                                                         cfg.sample_rate)
                    vad_offload = bool(getattr(vad, 'offload', False))
                    LOG.info('start call=%s leg=%s rate=%d codec=%d',
                             cfg.call_id, cfg.leg_id, cfg.sample_rate, cfg.encoding)
                    continue

                if what == 'audio':
                    if req.audio.sequence != seq + 1:
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                            'audio.sequence must start from 1, increment by 1')
                    seq = req.audio.sequence
                    data = req.audio.data
                    if len(data) > 64000:
                        await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                            'audio chunk above 64 KB')
                    if not data:
                        continue
                    try:
                        raw_pcm = decode_audio(data, cfg.encoding)
                    except ValueError as exc:
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                    # Process one VAD frame at a time. Semantic early commit can
                    # reset VAD safely without losing audio later in the same RPC.
                    for pos in range(0, raw_pcm.size, vad.frame_samples):
                        piece = raw_pcm[pos:pos + vad.frame_samples]
                        if vad_offload:
                            events = await loop.run_in_executor(self.vad_pool, vad.push, piece)
                        else:
                            events = vad.push(piece)
                        for event in events:
                            for response in await process_event(event):
                                yield response
                elif what in ('end_stream', 'end_utterance'):
                    for event in vad.finish():
                        for response in await process_event(event, forced_reason=what):
                            yield response
                    if what == 'end_stream':
                        return
                else:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                        'unexpected message type')

            if cfg is not None and not context.cancelled():
                for event in vad.finish():
                    for response in await process_event(event, forced_reason='stream_closed'):
                        yield response
        finally:
            alive = False
            reader_task.cancel()
            if get_task is not None:
                get_task.cancel()


async def serve(args):
    """Load models and serve until terminated. `args` is asr_server.config.Settings."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    if not os.path.isdir(args.model):
        raise FileNotFoundError(f'missing local ASR checkpoint: {args.model}')
    if not os.path.isdir(args.punct_model):
        raise FileNotFoundError(f'missing local punctuation checkpoint: {args.punct_model}')
    if args.vad_backend == 'silero':
        if not os.path.isfile(args.silero_model):
            raise FileNotFoundError(f'missing local Silero VAD checkpoint: {args.silero_model}')
        # Validate the local JIT checkpoint before allocating ASR GPU memory.
        # Inference is CPU-only and never accesses the model hub.
        check_vad = SileroEndpointer(
            16000, model_path=args.silero_model,
            threshold=args.vad_threshold, neg_threshold=args.vad_neg_threshold,
            endpoint_ms=args.vad_endpoint_ms, pause_ms=args.vad_pause_ms,
            min_voiced_ms=args.min_voiced_ms)
        check_vad.push(np.zeros(check_vad.frame_samples, dtype=np.float32))
        del check_vad
        LOG.info('Silero VAD (CPU) from %s, threshold=%.2f', args.silero_model, args.vad_threshold)
    else:
        LOG.info('WebRTC VAD, mode=%d', args.vad_mode)
    if args.asr_backend == 'native_vllm':
        from .asr_backend import VLLMASRBackend
        LOG.info('loading Qwen3-ASR via native vLLM: %s', args.model)
        asr = VLLMASRBackend(
            model_path=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager)
    else:
        from qwen_asr import Qwen3ASRModel
        LOG.info('loading Qwen3-ASR via qwen_asr streaming: %s', args.model)
        asr = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens)
    LOG.info('loading punctuation model: %s', args.punct_model)
    punctuator = HFPunctuator(args.punct_model, device=args.punct_device,
                              threshold=args.punct_threshold)
    service = ASRService(
        asr, punctuator, chunk_sec=args.chunk_sec,
        max_utterance_sec=args.max_utterance_sec,
        endpoint_ms=args.vad_endpoint_ms, pause_ms=args.vad_pause_ms,
        vad_mode=args.vad_mode, min_voiced_ms=args.min_voiced_ms,
        semantic_early_commit=args.semantic_early_commit,
        semantic_confidence=args.semantic_confidence,
        stable_updates=args.stable_updates,
        vad_backend=args.vad_backend, silero_model=args.silero_model,
        vad_threshold=args.vad_threshold, vad_neg_threshold=args.vad_neg_threshold,
        vad_workers=args.vad_workers,
        ingress_max_ms=args.ingress_max_ms,
        audio_timeout_ms=args.audio_timeout_ms,
        partial_max_backlog_ms=args.partial_max_backlog_ms,
        max_consecutive_errors=args.asr_max_consecutive_errors,
        punct_workers=args.punct_workers,
        punct_partial_timeout_ms=args.punct_partial_timeout_ms,
        punct_commit_timeout_ms=args.punct_commit_timeout_ms)
    server = grpc.aio.server(
        maximum_concurrent_rpcs=args.max_streams,
        options=[('grpc.max_receive_message_length', 2 * 1024 * 1024),
                 ('grpc.keepalive_time_ms', 30000),
                 ('grpc.keepalive_timeout_ms', 10000)])
    from .proto import speech_pb2_grpc
    from .speech_service import SpeechStreamService
    speech_pb2_grpc.add_SpeechStreamServicer_to_server(
        SpeechStreamService(service, use_language_source=args.use_language_source), server)
    addr = f'{args.host}:{args.port}'
    if server.add_insecure_port(addr) == 0:
        raise RuntimeError(f'failed to bind {addr}')
    await server.start()
    LOG.info('listening on %s; plaintext for lab only', addr)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        service.close()
