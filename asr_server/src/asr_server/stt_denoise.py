"""speech_to_text preprocessing (asr_denoise): DPDFNet-2 in front of VAD and ASR.

SpeechStreamService passes the StreamRequests of a speech_to_text stream through stream(),
which denoises their audio with denoise.DPDFNet2Processor(align=True): as many samples as came
in, on the input's timeline, so VAD, ASR, commits and timestamps work as without it. The model
keeps its state for the whole stream and is flushed once, at its end.
Design: docs/DPDFNet2_Task_Routing_Design_VI.md (audio-engine repo root).

Which streams: apply_to 'wideband' (default) denoises only the streams the gateway carries as
G.722 or L16 (first chunk: output_audio_format PCM_S16LE, sample_rate 16000): on band-limited
G.711 audio DPDFNet-2 turned down much of the speech. 'all': every 16 kHz stream. Never 8 kHz.

Audio is never lost. A stream that finds no denoiser free is not denoised. When the model fails,
one model call takes longer than max_backlog_ms, or more than max_backlog_ms of audio waits for
the model, the stream goes on with the audio as sent from the next sample on: the input the
model has not given back yet comes first (kept here, not in the processor, which a slow call may
still be using). One WARNING per fallback, one INFO line per stream, counters in `stats`.
"""
import asyncio
import collections
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from . import messages as m
from .denoise import SAMPLE_RATE, DPDFNet2Processor
from .noise_service import _Histogram
from .proto import speech_pb2 as pb

LOG = logging.getLogger('qwen3_asr.stt_denoise')

BYTES_PER_MS = SAMPLE_RATE * 2 / 1000       # PCM16 mono
_EOF = object()
_STATS_KEY = {'model_error': 'fallback_error', 'model_slow': 'fallback_slow', 'backlog': 'fallback_backlog'}


class _Slow(Exception):
    """A model call took longer than max_backlog_ms; it goes on, its result is dropped."""


class _Run:
    """One denoised stream."""

    def __init__(self, owner, denoiser, session_id, stream_id):
        self.owner = owner
        self.denoiser = denoiser
        self.session_id = session_id
        self.stream_id = stream_id
        self.processor = DPDFNet2Processor(denoiser, align=True)
        self.raw = bytearray()          # input passed to the model, not given back yet
        self.fallback = None            # reason, once the stream goes on with the input as sent
        self.job = None                 # the model call on a worker thread
        self.broken = False             # the model failed: the pool drops this denoiser
        self.outcome = 'cancelled'
        self.seq = 0                    # StreamRequest audio sequence (the core wants 1, 2, 3...)
        self.chunks = 0
        self.samples_in = 0
        self.samples_out = 0
        self.queued_ms = 0.0            # audio read from the stream, not given to the model yet
        self.backlog_ms_max = 0.0
        self.model_ms = _Histogram()

    def request(self, pcm):
        self.seq += 1
        self.samples_out += len(pcm) // 2
        return m.StreamRequest(audio=m.AudioChunk(data=pcm, sequence=self.seq))

    async def call(self, fn, *args):
        def timed():
            started = time.perf_counter()
            return fn(*args), time.perf_counter() - started
        self.job = self.owner._executor.submit(timed)     # RuntimeError once closed: a model error
        future = asyncio.wrap_future(self.job)
        limit = self.owner.max_backlog_ms
        if limit:
            done, _ = await asyncio.wait({future}, timeout=limit / 1000)
            if not done:
                future.add_done_callback(lambda f: f.cancelled() or f.exception())   # nobody waits
                raise _Slow()
        result, seconds = await future
        self.model_ms.add(1000 * seconds)
        return result

    def go_raw(self, reason, detail=''):
        """From the next sample on the stream gets the input as sent; returns the input held back."""
        self.fallback = reason
        self.owner.stats[_STATS_KEY[reason]] += 1
        LOG.warning('stt_denoise fallback session=%s stream=%s reason=%s at_sample=%d%s: the stream '
                    'goes on with the audio as sent', self.session_id, self.stream_id, reason,
                    self.samples_out, f' ({detail})' if detail else '')
        out = bytes(self.raw)
        self.raw.clear()
        return out

    async def _model(self, fn, *args):
        try:
            return await self.call(fn, *args), None
        except _Slow:
            return None, self.go_raw('model_slow', f'one call over {self.owner.max_backlog_ms} ms')
        except Exception as exc:
            self.broken = True
            return None, self.go_raw('model_error', f'{type(exc).__name__}: {exc}')

    async def audio(self, pcm):
        self.chunks += 1
        self.samples_in += len(pcm) // 2
        if self.fallback:
            return pcm
        self.raw += pcm
        limit = self.owner.max_backlog_ms
        if limit and self.queued_ms > limit:
            return self.go_raw('backlog', f'{self.queued_ms:.0f} ms waiting')
        out, raw = await self._model(self.processor.feed, pcm)
        if out is None:
            return raw
        del self.raw[:len(out)]
        return out

    async def finish(self):
        if self.fallback:
            return b''                  # everything was passed on as it came
        out, raw = await self._model(self.processor.flush)
        if out is None:
            return raw
        if len(out) != len(self.raw):   # DPDFNet2Processor returns exactly the rest
            self.broken = True
            return self.go_raw('model_error', f'{len(out)} bytes at the end for {len(self.raw)}')
        self.raw.clear()
        return out


class SttDenoiser:
    def __init__(self, pool, *, workers=2, max_backlog_ms=500, apply_to='wideband'):
        if apply_to not in ('wideband', 'all'):
            raise ValueError(f"apply_to must be 'wideband' or 'all', got {apply_to!r}")
        self.pool = pool
        self.max_backlog_ms = max_backlog_ms
        self.apply_to = apply_to
        self.stats = collections.Counter()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='stt-denoise')
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stt-denoise-load')

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._loader.shutdown(wait=False, cancel_futures=True)

    def applies(self, chunk):
        """Whether the stream that starts with AudioChunk `chunk` is denoised (apply_to)."""
        if chunk.sample_rate == SAMPLE_RATE and (self.apply_to == 'all'
                                                 or chunk.output_audio_format == pb.PCM_S16LE):
            return True
        self.stats['narrowband'] += 1
        return False

    async def stream(self, requests, session_id, stream_id):
        """The StreamRequests of `requests` with their audio denoised (see the module doc)."""
        denoiser = self.pool.acquire()
        if denoiser is None:
            self.stats['no_denoiser'] += 1
            LOG.warning('stt_denoise: no denoiser free (asr_denoise.max_streams=%d) session=%s stream=%s; '
                        'the stream is recognised from the audio as sent', self.pool.size, session_id, stream_id)
            async for req in requests:
                yield req
            return
        run = _Run(self, denoiser, session_id, stream_id)
        queue = asyncio.Queue()

        async def reader():
            """Reads as the audio comes, so that the audio waiting for the model can be measured."""
            try:
                async for req in requests:
                    ms = len(req.audio.data) / BYTES_PER_MS if req.WhichOneof('payload') == 'audio' else 0.0
                    run.queued_ms += ms
                    run.backlog_ms_max = max(run.backlog_ms_max, run.queued_ms)
                    queue.put_nowait((req, ms))
                queue.put_nowait(_EOF)
            except asyncio.CancelledError:
                raise
            except Exception as exc:    # a protocol error of the stream: raised to the core as it is
                queue.put_nowait(exc)

        task = asyncio.ensure_future(reader())
        try:
            while True:
                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                if item is _EOF or item[0].WhichOneof('payload') == 'end_stream':
                    tail = await run.finish()
                    if tail:
                        yield run.request(tail)
                    run.outcome = 'fallback' if run.fallback else 'denoised'
                    if not run.fallback:
                        self.stats['denoised'] += 1
                    if item is not _EOF:
                        yield item[0]
                    return
                req, ms = item
                run.queued_ms -= ms
                if req.WhichOneof('payload') != 'audio':
                    yield req
                    continue
                out = await run.audio(req.audio.data)
                if out:
                    yield run.request(out)
        finally:
            task.cancel()
            self._release(run)
            self._summary(run)

    def _release(self, run):
        """Give the denoiser back once no model call uses it (a slow or cancelled call may still run)."""
        def give_back():
            if not self.pool.release(run.denoiser, broken=run.broken):
                try:
                    self._loader.submit(self.pool.replace)
                except RuntimeError:        # closed: the server is stopping
                    pass
        job = run.job
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

    @staticmethod
    def _summary(run):
        LOG.info('stt_denoise stream ended session=%s stream=%s outcome=%s reason=%s chunks=%d '
                 'samples_in=%d samples_out=%d model_ms_p50=%d model_ms_p95=%d model_ms_max=%.1f '
                 'backlog_ms_max=%d', run.session_id, run.stream_id, run.outcome, run.fallback or '-',
                 run.chunks, run.samples_in, run.samples_out, run.model_ms.percentile(50),
                 run.model_ms.percentile(95), run.model_ms.max, run.backlog_ms_max)
