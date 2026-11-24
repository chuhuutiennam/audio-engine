"""DPDFNet-2 speech denoising: the task noise_reduction and the speech_to_text preprocessing.

Design: docs/DPDFNet2_Task_Routing_Design_VI.md (audio-engine repo root), sections 5 and 7.

- DenoiserPool: every denoiser is loaded at start. Loading one took 0.24-0.36 s on a laptop
  CPU, and the gateway gives up on a stream that returns no audio for 500 ms. A stream borrows
  one and gives it back reset.
- DPDFNet2Processor: the model path of one stream, shared by both uses. In and out: 16 kHz
  audio. align=True (speech_to_text, stt_denoise.py) removes the model's delay so that VAD and
  ASR keep the input's timeline.
- NoiseStream: the audio path of one noise_reduction stream. In: PCM16 LE mono 16 kHz. Out:
  frames of frame_ms, either PCM16 LE mono 16 kHz or G.711 PCMU 8 kHz
  (AudioChunk.output_audio_format).
Their calls block while the model runs: the gRPC layer runs them off the event loop, one at a
time per stream.

The model (sherpa_onnx.OnlineSpeechDenoiser, or a stand-in in the tests) needs
run(samples, sample_rate), flush() and reset(); run() and flush() return an object with
.samples and .sample_rate. Measured on DPDFNet-2 with sherpa-onnx 2.0.2: run() returns whole
10 ms hops and keeps the last one back, flush() returns what is left, so the samples out add up
to the samples in; the audio comes out MODEL_DELAY samples (40 ms) late, whatever the chunk size.
"""
import logging
import os
import threading

import numpy as np

from .audio import Downsampler16kTo8k, decode_audio, mulaw_encode
from .messages import PCM_S16LE

LOG = logging.getLogger('qwen3_asr.denoise')

SAMPLE_RATE = 16000
OUTPUT_PCM16 = 'pcm_s16le'     # PCM16 LE mono 16 kHz: 640 bytes per 20 ms
OUTPUT_PCMU = 'pcmu'           # G.711 mu-law 8 kHz: 160 bytes per 20 ms
FRAME_MS = (20, 40, 80)        # the gateway wants multiples of 20 ms
MODEL_DELAY = 640              # samples (40 ms): output n carries input n - 640 (cross-correlation)


def dpdfnet2_factory(model_path, num_threads=1, setting='noise_reduction'):
    """Factory of DPDFNet-2 denoisers: sherpa_onnx.OnlineSpeechDenoiser on CPU, 16 kHz.

    Checks the model file and the package now, so that a bad setup stops the start before any
    model loads. Measured with sherpa-onnx 2.0.2 and dpdfnet2.onnx (SHA-256 ce35d602...87a9).
    `setting`: the config section the errors name (noise_reduction or asr_denoise).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'missing local DPDFNet-2 model ({setting}.model): {model_path}')
    try:
        import sherpa_onnx
    except ImportError as exc:              # not installed, or its native library does not load
        raise ImportError(f'{setting}.enabled needs the sherpa-onnx package '
                          f'(sherpa-onnx==2.0.2, see requirements.txt); importing it failed: {exc}') from exc
    config = sherpa_onnx.OnlineSpeechDenoiserConfig(model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(model=model_path),
        num_threads=num_threads, provider='cpu', debug=False))
    if not config.validate():
        raise ValueError(f'sherpa-onnx rejected the DPDFNet-2 configuration for {model_path}')

    def make():
        denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
        if denoiser.sample_rate != SAMPLE_RATE:
            raise ValueError(f'{model_path} is a {denoiser.sample_rate} Hz model; '
                             f'{setting} needs {SAMPLE_RATE} Hz')
        return denoiser
    return make


def check_model_delay(factory):
    """Raise ValueError unless a fresh denoiser from `factory` delays audio by MODEL_DELAY.

    align=True relies on that number; another model file or sherpa-onnx version could differ
    and would put VAD/ASR off the input timeline with no error. A lone click is fed in 20 ms
    chunks; its peak comes out MODEL_DELAY samples later (the model removes most of the click,
    so the float output is read, not PCM16). Blocks for one model load and 1 s of audio.
    """
    denoiser = factory()
    x = np.zeros(SAMPLE_RATE, dtype=np.float32)
    x[1000] = 0.5
    y = np.concatenate([np.asarray(denoiser.run(x[p:p + 320], SAMPLE_RATE).samples, dtype=np.float32)
                        for p in range(0, len(x), 320)] + [np.asarray(denoiser.flush().samples, dtype=np.float32)])
    delay = int(np.argmax(np.abs(y))) - 1000
    if delay != MODEL_DELAY:
        raise ValueError(f'the denoiser delays audio by {delay} samples, not {MODEL_DELAY}: '
                         'asr_denoise supports DPDFNet-2 dpdfnet2.onnx with sherpa-onnx 2.0.2 only')
    return delay


def _to_pcm16(samples):
    """float [-1, 1] -> int16, saturating; returns (samples, how many were clipped)."""
    scaled = np.round(np.asarray(samples, dtype=np.float32) * 32768.0)
    clipped = int(np.count_nonzero((scaled > 32767) | (scaled < -32768)))
    return np.clip(scaled, -32768, 32767).astype(np.int16), clipped


class DPDFNet2Processor:
    """DPDFNet-2 on one stream: 16 kHz mono in, 16 kHz mono out. Not thread-safe: one call at a time.

    run(samples) takes float audio in [-1, 1] and returns the float samples ready now; finish()
    returns the rest, once. feed(pcm) / flush() do the same with PCM16 LE bytes. The model keeps
    its state from call to call; nothing here resets it.

    align=False: the model's output as it is (noise_reduction).
    align=True: output sample k is input sample k, denoised. The first MODEL_DELAY samples the
      model returns are dropped, and finish() feeds MODEL_DELAY samples of silence before the
      model flush, so the output has exactly as many samples as the input, on its timeline
      (VAD and ASR timestamps stay right). The input not yet returned is kept for a model that
      would return too few samples at the end (`filled` counts those).

    When the model fails, the caller goes on without it: stt_denoise keeps its own copy of the
    input not yet returned, because a slow model call may still be using this object.
    """

    def __init__(self, denoiser, *, align=False):
        self.denoiser = denoiser
        self.align = align
        self._skip = MODEL_DELAY if align else 0        # model output still to drop
        self._pending = np.zeros(0, dtype=np.float32)   # align: input not yet returned
        self.finished = False
        self.samples_in = 0
        self.samples_out = 0
        self.filled = 0                 # samples the model did not return, taken from the input
        self.clipped = 0                # in feed() / flush()

    def _model(self, result):
        if result.sample_rate != SAMPLE_RATE:
            raise RuntimeError(f'denoiser returned {result.sample_rate} Hz audio, expected {SAMPLE_RATE}')
        out = np.asarray(result.samples, dtype=np.float32).reshape(-1)
        if self._skip:
            drop = min(self._skip, len(out))
            self._skip -= drop
            out = out[drop:]
        return out

    def _ready(self, out):
        if self.align:
            self._pending = self._pending[len(out):]
        self.samples_out += len(out)
        return out

    def run(self, samples):
        if self.finished:
            raise RuntimeError('denoiser stream already finished')
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.samples_in += len(samples)
        if self.align:
            self._pending = np.concatenate([self._pending, samples])
        if not len(samples):
            return samples
        return self._ready(self._model(self.denoiser.run(samples, SAMPLE_RATE)))

    def finish(self):
        """End of the stream: the rest of the audio. The model is flushed once. When the model
        fails here the processor stays open (finished is set only after the model calls)."""
        if self.finished:
            raise RuntimeError('denoiser stream already finished')
        parts = []
        if self.align:
            parts.append(self._model(self.denoiser.run(np.zeros(MODEL_DELAY, dtype=np.float32), SAMPLE_RATE)))
        parts.append(self._model(self.denoiser.flush()))
        self.finished = True
        out = np.concatenate(parts)
        if self.align:
            missing = len(self._pending) - len(out)
            if missing > 0:             # never fewer samples than in: the ASR timeline would drift
                LOG.warning('denoiser returned %d samples fewer than it was given; the input fills them',
                            missing)
                self.filled = missing
                out = np.concatenate([out, self._pending[len(out):]])
            out = out[:len(self._pending)]
        return self._ready(out)

    def feed(self, pcm):
        """PCM16 LE mono 16 kHz in and out."""
        return self._pcm16(self.run(decode_audio(pcm, PCM_S16LE)))

    def flush(self):
        return self._pcm16(self.finish())

    def _pcm16(self, samples):
        pcm, clipped = _to_pcm16(samples)
        self.clipped += clipped
        return pcm.astype('<i2').tobytes()


class NoiseStream:
    """Audio path of one noise_reduction stream. Not thread-safe: one call at a time."""

    def __init__(self, denoiser, *, output, frame_ms):
        if output not in (OUTPUT_PCM16, OUTPUT_PCMU):
            raise ValueError(f'unknown output {output!r}')
        if frame_ms not in FRAME_MS:
            raise ValueError(f'frame_ms must be one of {FRAME_MS}, got {frame_ms}')
        self.denoiser = denoiser
        self._model = DPDFNet2Processor(denoiser, align=False)   # the model's output as it is
        self.output = output
        if output == OUTPUT_PCMU:
            # One stateful resampler for the whole stream: no discontinuity between chunks.
            self._downsampler = Downsampler16kTo8k()
            self.frame_bytes = 8 * frame_ms                     # 8000 Hz, one byte per sample
        else:
            self._downsampler = None
            self.frame_bytes = SAMPLE_RATE * frame_ms // 1000 * 2
        self._pending = bytearray()                             # encoded audio not yet framed
        self.samples_in = 0             # 16 kHz samples given to the model
        self.samples_out = 0            # 16 kHz samples the model returned
        self.samples_sent = 0           # samples encoded for the client (8 kHz ones for PCMU)
        self.clipped = 0                # of samples_sent, those saturated to 16 bits
        self.finished = False

    def process(self, pcm):
        """Denoise one chunk of PCM16 LE 16 kHz; returns the whole frames now ready."""
        if self.finished:
            raise RuntimeError('noise stream already finished')
        if not pcm:
            return []
        samples = decode_audio(pcm, PCM_S16LE)
        self.samples_in += len(samples)
        self._add(self._model.run(samples), last=False)
        return self._frames()

    def finish(self):
        """End of the stream: flush the model once; returns the rest, the last frame may be short."""
        if self.finished:
            raise RuntimeError('noise stream already finished')
        self.finished = True
        self._add(self._model.finish(), last=True)
        frames = self._frames()
        if self._pending:
            frames.append(bytes(self._pending))
            self._pending.clear()
        return frames

    def _add(self, samples, *, last):
        self.samples_out += len(samples)
        if self._downsampler is not None:
            samples = self._downsampler.push(samples)
            if last:
                samples = np.concatenate([samples, self._downsampler.flush()])
        pcm, clipped = _to_pcm16(samples)
        self.samples_sent += len(pcm)
        self.clipped += clipped
        self._pending += mulaw_encode(pcm) if self._downsampler is not None else pcm.astype('<i2').tobytes()

    def _frames(self):
        size = len(self._pending) // self.frame_bytes * self.frame_bytes
        if not size:
            return []
        block = bytes(self._pending[:size])
        del self._pending[:size]
        return [block[p:p + self.frame_bytes] for p in range(0, size, self.frame_bytes)]


class DenoiserPool:
    """Denoisers loaded once at start; a stream borrows one and never waits for one.

    acquire() and release() run on the event loop; replace() loads a model (about 0.3 s) and
    runs on a worker thread, hence the lock.
    """

    def __init__(self, factory, size):
        self.size = size
        self._factory = factory
        self._lock = threading.Lock()
        self._free = [factory() for _ in range(size)]
        self._lent = set()              # id() of the denoisers lent out

    @property
    def free(self):
        with self._lock:
            return len(self._free)

    def acquire(self):
        """A denoiser, or None when all are in use: the caller refuses the stream."""
        with self._lock:
            if not self._free:
                return None
            denoiser = self._free.pop()
            self._lent.add(id(denoiser))
            return denoiser

    def release(self, denoiser, broken=False):
        """Take back a lent denoiser that no model call uses any more, reset for the next stream.

        Returns False when the denoiser is dropped instead, because it is `broken` (its model
        call failed) or its reset() failed: the caller then runs replace() off the event loop.
        """
        with self._lock:
            if id(denoiser) not in self._lent:
                LOG.error('denoiser given back twice or not from this pool; ignored')
                return True
            self._lent.discard(id(denoiser))
        if not broken:
            try:
                denoiser.reset()
            except Exception as exc:
                LOG.error('denoiser reset failed (%s: %s); a new one is loaded', type(exc).__name__, exc)
                broken = True
        if broken:
            return False
        with self._lock:
            self._free.append(denoiser)
        return True

    def replace(self):
        """Load a new denoiser in place of a dropped one. Blocks: run it off the event loop."""
        try:
            denoiser = self._factory()
        except Exception:
            LOG.exception('could not load a denoiser in place of a dropped one; the pool is one short')
            return
        with self._lock:
            self._free.append(denoiser)
