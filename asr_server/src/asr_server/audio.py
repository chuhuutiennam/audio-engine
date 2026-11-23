"""G.711 / PCM decoding, G.711 mu-law encoding and streaming sample-rate conversion.

All incoming frames are RTP payload bytes (not complete RTP datagrams).
"""
import numpy as np
import soxr


def _alaw_decode(values: np.ndarray) -> np.ndarray:
    a = np.bitwise_xor(values.astype(np.int32), 0x55)
    segment = (a & 0x70) >> 4
    magnitude = ((a & 0x0F) << 4) + 8
    magnitude = np.where(segment >= 1, magnitude + 0x100, magnitude)
    magnitude = np.where(segment > 1, magnitude << np.maximum(segment - 1, 0), magnitude)
    return np.where((a & 0x80) != 0, magnitude, -magnitude).astype(np.int16)


def _mulaw_decode(values: np.ndarray) -> np.ndarray:
    u = np.bitwise_xor(values.astype(np.int32), 0xFF)
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return np.where((u & 0x80) != 0, 0x84 - t, t - 0x84).astype(np.int16)


_ALAW_TABLE = _alaw_decode(np.arange(256, dtype=np.uint8))
_MULAW_TABLE = _mulaw_decode(np.arange(256, dtype=np.uint8))


def _mulaw_encode(values: np.ndarray) -> np.ndarray:
    """ITU-T G.711 mu-law of 16-bit samples (bias 0x84, clip 32635), as the classic g711.c."""
    x = values.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(x), 32635) + 0x84
    exponent = np.floor(np.log2(magnitude >> 7)).astype(np.int32)     # magnitude >> 7 is 1..255
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


_MULAW_ENCODE_TABLE = _mulaw_encode(np.arange(-32768, 32768))


def mulaw_encode(samples: np.ndarray) -> bytes:
    """16-bit PCM samples -> G.711 mu-law payload (one byte per sample).

    Integer samples only (float audio must be scaled to 16 bits first); values beyond 16 bits
    saturate."""
    samples = np.asarray(samples)
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError(f'mulaw_encode needs integer samples, got {samples.dtype}')
    return _MULAW_ENCODE_TABLE[np.clip(samples.astype(np.int64), -32768, 32767) + 32768].tobytes()


def decode_audio(data: bytes, encoding: int) -> np.ndarray:
    """Decode payload into mono float32 PCM normalized to [-1,1]."""
    if encoding == 1:  # PCMA
        return _ALAW_TABLE[np.frombuffer(data, dtype=np.uint8)].astype(np.float32) / 32768.0
    if encoding == 2:  # PCMU
        return _MULAW_TABLE[np.frombuffer(data, dtype=np.uint8)].astype(np.float32) / 32768.0
    if encoding == 3:  # PCM_S16LE
        if len(data) % 2:
            raise ValueError("PCM_S16LE payload length must be divisible by 2")
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    raise ValueError(f"unsupported audio encoding: {encoding}")


class Downsampler16kTo8k:
    """16 kHz -> 8 kHz for G.711, streaming: output for every chunk, n/2 samples in total for n.

    soxr streams return their output in bursts (nothing at all for some 80 ms chunks), which a
    caller that must return audio for every chunk cannot use. This is a linear-phase FIR
    low-pass (Kaiser, 127 taps, -6 dB at 3650 Hz: flat to 3300 Hz, below -80 dB from 4000 Hz)
    that keeps every second sample. Output sample m is input time 2m (the filter delay is
    compensated), so the last 63 input samples wait for more audio or for flush(): the first
    push of n samples returns about (n - 63) / 2, the next ones n / 2.
    """
    TAPS = 127
    _DELAY = (TAPS - 1) // 2

    def __init__(self):
        n = np.arange(self.TAPS)
        cutoff = 2 * 3650 / 16000
        taps = cutoff * np.sinc(cutoff * (n - self._DELAY)) * np.kaiser(self.TAPS, 7.86)
        self._taps = (taps / taps.sum()).astype(np.float32)
        # Input from x[2m - DELAY] on, m being the next output; the leading zeros stand for the
        # time before the stream.
        self._buf = np.zeros(self._DELAY, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        if len(self._buf) < self.TAPS:
            return np.zeros(0, dtype=np.float32)
        out = np.convolve(self._buf, self._taps, mode='valid')[::2]
        self._buf = self._buf[2 * len(out):]
        return out.astype(np.float32)

    def flush(self) -> np.ndarray:
        """End of the stream: the samples held back, as if silence followed. Afterwards the
        downsampler is back at the start of a new stream."""
        out = self.push(np.zeros(self._DELAY, dtype=np.float32))
        self._buf = np.zeros(self._DELAY, dtype=np.float32)
        return out


class PCM16kConverter:
    """One resampler per utterance (stateful, no chunk-boundary audio discontinuity)."""

    def __init__(self, sample_rate: int):
        if sample_rate not in (8000, 16000):
            raise ValueError("only 8000 and 16000 Hz input is accepted")
        self.sample_rate = sample_rate
        self.resampler = (
            soxr.ResampleStream(sample_rate, 16000, 1, dtype="float32", quality="HQ")
            if sample_rate != 16000 else None
        )

    def push(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if self.resampler is None:
            return samples
        return self.resampler.resample_chunk(samples, last=False)

    def flush(self) -> np.ndarray:
        if self.resampler is None:
            return np.empty(0, dtype=np.float32)
        return self.resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
