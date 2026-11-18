"""G.711 / PCM decoding and streaming sample-rate conversion.

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
