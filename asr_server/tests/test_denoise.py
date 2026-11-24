"""noise_reduction audio path (denoise.py) and the G.711 mu-law encoder, with a stand-in model.

FakeDenoiser behaves like DPDFNet-2 under sherpa-onnx 2.0.2 as measured on the real model:
it holds back one 10 ms hop (160 samples), flush() returns what it holds, and the number of
samples out equals the number in. It also scales the audio so that the tests can tell
processed audio from the input.
"""
import numpy as np
import pytest

from asr_server.audio import Downsampler16kTo8k, decode_audio, mulaw_encode
from asr_server.denoise import OUTPUT_PCM16, OUTPUT_PCMU, DenoiserPool, NoiseStream
from asr_server.messages import PCMU

RATE = 16000
HOP = 160


class _Result:
    def __init__(self, samples):
        self.samples = samples
        self.sample_rate = RATE


class FakeDenoiser:
    sample_rate = RATE
    frame_shift_in_samples = HOP

    def __init__(self, gain=0.5):
        self.gain = gain
        self.held = np.zeros(0, dtype=np.float32)
        self.runs = self.flushes = self.resets = 0

    def run(self, samples, sample_rate):
        assert sample_rate == RATE
        self.runs += 1
        data = np.concatenate([self.held, np.asarray(samples, dtype=np.float32)])
        keep = min(HOP, len(data))
        self.held = data[len(data) - keep:]
        return _Result((data[:len(data) - keep] * self.gain).tolist())

    def flush(self):
        self.flushes += 1
        out, self.held = self.held, np.zeros(0, dtype=np.float32)
        return _Result((out * self.gain).tolist())

    def reset(self):
        self.resets += 1
        self.held = np.zeros(0, dtype=np.float32)


def tone(seconds, freq=440.0, level=0.25):
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm16(samples):
    return (np.clip(np.round(np.asarray(samples) * 32768), -32768, 32767).astype('<i2')).tobytes()


def split(data, chunk_ms):
    size = RATE * chunk_ms // 1000 * 2
    return [data[p:p + size] for p in range(0, len(data), size)]


def run_stream(data, chunk_ms=80, **options):
    stream = NoiseStream(FakeDenoiser(), **options)
    per_chunk = [stream.process(piece) for piece in split(data, chunk_ms)]
    return stream, per_chunk, stream.finish()


# ------------------------------------------------------------------ mu-law encoder
def test_mulaw_encode_known_codes():
    values = np.array([0, -1, 32767, -32768, 32124, -32124], dtype=np.int16)
    assert list(mulaw_encode(values)) == [0xFF, 0x7F, 0x80, 0x00, 0x80, 0x00]


def test_mulaw_encode_inverts_the_decoder():
    codes = np.array([c for c in range(256) if c != 0x7F], dtype=np.uint8)   # 0x7F and 0xFF both mean 0
    decoded = (decode_audio(codes.tobytes(), PCMU) * 32768).astype(np.int16)
    assert list(mulaw_encode(decoded)) == list(codes)


def test_mulaw_encode_saturates_wide_integers_and_refuses_floats():
    # review L2: int32 values used to wrap (40000 -> a loud negative code), floats became 0
    assert list(mulaw_encode(np.array([40000, -40000], dtype=np.int32))) == [0x80, 0x00]
    with pytest.raises(TypeError):
        mulaw_encode(np.array([0.5], dtype=np.float32))


def test_mulaw_encode_error_stays_within_one_step():
    values = np.random.default_rng(0).integers(-32768, 32768, 20000).astype(np.int16)
    back = decode_audio(mulaw_encode(values), PCMU) * 32768
    err = np.abs(back - values.astype(np.float64))
    assert np.all(err <= np.maximum(16, np.abs(values.astype(np.float64)) / 16))


# ---------------------------------------------------------------------- PCM16 path
@pytest.mark.parametrize('frame_ms', [20, 40, 80])
def test_pcm16_frames_have_the_configured_size_and_carry_all_audio(frame_ms):
    data = pcm16(tone(1.03))                         # not a whole number of frames
    stream, per_chunk, tail = run_stream(data, output=OUTPUT_PCM16, frame_ms=frame_ms)
    frames = [f for chunk in per_chunk for f in chunk] + tail
    size = 640 * frame_ms // 20
    assert all(len(f) == size for f in frames[:-1])
    assert 0 < len(frames[-1]) <= size and len(frames[-1]) % 2 == 0
    out = np.frombuffer(b''.join(frames), dtype='<i2')
    expected = np.frombuffer(pcm16(np.frombuffer(data, dtype='<i2') / 32768 * 0.5), dtype='<i2')
    assert np.array_equal(out, expected)             # same samples, no zero inserted, nothing lost
    assert (stream.samples_in, stream.samples_out) == (len(expected), len(expected))


def test_every_80ms_chunk_returns_audio():
    _, per_chunk, _ = run_stream(pcm16(np.zeros(RATE * 2, dtype=np.float32)), output=OUTPUT_PCM16,
                                 frame_ms=20)
    # the first chunk gives 1280 - 160 held samples -> 3 frames; the next ones 4
    assert [len(c) for c in per_chunk] == [3] + [4] * (len(per_chunk) - 1)


@pytest.mark.parametrize('chunk_ms', [19, 20, 21, 79, 81])
def test_output_does_not_depend_on_the_input_chunk_size(chunk_ms):
    data = pcm16(tone(0.7))
    ref = run_stream(data, 80, output=OUTPUT_PCM16, frame_ms=20)
    other = run_stream(data, chunk_ms, output=OUTPUT_PCM16, frame_ms=20)

    def joined(result):
        return b''.join(f for chunk in result[1] for f in chunk) + b''.join(result[2])
    assert joined(other) == joined(ref)


@pytest.mark.parametrize('output,sent', [(OUTPUT_PCM16, 1280), (OUTPUT_PCMU, 640)])
def test_loud_output_is_clipped_and_counted_in_sent_samples(output, sent):
    # review L2: in PCMU mode `clipped` counted 8 kHz samples next to 16 kHz totals
    stream = NoiseStream(FakeDenoiser(gain=4.0), output=output, frame_ms=20)
    frames = stream.process(pcm16(np.full(1280, 0.5, dtype=np.float32))) + stream.finish()
    assert stream.samples_sent == sent
    assert stream.clipped >= sent - 70           # all but the filter's ramp at the edges (PCMU)
    if output == OUTPUT_PCM16:
        assert np.frombuffer(b''.join(frames), dtype='<i2').max() == 32767 and stream.clipped == sent


def test_empty_chunk_does_not_run_the_model():
    model = FakeDenoiser()
    stream = NoiseStream(model, output=OUTPUT_PCM16, frame_ms=20)
    assert stream.process(b'') == [] and model.runs == 0


def test_finish_flushes_the_model_once_and_closes_the_stream():
    model = FakeDenoiser()
    stream = NoiseStream(model, output=OUTPUT_PCM16, frame_ms=20)
    stream.process(pcm16(tone(0.08)))
    tail = stream.finish()
    assert model.flushes == 1 and b''.join(tail)
    with pytest.raises(RuntimeError):
        stream.finish()
    with pytest.raises(RuntimeError):
        stream.process(pcm16(tone(0.02)))


def test_finish_without_audio_returns_nothing():
    assert NoiseStream(FakeDenoiser(), output=OUTPUT_PCM16, frame_ms=20).finish() == []


# -------------------------------------------------------- 16 kHz -> 8 kHz for PCMU
def downsample(x, chunk=1280):
    d = Downsampler16kTo8k()
    parts = [d.push(x[p:p + chunk]) for p in range(0, len(x), chunk)]
    return parts, d.flush()


def test_downsampler_returns_audio_for_every_chunk_and_all_of_it_at_the_end():
    # soxr streams give their output in bursts (0 samples for some 80 ms chunks): not usable here
    parts, tail = downsample(np.zeros(1280 * 10, dtype=np.float32))
    assert all(len(p) >= 480 for p in parts[:1]) and [len(p) for p in parts[1:]] == [640] * 9
    assert sum(len(p) for p in parts) + len(tail) == 1280 * 10 // 2


@pytest.mark.parametrize('chunk', [304, 336, 1264, 1296, 1])
def test_downsampler_output_does_not_depend_on_the_chunk_size(chunk):
    x = tone(0.5) + tone(0.5, freq=3000.0, level=0.1)
    ref = np.concatenate([*downsample(x)[0], downsample(x)[1]])
    parts, tail = downsample(x, chunk)
    assert np.allclose(np.concatenate([*parts, tail]), ref, atol=1e-6)


@pytest.mark.parametrize('freq,gain', [(440, 1.0), (1000, 1.0), (3000, 1.0), (3300, 1.0),
                                       (4500, 0.0), (6000, 0.0), (7500, 0.0)])
def test_downsampler_keeps_the_telephone_band_and_removes_what_would_alias(freq, gain):
    parts, tail = downsample(tone(1.0, freq=freq, level=0.5))
    out = np.concatenate([*parts, tail])[800:7200]
    rms = np.sqrt(np.mean(out ** 2)) / (0.5 / np.sqrt(2))
    assert abs(rms - gain) < (0.02 if gain else 0.001)       # passband within 2 %, stopband -60 dB


def test_downsampler_flush_starts_a_new_stream():
    # review L2: a second flush() returned filter ringing past the end of the audio
    x = tone(0.2)
    d = Downsampler16kTo8k()
    d.push(x)
    d.flush()
    assert len(d.flush()) == 0
    assert np.array_equal(d.push(x), Downsampler16kTo8k().push(x))


def test_downsampler_adds_no_delay():
    x = np.zeros(4000, dtype=np.float32)
    x[1000] = 1.0
    parts, tail = downsample(x, 320)
    assert int(np.argmax(np.concatenate([*parts, tail]))) == 500


# ----------------------------------------------------------------------- PCMU path
def test_pcmu_frames_are_8khz_mulaw_of_the_processed_audio():
    data = pcm16(tone(1.0, freq=440.0, level=0.5))
    stream, per_chunk, tail = run_stream(data, output=OUTPUT_PCMU, frame_ms=20)
    frames = [f for chunk in per_chunk for f in chunk] + tail
    assert all(len(f) == 160 for f in frames[:-1]) and 0 < len(frames[-1]) <= 160
    assert all(chunk for chunk in per_chunk)                     # audio back for every 80 ms chunk
    out = decode_audio(b''.join(frames), PCMU)
    assert len(out) == RATE // 2                                 # 1 s at 8 kHz
    middle = out[1000:7000]
    assert abs(np.sqrt(np.mean(middle ** 2)) - 0.25 / np.sqrt(2)) < 0.01    # gain 0.5 applied
    spectrum = np.abs(np.fft.rfft(middle))
    assert abs(np.argmax(spectrum) * 8000 / len(middle) - 440) < 5


def test_pcmu_frame_ms_scales_the_frame_size():
    _, per_chunk, tail = run_stream(pcm16(tone(0.5)), output=OUTPUT_PCMU, frame_ms=40)
    frames = [f for chunk in per_chunk for f in chunk] + tail
    assert all(len(f) == 320 for f in frames[:-1])


def test_unknown_output_or_frame_size_is_rejected():
    with pytest.raises(ValueError):
        NoiseStream(FakeDenoiser(), output='opus', frame_ms=20)
    with pytest.raises(ValueError):
        NoiseStream(FakeDenoiser(), output=OUTPUT_PCM16, frame_ms=30)


# ---------------------------------------------------------------------------- pool
def test_pool_loads_every_denoiser_up_front_and_never_waits():
    made = []

    def factory():
        made.append(FakeDenoiser())
        return made[-1]
    pool = DenoiserPool(factory, 2)
    assert len(made) == 2 and pool.free == 2
    a, b = pool.acquire(), pool.acquire()
    assert {id(a), id(b)} == {id(d) for d in made}
    assert pool.acquire() is None and pool.free == 0


def test_pool_release_resets_the_denoiser():
    pool = DenoiserPool(FakeDenoiser, 1)
    d = pool.acquire()
    d.run(np.ones(400, dtype=np.float32), RATE)
    assert pool.release(d) is True
    assert d.resets == 1 and len(d.held) == 0 and pool.free == 1
    assert pool.acquire() is d


class BadResetDenoiser(FakeDenoiser):
    def reset(self):
        raise RuntimeError('reset failed')


def test_pool_drops_a_denoiser_whose_reset_fails_and_replace_loads_a_new_one():
    # review L2/L3 (important): the denoiser was lost for good, the pool shrank by one
    made = []

    def factory():
        made.append(BadResetDenoiser())
        return made[-1]
    pool = DenoiserPool(factory, 2)
    d = pool.acquire()
    assert pool.release(d) is False and pool.free == 1
    pool.replace()
    assert pool.free == 2 and len(made) == 3 and d not in pool._free


def test_pool_drops_a_broken_denoiser_without_resetting_it():
    pool = DenoiserPool(FakeDenoiser, 1)
    d = pool.acquire()
    assert pool.release(d, broken=True) is False and d.resets == 0 and pool.free == 0
    pool.replace()
    assert pool.free == 1 and pool.acquire() is not d


def test_pool_ignores_a_second_release_and_a_foreign_denoiser():
    # review L2: a double release put one stateful denoiser in the pool twice
    pool = DenoiserPool(FakeDenoiser, 1)
    d = pool.acquire()
    pool.release(d)
    pool.release(d)
    pool.release(FakeDenoiser())
    assert pool.free == 1


def test_pool_replace_survives_a_failing_factory():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('model file gone')
        return FakeDenoiser()
    pool = DenoiserPool(factory, 1)
    pool.release(pool.acquire(), broken=True)
    pool.replace()
    assert pool.free == 0
