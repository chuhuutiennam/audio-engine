"""DPDFNet2Processor (denoise.py): the model path shared by noise_reduction and the speech_to_text
preprocessing, with a stand-in model.

DelayedDenoiser has the timing of DPDFNet-2 under sherpa-onnx 2.0.2 as measured on the real
model (speech, chunks of 20 / 40 / 80 ms): it returns whole 10 ms hops (160 samples) and keeps
the last one back until flush(), so the samples out add up to the samples in, and its audio
comes out 640 samples (40 ms) late. It scales the audio by `gain`, so that the tests can tell
processed audio from the input.
"""
from itertools import zip_longest

import numpy as np
import pytest

from asr_server.denoise import DPDFNet2Processor, MODEL_DELAY, check_model_delay

from test_denoise import FakeDenoiser, RATE, _Result, pcm16, tone

HOP = 160


class DelayedDenoiser(FakeDenoiser):
    def __init__(self, gain=0.5, drop_at_flush=0):
        super().__init__(gain)
        self.drop_at_flush = drop_at_flush      # a model that loses samples at the end
        self.reset()
        self.resets = 0

    def _emit(self, upto):
        out = self.history[self.out:upto] * self.gain
        self.out = upto
        return _Result(out.tolist())

    def run(self, samples, sample_rate):
        assert sample_rate == RATE
        self.runs += 1
        x = np.asarray(samples, dtype=np.float32)
        self.history = np.concatenate([self.history, x])
        self.fed += len(x)
        return self._emit(max(self.out, (self.fed // HOP - 1) * HOP))

    def flush(self):
        self.flushes += 1
        return self._emit(self.fed - self.drop_at_flush)

    def reset(self):
        self.resets += 1
        self.history = np.zeros(MODEL_DELAY, dtype=np.float32)    # output n = input n - 640
        self.out = self.fed = 0


def chunks(x, size):
    return [x[p:p + size] for p in range(0, len(x), size)]


def run_all(processor, x, size):
    out = [processor.run(piece) for piece in chunks(x, size)]
    return np.concatenate(out + [processor.finish()])


def test_model_delay_is_the_measured_one():
    assert MODEL_DELAY == 640           # 40 ms at 16 kHz, cross-correlation on the real model


# ------------------------------------------------------------- not aligned (noise_reduction)
@pytest.mark.parametrize('size', [320, 640, 1280])
def test_unaligned_output_is_the_model_output_as_is(size):
    x = tone(1.0)
    model, reference = DelayedDenoiser(), DelayedDenoiser()
    out = run_all(DPDFNet2Processor(model, align=False), x, size)
    expected = np.concatenate([np.asarray(reference.run(x, RATE).samples), reference.flush().samples])
    assert np.array_equal(out, expected.astype(np.float32))
    assert (model.flushes, model.fed) == (1, len(x))              # no padding without align


# ------------------------------------------------------------------ aligned (ASR preprocessing)
@pytest.mark.parametrize('size', [320, 640, 1280, 208, 1])   # 20 / 40 / 80 ms, 13 ms, one sample
def test_aligned_output_is_on_the_input_timeline(size):
    x = tone(0.7)
    processor = DPDFNet2Processor(DelayedDenoiser(gain=0.5), align=True)
    out = run_all(processor, x, size)
    assert len(out) == len(x)
    assert np.allclose(out, 0.5 * x, atol=1e-7)                   # sample k is input sample k
    assert (processor.samples_in, processor.samples_out) == (len(x), len(x))


def test_aligned_output_comes_as_the_model_returns_it():
    """The model returns nothing for the first 40 ms, then hop by hop: the processor passes on
    only what the model produced and never makes up samples."""
    processor = DPDFNet2Processor(DelayedDenoiser(), align=True)
    sizes = [len(processor.run(piece)) for piece in chunks(tone(0.4), 320)]
    # the model returns 160, 480 and 800 samples in all after 1, 2 and 3 chunks; 640 are delay
    assert sizes[:3] == [0, 0, 160]
    assert set(sizes[3:]) == {320}
    assert sum(sizes) == len(tone(0.4)) - MODEL_DELAY - HOP


@pytest.mark.parametrize('n', [1, 100, 639, 640, 641, 800])
def test_aligned_stream_shorter_than_the_delay(n):
    x = tone(1.0)[:n]
    out = run_all(DPDFNet2Processor(DelayedDenoiser(gain=0.5), align=True), x, 320)
    assert np.allclose(out, 0.5 * x, atol=1e-7) and len(out) == n


def test_aligned_finish_feeds_the_delay_as_silence_then_flushes_once():
    model = DelayedDenoiser()
    processor = DPDFNet2Processor(model, align=True)
    run_all(processor, tone(0.5), 1280)
    assert model.fed == len(tone(0.5)) + MODEL_DELAY
    assert np.all(model.history[-MODEL_DELAY:] == 0)
    assert (model.flushes, model.resets) == (1, 0)                # one flush, state never reset


def test_state_is_kept_from_chunk_to_chunk():
    x = tone(0.5, freq=300.0)
    one = run_all(DPDFNet2Processor(DelayedDenoiser(), align=True), x, len(x))
    for size in (320, 640, 1280, 999):
        assert np.array_equal(run_all(DPDFNet2Processor(DelayedDenoiser(), align=True), x, size), one)


def test_streams_are_independent():
    a, b = tone(0.6, freq=300.0), tone(0.6, freq=900.0, level=0.1)
    pa = DPDFNet2Processor(DelayedDenoiser(gain=0.5), align=True)
    pb = DPDFNet2Processor(DelayedDenoiser(gain=0.25), align=True)
    oa, ob = [], []
    for x, y in zip_longest(chunks(a, 320), chunks(b, 640)):   # interleaved calls
        if x is not None:
            oa.append(pa.run(x))
        if y is not None:
            ob.append(pb.run(y))
    oa, ob = np.concatenate(oa + [pa.finish()]), np.concatenate(ob + [pb.finish()])
    assert np.allclose(oa, 0.5 * a, atol=1e-7) and np.allclose(ob, 0.25 * b, atol=1e-7)


def test_a_model_short_at_the_end_is_completed_with_the_input():
    """Never fewer samples than in: the ASR timeline would drift."""
    x = tone(0.5)
    processor = DPDFNet2Processor(DelayedDenoiser(gain=0.5, drop_at_flush=100), align=True)
    out = run_all(processor, x, 640)
    assert len(out) == len(x)
    assert np.allclose(out[:-100], 0.5 * x[:-100], atol=1e-7) and np.array_equal(out[-100:], x[-100:])
    assert processor.filled == 100


# ------------------------------------------------------------------- PCM16 bytes
def test_feed_and_flush_pcm16():
    x = tone(0.5)
    processor = DPDFNet2Processor(DelayedDenoiser(gain=0.5), align=True)
    data = pcm16(x)
    out = b''.join([processor.feed(data[p:p + 2560]) for p in range(0, len(data), 2560)] + [processor.flush()])
    assert len(out) == len(data)
    assert np.abs(np.frombuffer(out, '<i2').astype(int)
                  - np.frombuffer(pcm16(0.5 * x), '<i2').astype(int)).max() <= 1


def test_feed_refuses_odd_bytes_and_counts_clipped_samples():
    processor = DPDFNet2Processor(DelayedDenoiser(gain=8.0), align=True)
    with pytest.raises(ValueError):
        processor.feed(b'\x00\x01\x02')
    samples = np.frombuffer(processor.feed(pcm16(tone(0.2, level=0.5))) + processor.flush(), '<i2')
    assert processor.clipped > 0
    assert (samples.max(), samples.min()) == (32767, -32768)       # saturated, not wrapped


def test_wrong_model_rate_is_refused():
    class WrongRate(DelayedDenoiser):
        def run(self, samples, sample_rate):
            result = super().run(samples, sample_rate)
            result.sample_rate = 8000
            return result
    with pytest.raises(RuntimeError, match='8000 Hz'):
        DPDFNet2Processor(WrongRate(), align=True).run(tone(0.1))


def test_finish_closes_the_processor():
    processor = DPDFNet2Processor(DelayedDenoiser(), align=True)
    processor.run(tone(0.1))
    processor.finish()
    with pytest.raises(RuntimeError):
        processor.run(tone(0.1))
    with pytest.raises(RuntimeError):
        processor.finish()


# ------------------------------------------------------- model failures (review of slice 1)
class FailingDenoiser(DelayedDenoiser):
    def __init__(self, fail_run_at=None, fail_flush=False, **options):
        super().__init__(**options)
        self.fail_run_at, self.fail_flush = fail_run_at, fail_flush

    def run(self, samples, sample_rate):
        if self.runs + 1 == self.fail_run_at:
            self.runs += 1
            raise RuntimeError('model failed')
        return super().run(samples, sample_rate)

    def flush(self):
        if self.fail_flush:
            raise RuntimeError('model failed')
        return super().flush()


def test_a_model_failing_at_the_end_leaves_the_processor_open():
    """stt_denoise then goes on with its own copy of the input; finish() may be called again."""
    processor = DPDFNet2Processor(FailingDenoiser(fail_flush=True), align=True)
    processor.run(tone(0.2))
    with pytest.raises(RuntimeError, match='model failed'):
        processor.finish()
    assert not processor.finished


class EarlyDenoiser(DelayedDenoiser):
    """A model with less delay than MODEL_DELAY: it would give more samples than were fed."""
    def reset(self):
        super().reset()
        self.history = np.zeros(MODEL_DELAY - 320, dtype=np.float32)


def test_aligned_run_never_returns_more_samples_than_were_fed():
    x = tone(0.5)
    processor = DPDFNet2Processor(EarlyDenoiser(), align=True)
    total = 0
    for piece in chunks(x, 320):
        total += len(processor.run(piece))
        assert total <= processor.samples_in
    total += len(processor.finish())
    assert total == len(x)


def test_check_model_delay_accepts_the_measured_delay_and_refuses_another():
    check_model_delay(DelayedDenoiser)
    with pytest.raises(ValueError, match='320'):
        check_model_delay(EarlyDenoiser)
