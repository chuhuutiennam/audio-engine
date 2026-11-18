import numpy as np
from asr_server.vad import SileroEndpointer


def classifier(x, sr):
    return np.max(np.abs(x)) > 0.05


def feed_samples(vad, data, frame_ms=20):
    out = []
    block = vad.sample_rate * frame_ms // 1000
    for k in range(0, len(data), block):
        out.extend(vad.push(data[k:k + block]))
    return out


def test_audio_start_endpoint_and_pause_candidate():
    v = SileroEndpointer(8000, classifier=classifier,
                         endpoint_ms=600, pause_ms=300)
    audio = np.concatenate([np.zeros(1600, np.float32),
                            np.ones(8000, np.float32)*0.2,
                            np.zeros(6400, np.float32)])
    events = feed_samples(v, audio)
    kinds = [e.kind for e in events]
    assert kinds.count('pause_candidate') == 1
    assert kinds.count('endpoint') == 1
    assert kinds.count('audio') > 10
    final = [e for e in events if e.kind == 'endpoint'][0]
    assert 1500 <= final.end_ms <= 1900
    assert not v.active


def test_silence_and_short_noise_not_committed():
    v = SileroEndpointer(16000, classifier=classifier)
    assert not feed_samples(v, np.zeros(16000, np.float32))
    tiny = np.concatenate([np.ones(16000 // 10, np.float32)*0.2,
                           np.zeros(16000, np.float32)])
    events = feed_samples(v, tiny)
    assert all(e.kind != 'endpoint' for e in events)
    assert 'discard' in [e.kind for e in events]


def test_reset_after_candidate_no_time_reset():
    v = SileroEndpointer(8000, classifier=classifier)
    chunk = np.concatenate([np.ones(8000, np.float32)*0.2,
                            np.zeros(4000, np.float32)])
    events = feed_samples(v, chunk)
    assert any(e.kind == 'pause_candidate' for e in events)
    timestamp = v.processed_frames
    v.reset_activity()
    assert not v.active
    assert v.processed_frames == timestamp


def test_end_utterance_keeps_absolute_clock_with_partial_32ms_frame():
    v = SileroEndpointer(8000, classifier=classifier)
    # 350 ms is not an integer multiple of Silero's 32 ms window.
    first = np.full(2800, 0.2, np.float32)
    assert any(e.kind == 'audio' for e in v.push(first))
    ended = v.finish()
    assert ended[-1].end_ms == 350
    second = v.push(np.full(2800, 0.2, np.float32))
    starts = [e.start_ms for e in second if e.kind == 'audio']
    assert starts and starts[0] >= 350


def test_local_torchscript_loading_with_independent_state(tmp_path):
    """Use a tiny scripted stand-in to verify offline JIT I/O, not Silero accuracy."""
    import torch

    class StubVad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('calls', torch.zeros(1))

        @torch.jit.export
        def reset_states(self):
            self.calls.zero_()

        def forward(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
            if sample_rate == 8000:
                assert samples.numel() == 256
            else:
                assert sample_rate == 16000
                assert samples.numel() == 512
            self.calls.add_(1.0)
            return torch.clamp(samples.abs().mean() * 10, 0, 1).reshape(1)

    checkpoint = tmp_path / 'vad_stub.jit'
    torch.jit.script(StubVad()).save(str(checkpoint))

    for sr in (8000, 16000):
        a = SileroEndpointer(sr, model_path=checkpoint)
        b = SileroEndpointer(sr, model_path=checkpoint)
        assert a._model is not b._model
        assert not a.push(np.zeros(a.frame_samples, np.float32))
        assert a._model.calls.item() == 1
        assert b._model.calls.item() == 0
        events = feed_samples(a, np.r_[np.full(sr, .2, np.float32),
                                       np.zeros(sr, np.float32)], frame_ms=20)
        assert any(x.kind == 'endpoint' for x in events)
