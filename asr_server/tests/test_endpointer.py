import numpy as np
from asr_server.vad import WebRTCEndpointer


def classifier(x, sr):
    return np.max(np.abs(x)) > 0.05


def feed_samples(vad, data, frame_ms=20):
    out = []
    block = vad.sample_rate * frame_ms // 1000
    for k in range(0, len(data), block):
        out.extend(vad.push(data[k:k + block]))
    return out


def test_audio_start_endpoint_and_pause_candidate():
    v = WebRTCEndpointer(8000, classifier=classifier,
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
    v = WebRTCEndpointer(16000, classifier=classifier)
    assert not feed_samples(v, np.zeros(16000, np.float32))
    tiny = np.concatenate([np.ones(16000 // 10, np.float32)*0.2,
                           np.zeros(16000, np.float32)])
    events = feed_samples(v, tiny)
    assert all(e.kind != 'endpoint' for e in events)
    assert 'discard' in [e.kind for e in events]


def test_reset_after_candidate_no_time_reset():
    v = WebRTCEndpointer(8000, classifier=classifier)
    chunk = np.concatenate([np.ones(8000, np.float32)*0.2,
                            np.zeros(4000, np.float32)])
    events = feed_samples(v, chunk)
    assert any(e.kind == 'pause_candidate' for e in events)
    timestamp = v.processed_frames
    v.reset_activity()
    assert not v.active
    assert v.processed_frames == timestamp
