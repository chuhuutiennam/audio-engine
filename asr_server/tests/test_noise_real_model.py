"""noise_reduction with the real DPDFNet-2 model (sherpa-onnx), not a stand-in.

Skipped unless the sherpa_onnx package is installed and ASR_TEST_DPDFNET2_MODEL names the model
file, e.g.:
    ASR_TEST_DPDFNET2_MODEL=/opt/models/dpdfnet2.onnx python -m pytest tests/test_noise_real_model.py -s
Imports nothing of the speech_to_text core, so it also runs in an environment without torch/vLLM.
Expected on sherpa-onnx 2.0.2 with dpdfnet2.onnx (SHA-256 ce35d602...87a9): one 10 ms hop held
back, flush() returns it, 40 ms algorithmic delay, the same output for any chunk size.
"""
import asyncio
import itertools
import logging
import os

import numpy as np
import pytest

pytest.importorskip('sherpa_onnx')
MODEL = os.environ.get('ASR_TEST_DPDFNET2_MODEL', '')
pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL),
                                reason='set ASR_TEST_DPDFNET2_MODEL to the dpdfnet2.onnx file')

from asr_server.audio import decode_audio                     # noqa: E402
from asr_server.denoise import (MODEL_DELAY, OUTPUT_PCM16, OUTPUT_PCMU,  # noqa: E402
                                DenoiserPool, DPDFNet2Processor, NoiseStream, dpdfnet2_factory)
from asr_server.messages import PCMU                          # noqa: E402
from asr_server.noise_service import NoiseReductionService    # noqa: E402
from asr_server.proto import speech_pb2 as pb                 # noqa: E402
from asr_server.speech_service import SpeechStreamService     # noqa: E402

RATE = 16000


def noisy_speech_like(seconds=3.0, seed=0):
    """A 220 Hz tone switched on and off every 0.5 s, in white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    gate = (np.floor(t * 2) % 2 == 0).astype(np.float32)
    x = 0.3 * gate * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    return np.clip(np.round(x * 32768), -32768, 32767).astype('<i2').tobytes()


def denoise(pcm, chunk_ms, output=OUTPUT_PCM16, denoiser=None):
    stream = NoiseStream(denoiser or dpdfnet2_factory(MODEL)(), output=output, frame_ms=20)
    size = RATE * chunk_ms // 1000 * 2
    per_chunk = [stream.process(pcm[p:p + size]) for p in range(0, len(pcm), size)]
    return per_chunk, stream.finish(), stream


def joined(per_chunk, tail):
    return b''.join(f for chunk in per_chunk for f in chunk) + b''.join(tail)


def test_every_sample_comes_back_whatever_the_chunk_size():
    pcm = noisy_speech_like()
    ref_chunks, ref_tail, stream = denoise(pcm, 80)
    out = joined(ref_chunks, ref_tail)
    assert len(out) == len(pcm) and stream.samples_out == stream.samples_in == len(pcm) // 2
    assert out != pcm                                          # the model changed the audio
    assert all(chunk for chunk in ref_chunks)                  # audio for every 80 ms chunk
    for chunk_ms in (20, 19, 81):
        assert joined(*denoise(pcm, chunk_ms)[:2]) == out


def test_reset_denoiser_gives_the_output_of_a_fresh_one():
    pcm = noisy_speech_like(seed=1)
    pool = DenoiserPool(dpdfnet2_factory(MODEL), 1)
    used = pool.acquire()
    denoise(noisy_speech_like(seed=2), 80, denoiser=used)      # leaves state behind
    assert pool.release(used) is True
    again = pool.acquire()
    assert again is used
    assert joined(*denoise(pcm, 80, denoiser=again)[:2]) == joined(*denoise(pcm, 80)[:2])


def test_model_delay_is_40_ms():
    # On the model's float output: it removes a lone click almost entirely (peak ~2e-5, below
    # one 16-bit step), so the PCM16 output cannot show where the peak is.
    model = dpdfnet2_factory(MODEL)()
    x = np.zeros(RATE, dtype=np.float32)
    x[1000] = 0.5
    y = np.concatenate([np.asarray(model.run(x[p:p + 320], RATE).samples) for p in range(0, RATE, 320)]
                       + [np.asarray(model.flush().samples)])
    assert int(np.argmax(np.abs(y))) - 1000 == 640 == MODEL_DELAY


@pytest.mark.parametrize('chunk', [320, 640, 1280])
def test_aligned_processor_keeps_the_input_timeline(chunk):
    """speech_to_text preprocessing: as many samples as in, and no delay left (VAD/ASR timestamps)."""
    # A lone click, as in test_model_delay_is_40_ms (the periodic test signal would make a
    # cross-correlation peak one period off): it must come out where it went in.
    x = np.zeros(RATE, dtype=np.float32)
    x[1000] = 0.5
    processor = DPDFNet2Processor(dpdfnet2_factory(MODEL)(), align=True)
    y = np.concatenate([processor.run(x[p:p + chunk]) for p in range(0, len(x), chunk)] + [processor.finish()])
    assert len(y) == len(x) and processor.filled == 0
    assert int(np.argmax(np.abs(y))) == 1000
    # and on noisy audio: as many samples out as in, for any chunk size
    pcm = decode_audio(noisy_speech_like(), 3)
    processor = DPDFNet2Processor(dpdfnet2_factory(MODEL)(), align=True)
    y = np.concatenate([processor.run(pcm[p:p + chunk]) for p in range(0, len(pcm), chunk)] + [processor.finish()])
    assert len(y) == len(pcm) and processor.filled == 0


def test_pcmu_path_returns_8khz_audio_for_every_chunk():
    pcm = noisy_speech_like()
    per_chunk, tail, _ = denoise(pcm, 80, output=OUTPUT_PCMU)
    assert all(chunk for chunk in per_chunk)
    out = joined(per_chunk, tail)
    assert len(out) == len(pcm) // 4                           # 8 kHz, 1 byte per sample
    assert np.max(np.abs(decode_audio(out, PCMU))) > 0.01      # not silence


def test_grpc_stream_with_the_real_model_and_its_log_line(caplog):
    pcm = noisy_speech_like()
    size = 2560
    pieces = [pcm[p:p + size] for p in range(0, len(pcm), size)]
    messages = [pb.AudioChunk(session_id='call-1', stream_id='call-1-a2b', pcm=piece, sample_rate=16000,
                              channels=1, rtp_timestamp=n * 640, rtp_clock_rate=8000, chunk_seq=n,
                              task='noise_reduction', output_audio_format=pb.AudioFormat.PCM_S16LE,
                              end_of_stream=n == len(pieces) - 1)
                for n, piece in enumerate(pieces)]
    # max_backlog_ms 0: the whole 3 s is sent at once
    svc = NoiseReductionService(DenoiserPool(dpdfnet2_factory(MODEL), 2), frame_ms=20, workers=2,
                                max_backlog_ms=0)
    log = []

    async def logged():
        for message in messages:
            log.append(('in', message.rtp_timestamp))
            yield message

    class Context:
        async def abort(self, code, details):
            raise AssertionError(f'{code}: {details}')

    async def go():
        async for r in SpeechStreamService(None, noise=svc).Recognize(logged(), Context()):
            log.append(('out', r))
    try:
        with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.noise'):
            asyncio.run(go())
    finally:
        svc.close()
    outs = [value for kind, value in log if kind == 'out']
    assert [r.is_final for r in outs].count(True) == 1 and outs[-1].is_final
    assert sum(len(r.audio_payload) for r in outs) == len(pcm)
    # in chunk order, each result carries the rtp_timestamp of the chunk just given to the model
    stamps = [ts for ts, _ in itertools.groupby(r.ts_end for r in outs[:-1])]
    assert stamps == [m.rtp_timestamp for m in messages]
    line = [r.getMessage() for r in caplog.records if 'noise_reduction stream ended' in r.getMessage()]
    print('\n' + line[0])
    fields = dict(item.split('=', 1) for item in line[0].split() if '=' in item)
    assert fields['outcome'] == 'final'
    assert fields['samples_in'] == fields['samples_out'] == fields['samples_sent'] == str(len(pcm) // 2)
    assert int(fields['first_audio_ms']) < 500                 # the gateway falls back after 500 ms


def test_stt_denoise_stage_with_the_real_model(caplog):
    """asr_denoise in front of the core: the requests the core gets from the real model.
    (No delay left: test_aligned_processor_keeps_the_input_timeline, on the float output.)"""
    from asr_server import messages as m
    from asr_server.stt_denoise import SttDenoiser
    pcm = noisy_speech_like(4.0)

    async def requests():
        yield m.StreamRequest(config=m.StreamConfig(call_id='c', leg_id='l', encoding=m.PCM_S16LE,
                                                    sample_rate=RATE))
        for n, p in enumerate(range(0, len(pcm), 2560), 1):
            yield m.StreamRequest(audio=m.AudioChunk(data=pcm[p:p + 2560], sequence=n))
        yield m.StreamRequest(end_stream=m.EndStream())

    stage = SttDenoiser(DenoiserPool(dpdfnet2_factory(MODEL), 1), max_backlog_ms=0)

    async def run():
        return [r async for r in stage.stream(requests(), 's', 't')]
    with caplog.at_level(logging.INFO, logger='qwen3_asr.stt_denoise'):
        out = asyncio.run(run())
    stage.close()
    kinds = [r.WhichOneof('payload') for r in out]
    assert kinds[0] == 'config' and kinds[-1] == 'end_stream' and set(kinds[1:-1]) == {'audio'}
    audio = [r.audio for r in out[1:-1]]
    assert [a.sequence for a in audio] == list(range(1, len(audio) + 1))
    y = np.frombuffer(b''.join(a.data for a in audio), '<i2')
    assert len(y) == len(pcm) // 2 and stage.stats['denoised'] == 1
    assert y.tobytes() != pcm                                 # the model changed the audio
    assert any('outcome=denoised' in r.getMessage() for r in caplog.records)
