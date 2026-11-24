"""asr_denoise: DPDFNet-2 in front of VAD and ASR for speech_to_text (stt_denoise.SttDenoiser).

The ASR core is the real ASRService with the fake ASR / punctuation / VAD of test_regression.py;
the denoiser is DelayedDenoiser of test_denoise_processor.py, which has the timing of DPDFNet-2
(160-sample hops, one held back, audio 640 samples late). With gain 1 it gives the audio back
unchanged, so a stream denoised by it must give exactly the results of a stream that is not
denoised: same text, timestamps, utterances and commits. Any sample lost, repeated or moved
would change them. tests/test_noise_real_model.py runs the real model.
"""
import asyncio
import logging
import time

import grpc
import pytest

from asr_server.denoise import DenoiserPool
from asr_server.proto import speech_pb2 as pb
from asr_server.proto import speech_pb2_grpc
from asr_server.speech_service import SpeechStreamService
from asr_server.stt_denoise import SttDenoiser

from test_denoise_processor import DelayedDenoiser, FailingDenoiser
from test_regression import Aborted, FakeASR, FakeContext, make_service, pcm
from test_speech_grpc import SILENCE_VOICE_SILENCE, TWO, as_stream, chunks

WIDE = {'output_audio_format': pb.PCM_S16LE}         # what rtpgw sends for G.722 and L16 calls
SPEECH = TWO + SILENCE_VOICE_SILENCE                   # three utterances and long silences
ONE_SHORT = [pcm(0.6, True), pcm(0.6, False)]


def stage(*denoisers, **options):
    """SttDenoiser over a pool of the given denoisers (default: one identity denoiser)."""
    denoisers = list(denoisers) or [DelayedDenoiser(gain=1.0)]
    made = iter(denoisers)
    options.setdefault('max_backlog_ms', 0)            # the tests send all audio at once
    return SttDenoiser(DenoiserPool(lambda: next(made), len(denoisers)), **options)


async def paced(messages, seconds):
    for message in messages:
        await asyncio.sleep(seconds)
        yield message


def recognise(messages, denoiser=None, core=None, pace=None):
    """pace: seconds between chunks; None sends them all at once."""
    service = SpeechStreamService(core or make_service(), stt_denoise=denoiser)

    async def run():
        source = as_stream(messages) if pace is None else paced(messages, pace)
        out = [r async for r in service.Recognize(source, FakeContext())]
        await asyncio.sleep(0.05)                      # a denoiser given back from a worker thread
        return out
    return asyncio.run(run())


def fields(results):
    return [(r.text, r.is_final, r.ts_start, r.ts_end, r.utterance_id, r.revision, r.reason, r.seq)
            for r in results]


def same_as_without_denoising(messages, denoiser):
    expected = fields(recognise(messages))
    assert expected and any(r[1] for r in expected)        # the input has finals to compare
    assert fields(recognise(messages, denoiser)) == expected


# ------------------------------------------------------------------- the timeline is kept
@pytest.mark.parametrize('clock, chunk_bytes', [(8000, 640), (8000, 2560), (16000, 1280), (16000, 2560)],
                         ids=['g722-20ms', 'g722-80ms', 'l16-40ms', 'l16-80ms'])
def test_identity_model_gives_exactly_the_results_without_denoising(clock, chunk_bytes):
    model = DelayedDenoiser(gain=1.0)
    s = stage(model)
    same_as_without_denoising(chunks(SPEECH, clock=clock, chunk_bytes=chunk_bytes, **WIDE), s)
    assert model.runs > 0 and model.flushes == 1          # denoised, flushed once at the end
    assert model.resets == 1                              # only when the pool took it back
    assert (s.pool.free, s.stats['denoised']) == (1, 1)


def test_half_close_without_end_of_stream_flushes_the_model_too():
    model = DelayedDenoiser(gain=1.0)
    same_as_without_denoising(chunks(SPEECH, end=False, **WIDE), stage(model))
    assert model.flushes == 1


def test_vad_and_asr_get_the_denoised_audio():
    model = DelayedDenoiser(gain=0.0)                     # a model that removes everything
    assert recognise(chunks(SPEECH, **WIDE))
    assert recognise(chunks(SPEECH, **WIDE), stage(model)) == []


# -------------------------------------------------------------------- which streams
def test_g711_streams_are_not_denoised_by_default():
    model = DelayedDenoiser(gain=0.0)
    s = stage(model)
    same_as_without_denoising(chunks(SPEECH), s)          # no output_audio_format: a G.711 call
    assert model.runs == 0 and s.stats['narrowband'] == 1 and s.pool.free == 1


def test_apply_to_all_denoises_g711_streams_too():
    model = DelayedDenoiser(gain=1.0)
    same_as_without_denoising(chunks(SPEECH), stage(model, apply_to='all'))
    assert model.runs > 0


def test_8khz_streams_are_never_denoised():
    model = DelayedDenoiser(gain=0.0)
    s = stage(model, apply_to='all')
    eight_khz = chunks([pcm(0.6, True)[:9600], pcm(0.4, False)[:6400]], sample_rate=8000, chunk_bytes=1280, **WIDE)
    assert fields(recognise(eight_khz, s)) == fields(recognise(eight_khz))
    assert model.runs == 0


def test_no_free_denoiser_recognises_the_audio_as_sent(caplog):
    s = stage()
    taken = s.pool.acquire()
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.stt_denoise'):
        same_as_without_denoising(chunks(SPEECH, **WIDE), s)
    assert s.stats['no_denoiser'] == 1
    assert any('no denoiser free' in r.getMessage() for r in caplog.records)
    s.pool.release(taken)


# ------------------------------------------- failures: the input from the next sample on
@pytest.mark.parametrize('failure', [{'fail_run_at': 5}, {'fail_run_at': 1}, {'fail_flush': True}],
                         ids=['mid-stream', 'first-chunk', 'at-the-end'])
def test_model_error_goes_on_with_the_input_and_loses_nothing(failure, caplog):
    broken = FailingDenoiser(gain=1.0, **failure)
    s = stage(broken)
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.stt_denoise'):
        same_as_without_denoising(chunks(SPEECH, **WIDE), s)
    assert s.stats['fallback_error'] == 1
    assert any('reason=model_error' in r.getMessage() for r in caplog.records)
    assert broken.resets == 0                             # dropped, not reset and lent again


def test_a_broken_denoiser_is_replaced():
    made = []

    def factory():
        made.append(FailingDenoiser(gain=1.0, fail_run_at=2) if not made else DelayedDenoiser(gain=1.0))
        return made[-1]
    s = SttDenoiser(DenoiserPool(factory, 1), max_backlog_ms=0)
    recognise(chunks(SPEECH, **WIDE), s)
    deadline = time.monotonic() + 2
    while s.pool.free != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(made) == 2 and s.pool.free == 1


class SlowAt(DelayedDenoiser):
    def __init__(self, slow_run, seconds, **options):
        super().__init__(**options)
        self.slow_run, self.seconds = slow_run, seconds

    def run(self, samples, sample_rate):
        if self.runs + 1 == self.slow_run:
            time.sleep(self.seconds)
        return super().run(samples, sample_rate)


def test_a_model_call_slower_than_max_backlog_ms_goes_on_with_the_input(caplog):
    model = SlowAt(slow_run=4, seconds=0.4, gain=1.0)
    s = stage(model, max_backlog_ms=150)
    messages = chunks(TWO, **WIDE)
    expected = fields(recognise(messages))
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.stt_denoise'):
        # 80 ms chunks every 20 ms: the queue stays under the limit, only the slow call is over it
        assert fields(recognise(messages, s, pace=0.02)) == expected
    assert (s.stats['fallback_slow'], s.stats['fallback_backlog']) == (1, 0)
    assert any('reason=model_slow' in r.getMessage() for r in caplog.records)
    deadline = time.monotonic() + 2
    while s.pool.free != 1 and time.monotonic() < deadline:   # given back once the call returns
        time.sleep(0.01)
    assert s.pool.free == 1 and model.resets == 1


def test_audio_waiting_longer_than_max_backlog_ms_goes_on_with_the_input():
    s = stage(max_backlog_ms=200)        # all chunks arrive at once: far more than 200 ms waits
    same_as_without_denoising(chunks(SPEECH, **WIDE), s)
    assert s.stats['fallback_backlog'] == 1


# ------------------------------------------------------------------ stream end and errors
def test_protocol_error_keeps_its_status_and_the_denoiser_comes_back():
    def change_session(messages):
        messages[5].session_id = 'other'
    bad = chunks(SPEECH, mutate=change_session, **WIDE)
    with pytest.raises(Aborted) as without:
        recognise(bad)
    s = stage()
    with pytest.raises(Aborted) as with_denoising:
        recognise(bad, s)
    assert with_denoising.value.code == without.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert s.pool.free == 1


def test_summary_line_per_stream(caplog):
    with caplog.at_level(logging.INFO, logger='qwen3_asr.stt_denoise'):
        recognise(chunks(SPEECH, **WIDE), stage())
    lines = [r.getMessage() for r in caplog.records if 'stt_denoise stream ended' in r.getMessage()]
    assert len(lines) == 1 and 'outcome=denoised' in lines[0] and 'session=call-1' in lines[0]


def test_client_cancel_over_grpc_gives_the_denoiser_back():
    s = stage(SlowAt(slow_run=3, seconds=0.3, gain=1.0))

    async def run():
        server = grpc.aio.server()
        speech_pb2_grpc.add_SpeechStreamServicer_to_server(SpeechStreamService(make_service(), stt_denoise=s),
                                                           server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
                call = speech_pb2_grpc.SpeechStreamStub(channel).Recognize(
                    as_stream(chunks(SPEECH, end=False, **WIDE)[:6]))
                await asyncio.sleep(0.2)                  # the slow model call is running
                call.cancel()
                deadline = time.monotonic() + 3
                while s.pool.free != 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                return s.pool.free
        finally:
            await server.stop(None)
    assert asyncio.run(run()) == 1
    s.close()


# ------------------------------------------------------------------------- serve()
from asr_server import denoise as denoise_module, server                    # noqa: E402
from asr_server.stt_denoise import SttDenoiser as _SttDenoiser            # noqa: E402

from test_denoise_processor import EarlyDenoiser                         # noqa: E402
from test_serve_noise import (fake_asr, no_asr, offline_env, settings,   # noqa: E402,F401
                              with_running_server)


def fake_factory(monkeypatch, kind=DelayedDenoiser):
    """dpdfnet2_factory -> `kind` (gain 1); records the arguments and every denoiser made."""
    calls = {'args': [], 'made': []}

    def factory(model, num_threads, setting='noise_reduction'):
        calls['args'].append((model, num_threads, setting))

        def make():
            calls['made'].append(kind(gain=1.0))
            return calls['made'][-1]
        return make
    monkeypatch.setattr(denoise_module, 'dpdfnet2_factory', factory)
    return calls


def stt_settings(tmp_path, **values):
    base = dict(stt_denoise_enabled=True, stt_denoise_max_streams=2, stt_denoise_max_backlog_ms=0)
    base.update(values)
    s = settings(tmp_path, **base)
    if 'stt_denoise_model' not in values:        # the model file settings() wrote
        s = settings(tmp_path, **{**base, 'stt_denoise_model': s.noise_model})
    return s


def test_serve_denoises_wideband_speech_to_text(tmp_path, fake_asr, monkeypatch):
    calls = fake_factory(monkeypatch)
    s = stt_settings(tmp_path, stt_denoise_num_threads=3)

    async def check(stub):
        out = [r async for r in stub.Recognize(as_stream(chunks(SPEECH, **WIDE)))]
        assert out[-1].is_final and out[-1].text
    asyncio.run(with_running_server(s, check))
    assert calls['args'] == [(s.stt_denoise_model, 3, 'asr_denoise')]
    assert len(calls['made']) == 1 + 2                    # the delay check, then max_streams
    assert sum(d.runs for d in calls['made'][1:]) > 0 and calls['made'][0].flushes == 1


def test_serve_refuses_a_model_with_another_delay_before_asr_loads(tmp_path, no_asr, monkeypatch, caplog):
    fake_factory(monkeypatch, kind=EarlyDenoiser)
    with caplog.at_level(logging.ERROR, logger='qwen3_asr'):
        with pytest.raises(ValueError, match='delays audio by 320 samples'):
            asyncio.run(server.serve(stt_settings(tmp_path)))
    assert any('asr_denoise cannot start' in r.getMessage() for r in caplog.records)


def test_missing_asr_denoise_model_is_named_with_its_setting(tmp_path, no_asr):
    with pytest.raises(FileNotFoundError, match=r'asr_denoise\.model.*nope\.onnx'):
        asyncio.run(server.serve(stt_settings(tmp_path, stt_denoise_model=str(tmp_path / 'nope.onnx'))))


def test_asr_denoise_off_by_default_loads_no_denoiser(tmp_path, fake_asr, monkeypatch):
    calls = fake_factory(monkeypatch)

    async def check(stub):
        assert [r async for r in stub.Recognize(as_stream(chunks(SPEECH, **WIDE)))][-1].is_final
    asyncio.run(with_running_server(settings(tmp_path), check))
    assert calls['made'] == []


def test_stt_denoiser_is_closed_when_the_server_stops(tmp_path, fake_asr, monkeypatch):
    fake_factory(monkeypatch)
    closed = []
    monkeypatch.setattr(_SttDenoiser, 'close', lambda self: closed.append(self))

    async def check(stub):
        pass
    asyncio.run(with_running_server(stt_settings(tmp_path), check))
    assert len(closed) == 1


# ------------------------------------------------------------ review of slice 2
def test_a_stream_waiting_for_a_busy_worker_is_not_blamed_on_its_model(caplog):
    """workers=1: stream B's call waits behind stream A's slow call. Both go on with the input
    (latency is bounded), but B's reason is `busy`, not `model_slow`."""
    a = SlowAt(slow_run=2, seconds=0.6, gain=1.0)
    b = DelayedDenoiser(gain=1.0)
    s = stage(b, a, max_backlog_ms=150, workers=1)   # the pool lends from the end: A gets `a`
    messages_a = chunks(TWO, **WIDE)
    messages_b = chunks(TWO, session='call-2', stream='call-2-a2b', **WIDE)
    expected_a, expected_b = fields(recognise(messages_a)), fields(recognise(messages_b))

    async def run():
        service = SpeechStreamService(make_service(), stt_denoise=s)

        async def one(messages, delay):
            await asyncio.sleep(delay)
            return [r async for r in service.Recognize(paced(messages, 0.02), FakeContext())]
        return await asyncio.gather(one(messages_a, 0), one(messages_b, 0.1))
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.stt_denoise'):
        out_a, out_b = asyncio.run(run())
    assert fields(out_a) == expected_a and fields(out_b) == expected_b
    assert (s.stats['fallback_slow'], s.stats['fallback_busy']) == (1, 1)
    assert any('session=call-2' in r.getMessage() and 'reason=busy' in r.getMessage() for r in caplog.records)


def test_no_free_denoiser_warning_is_not_repeated_for_every_stream(caplog):
    s = stage()
    taken = s.pool.acquire()
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.stt_denoise'):
        for _ in range(6):
            recognise(chunks(ONE_SHORT, **WIDE), s)
    warnings = [r for r in caplog.records if 'no denoiser free' in r.getMessage()]
    assert s.stats['no_denoiser'] == 6 and len(warnings) == 3        # the first 3, then every 500th
    s.pool.release(taken)


def test_a_non_16khz_model_is_refused_naming_asr_denoise(tmp_path, monkeypatch):
    from test_serve_noise import fake_sherpa
    model = tmp_path / 'dpdfnet2.onnx'
    model.write_bytes(b'onnx')
    monkeypatch.setitem(__import__('sys').modules, 'sherpa_onnx', fake_sherpa(rate=8000))
    with pytest.raises(ValueError, match=r'asr_denoise needs 16000 Hz'):
        denoise_module.dpdfnet2_factory(str(model), setting='asr_denoise')()


# ------------------------------------------------------------ review of slice 3
def test_a_stream_that_went_raw_gives_its_denoiser_back_at_once():
    """After a fallback the denoiser is of no use to the stream: another call must get it."""
    s = stage(max_backlog_ms=200)                    # all chunks at once: backlog at the first chunk
    service = SpeechStreamService(make_service(), stt_denoise=s)

    async def run():
        source = as_stream(chunks(SPEECH, end=False, **WIDE))
        results = service.Recognize(paced_after_first(source), FakeContext())
        first = asyncio.ensure_future(anext(results, None))
        await asyncio.sleep(0.3)                      # stream 1 fell back and is still open
        free = s.pool.free
        first.cancel()
        await results.aclose()
        return free
    assert asyncio.run(run()) == 1 and s.stats['fallback_backlog'] == 1


async def paced_after_first(source):
    """All chunks at once, then the stream stays open."""
    async for message in source:
        yield message
    await asyncio.sleep(10)


def test_a_stream_rejected_on_its_first_chunk_does_not_borrow_a_denoiser(caplog):
    s = stage()
    taken = s.pool.acquire()                          # pool empty: a borrow attempt would warn
    bad = chunks(SPEECH, task='realtime_translation', **WIDE)
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.stt_denoise'):
        with pytest.raises(Aborted):
            recognise(bad, s)
    assert s.stats['no_denoiser'] == 0 and s.stats['narrowband'] == 0
    assert not [r for r in caplog.records if 'no denoiser free' in r.getMessage()]
    s.pool.release(taken)


def test_a_stream_ended_by_a_protocol_error_is_not_logged_as_cancelled(caplog):
    def change_session(messages):
        messages[5].session_id = 'other'
    with caplog.at_level(logging.INFO, logger='qwen3_asr.stt_denoise'):
        with pytest.raises(Aborted):
            recognise(chunks(SPEECH, mutate=change_session, **WIDE), stage())
    line = [r.getMessage() for r in caplog.records if 'stt_denoise stream ended' in r.getMessage()]
    assert len(line) == 1 and 'outcome=error' in line[0]


def test_ingress_overflow_of_the_core_gives_the_denoiser_back():
    s = stage()
    with pytest.raises(Aborted) as aborted:
        # a slow ASR: the audio piles up in the core's queue, past server.ingress_max_ms
        recognise(chunks(SPEECH, **WIDE), s, core=make_service(FakeASR(text='xin chao anh', delay=0.2),
                                                               ingress_max_ms=200))
    assert aborted.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert s.pool.free == 1
