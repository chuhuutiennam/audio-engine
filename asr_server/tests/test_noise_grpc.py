"""speech.proto task noise_reduction (noise_service.py) and the task routing of SpeechStreamService.

The model is the stand-in of test_denoise.py (holds back 160 samples like DPDFNet-2, halves the
level). Design: docs/DPDFNet2_Task_Routing_Design_VI.md, sections 4, 5 and 8.
"""
import asyncio
import itertools
import logging
import threading
import time

import grpc
import numpy as np
import pytest

from asr_server.denoise import DenoiserPool
from asr_server.noise_service import NoiseReductionService
from asr_server.proto import speech_pb2 as pb
from asr_server.proto import speech_pb2_grpc
from asr_server.speech_service import SpeechStreamService

from test_denoise import FakeDenoiser, pcm16, tone
from test_regression import Aborted, FakeContext, make_service
from test_speech_grpc import ONE, as_stream, chunks, recognise

CHUNK = 2560            # 80 ms of PCM16 mono at 16 kHz, as the gateway sends
DATA = pcm16(tone(1.0)) # 12.5 chunks: the last one is short


def noise_chunks(data=DATA, *, fmt=pb.PCM_S16LE, chunk_bytes=CHUNK, rtp_start=1000, clock=8000,
                 end=True, session='call-1', stream='call-1-a2b', sample_rate=16000, mutate=None):
    pieces = [data[p:p + chunk_bytes] for p in range(0, len(data), chunk_bytes)]
    ticks = chunk_bytes // 2 * clock // sample_rate
    out = []
    for n, piece in enumerate(pieces):
        chunk = pb.AudioChunk(
            session_id=session, stream_id=stream, pcm=piece, sample_rate=sample_rate, channels=1,
            rtp_timestamp=(rtp_start + n * ticks) % (1 << 32), rtp_clock_rate=clock, chunk_seq=n,
            duration_ms=len(piece) // 32, task='noise_reduction',
            end_of_stream=(end and n == len(pieces) - 1))
        if fmt is not None:
            chunk.output_audio_format = fmt
        out.append(chunk)
    if mutate:
        mutate(out)
    return out


def noise_service(size=2, factory=FakeDenoiser, frame_ms=20, max_backlog_ms=0):
    # Most tests send all their audio at once, which the backlog limit is meant to refuse.
    return NoiseReductionService(DenoiserPool(factory, size), frame_ms=frame_ms, workers=2,
                                 max_backlog_ms=max_backlog_ms)


def run_noise(messages, svc=None, core=None):
    own = svc is None
    svc = svc or noise_service()
    service = SpeechStreamService(core, noise=svc)

    async def go():
        return [r async for r in service.Recognize(as_stream(messages), FakeContext())]
    try:
        return asyncio.run(go())
    finally:
        if own:
            svc.close()


def noise_rejected(messages, **options):
    try:
        run_noise(messages, **options)
    except Aborted as err:
        return err.code
    raise AssertionError('expected the RPC to be aborted')


async def wait_until(predicate, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError('condition not reached')


# ------------------------------------------------------------------ PCM16 16 kHz
def test_pcm16_results_carry_the_denoised_audio_and_one_empty_final():
    out = run_noise(noise_chunks())
    audio, final = out[:-1], out[-1]
    assert final.is_final and final.audio_payload == b'' and not any(r.is_final for r in audio)
    assert all(len(r.audio_payload) == 640 for r in audio[:-1]) and 0 < len(audio[-1].audio_payload) <= 640
    pcm = np.frombuffer(b''.join(r.audio_payload for r in audio), dtype='<i2')
    expected = np.frombuffer(pcm16(np.frombuffer(DATA, dtype='<i2') / 32768 * 0.5), dtype='<i2')
    assert np.array_equal(pcm, expected)                    # all samples, in order, nothing added
    for r in out:
        assert (r.session_id, r.stream_id, r.text) == ('call-1', 'call-1-a2b', '')
        assert (r.audio_format, r.audio_sample_rate) == (pb.PCM_S16LE, 16000)
        assert r.ts_start == r.ts_end
    assert [r.seq for r in out] == list(range(1, len(out) + 1))


def test_audio_comes_back_for_every_chunk_with_ts_end_of_that_chunk():
    messages = noise_chunks()
    audio = run_noise(messages)[:-1]
    # In chunk order, each result carries the rtp_timestamp of the chunk just given to the model.
    groups = [(ts, len(list(results))) for ts, results in itertools.groupby(r.ts_end for r in audio)]
    assert [ts for ts, _ in groups] == [m.rtp_timestamp for m in messages]
    # 1280 samples in -> 3 frames of 20 ms the first time (160 held back by the model), then 4
    assert [n for _, n in groups[:-1]] == [3] + [4] * (len(messages) - 2)


# ------------------------------------------------------------------- G.711 PCMU
@pytest.mark.parametrize('fmt', [None, pb.LEGACY_ENCODED])
def test_absent_or_legacy_format_returns_pcmu(fmt):
    out = run_noise(noise_chunks(fmt=fmt))
    audio, final = out[:-1], out[-1]
    assert all(len(r.audio_payload) == 160 for r in audio[:-1])
    assert sum(len(r.audio_payload) for r in audio) == len(DATA) // 4        # 8 kHz, 1 byte per sample
    for r in out:
        assert (r.audio_format, r.audio_pt, r.audio_sample_rate) == (pb.LEGACY_ENCODED, 0, 0)
    assert final.is_final and final.audio_payload == b''


def test_frame_ms_sets_the_result_size():
    svc = noise_service(frame_ms=80)
    try:
        out = run_noise(noise_chunks(), svc=svc)
    finally:
        svc.close()
    # 1 s = 12 frames of 80 ms + 40 ms, then the final (review L3: the check alone could be empty)
    assert [len(r.audio_payload) for r in out] == [2560] * 12 + [1280, 0]


def test_unsupported_frame_ms_is_refused_when_the_service_is_built():
    # review L3: a bad frame_ms raised after the denoiser was borrowed, which then never came back
    with pytest.raises(ValueError):
        noise_service(frame_ms=30)


# ------------------------------------------------------------------- end of stream
def test_half_close_without_end_of_stream_flushes_and_sends_one_final():
    out = run_noise(noise_chunks(end=False))
    assert [r.is_final for r in out].count(True) == 1 and out[-1].is_final
    assert sum(len(r.audio_payload) for r in out) == len(DATA)


def test_final_carries_the_rtp_timestamp_of_the_end_of_stream_chunk():
    messages = noise_chunks()
    out = run_noise(messages)
    assert out[-1].ts_end == messages[-1].rtp_timestamp


def test_audio_flushed_at_the_end_keeps_the_timestamp_of_the_last_audio_chunk():
    # review L3: an audio-less end_of_stream chunk lent its rtp_timestamp to the flushed audio
    def empty_end(out):
        out[-1].end_of_stream = False
        out.append(pb.AudioChunk(session_id='call-1', stream_id='call-1-a2b', sample_rate=16000,
                                 channels=1, rtp_timestamp=99999, rtp_clock_rate=8000,
                                 chunk_seq=len(out), task='noise_reduction',
                                 output_audio_format=pb.PCM_S16LE, end_of_stream=True))
    messages = noise_chunks(mutate=empty_end)
    out = run_noise(messages)
    last_audio_ts = messages[-2].rtp_timestamp
    assert out[-2].audio_payload and out[-2].ts_end == last_audio_ts
    assert out[-1].is_final and out[-1].ts_end == 99999


def test_chunks_after_end_of_stream_are_not_read():
    def extra(out):
        out.append(pb.AudioChunk(session_id='other', stream_id='x', task='noise_reduction'))
    out = run_noise(noise_chunks(mutate=extra))
    assert [r.is_final for r in out].count(True) == 1


def test_stream_without_audio_ends_with_only_the_final():
    out = run_noise([pb.AudioChunk(session_id='call-1', stream_id='s', sample_rate=16000, rtp_timestamp=7,
                                   task='noise_reduction', output_audio_format=pb.PCM_S16LE,
                                   end_of_stream=True)])
    assert len(out) == 1 and out[0].is_final and out[0].ts_end == 7


# ------------------------------------------------------------------ contract checks
def _set(index, **fields):
    def mutate(out):
        for name, value in fields.items():
            setattr(out[index], name, value)
    return mutate


@pytest.mark.parametrize('mutate', [
    _set(0, output_audio_format=2),                          # unknown format
    _set(3, output_audio_format=pb.LEGACY_ENCODED),          # format changes within the stream
    _set(0, session_id=''),
    _set(0, stream_id=''),
    _set(2, session_id='call-2'),
    _set(2, stream_id='call-1-b2a'),
    _set(2, rtp_clock_rate=16000),
    _set(0, channels=2),
    _set(2, channels=2),
    _set(2, pcm=b'\x00' * 3),                                # odd number of bytes
])
def test_contract_violations_are_invalid_argument(mutate):
    assert noise_rejected(noise_chunks(mutate=mutate)) == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize('rate', [8000, 48000])
def test_only_16khz_is_accepted(rate):
    assert noise_rejected(noise_chunks(sample_rate=rate)) == grpc.StatusCode.INVALID_ARGUMENT


def test_task_change_within_the_stream_is_invalid_argument():
    assert noise_rejected(noise_chunks(mutate=_set(2, task='speech_to_text'))) == \
        grpc.StatusCode.INVALID_ARGUMENT


def test_chunk_seq_gap_is_logged_and_processing_goes_on(caplog):
    def gap(out):
        for n, chunk in enumerate(out):
            chunk.chunk_seq = n + (5 if n >= 4 else 0)
    with caplog.at_level(logging.WARNING):
        out = run_noise(noise_chunks(mutate=gap))
    assert out[-1].is_final and 'chunk_seq gap' in caplog.text


# --------------------------------------------------------------- denoiser pool
async def _hold_one_stream(service):
    """Start a stream that keeps its denoiser until the returned event is set."""
    release = asyncio.Event()
    first = noise_chunks()

    async def held_open():
        yield first[0]
        await release.wait()
        for message in first[1:]:
            yield message
    got = []

    async def consume():
        async for r in service.Recognize(held_open(), FakeContext()):
            got.append(r)
    task = asyncio.create_task(consume())
    await wait_until(lambda: got)
    return release, task


async def _refused_code(service, messages):
    try:
        [r async for r in service.Recognize(as_stream(messages), FakeContext())]
    except Aborted as err:
        return err.code
    raise AssertionError('expected the RPC to be aborted')


def test_stream_beyond_the_pool_is_refused_at_once_and_the_slot_comes_back():
    async def go():
        svc = noise_service(size=1)
        release = None
        try:
            service = SpeechStreamService(None, noise=svc)
            release, task = await _hold_one_stream(service)
            assert await _refused_code(service, noise_chunks(stream='s2')) == \
                grpc.StatusCode.RESOURCE_EXHAUSTED
            release.set()
            await task
            assert svc.pool.free == 1
            out = [r async for r in service.Recognize(as_stream(noise_chunks(stream='s3')), FakeContext())]
            assert out[-1].is_final
        finally:
            if release is not None:
                release.set()
            svc.close()
    asyncio.run(go())


def test_bad_first_chunk_is_invalid_argument_even_when_the_pool_is_full():
    # review L3: channels / odd pcm of the first chunk were checked only after the pool,
    # so a full pool answered RESOURCE_EXHAUSTED, which the gateway retries
    async def go():
        svc = noise_service(size=1)
        release = None
        try:
            service = SpeechStreamService(None, noise=svc)
            release, task = await _hold_one_stream(service)
            for mutate in (_set(0, channels=2), _set(0, pcm=b'\x00')):
                assert await _refused_code(service, noise_chunks(stream='s2', mutate=mutate)) == \
                    grpc.StatusCode.INVALID_ARGUMENT
            release.set()
            await task
        finally:
            if release is not None:
                release.set()
            svc.close()
    asyncio.run(go())


def test_refused_streams_are_not_logged_one_by_one(caplog):
    # review L3: one WARNING per refused stream floods the log under overload
    async def go():
        svc = noise_service(size=1)
        release = None
        try:
            service = SpeechStreamService(None, noise=svc)
            release, task = await _hold_one_stream(service)
            for n in range(10):
                assert await _refused_code(service, noise_chunks(stream=f's{n}')) == \
                    grpc.StatusCode.RESOURCE_EXHAUSTED
            release.set()
            await task
        finally:
            if release is not None:
                release.set()
            svc.close()
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.noise'):
        asyncio.run(go())
    refused = [r for r in caplog.records if 'refused' in r.getMessage()]
    assert len(refused) == 3                         # the first 3, then every 100th
    assert 'denoisers in use' in refused[0].getMessage()


def test_slot_comes_back_after_a_contract_violation():
    svc = noise_service(size=1)
    try:
        with pytest.raises(Aborted):
            run_noise(noise_chunks(mutate=_set(2, pcm=b'\x00')), svc=svc)
        assert svc.pool.free == 1
    finally:
        svc.close()


class FailingDenoiser(FakeDenoiser):
    def run(self, samples, sample_rate):
        if self.runs >= 2:
            raise RuntimeError('onnx runtime error')
        return super().run(samples, sample_rate)


def test_model_failure_is_unavailable_and_the_denoiser_is_replaced():
    # review L2: the failed denoiser went back to the pool and was the next one handed out
    made = []

    def factory():
        made.append(FailingDenoiser())
        return made[-1]
    svc = noise_service(size=1, factory=factory)

    async def go():
        assert await _refused_code(SpeechStreamService(None, noise=svc), noise_chunks()) == \
            grpc.StatusCode.UNAVAILABLE
        await wait_until(lambda: svc.pool.free == 1)
    try:
        asyncio.run(go())
        assert len(made) == 2 and svc.pool._free == [made[1]] and made[0].resets == 0
    finally:
        svc.close()


class BadResetDenoiser(FakeDenoiser):
    def reset(self):
        raise RuntimeError('reset failed')


def test_reset_failure_does_not_spoil_the_stream_and_the_slot_is_refilled(caplog):
    # review L2/L3 (important): the reset error replaced the stream's outcome, skipped the
    # summary line and lost the slot for good
    svc = noise_service(size=1, factory=BadResetDenoiser)

    async def go():
        service = SpeechStreamService(None, noise=svc)
        out = [r async for r in service.Recognize(as_stream(noise_chunks()), FakeContext())]
        assert out[-1].is_final
        await wait_until(lambda: svc.pool.free == 1)
    try:
        with caplog.at_level(logging.INFO):
            asyncio.run(go())
    finally:
        svc.close()
    assert 'outcome=final' in caplog.text and 'reset failed' in caplog.text


def test_stream_after_close_is_unavailable():
    # review L3: the executor's RuntimeError escaped as an unknown error
    svc = noise_service()
    svc.close()
    assert noise_rejected(noise_chunks(), svc=svc) == grpc.StatusCode.UNAVAILABLE


class BlockingDenoiser(FakeDenoiser):
    """Blocks in run() until released; records a reset() that happens while run() is busy."""

    def __init__(self):
        super().__init__()
        self.started, self.go = threading.Event(), threading.Event()
        self.busy = False
        self.reset_while_busy = False

    def run(self, samples, sample_rate):
        self.busy = True
        try:
            self.started.set()
            self.go.wait(5)
            return super().run(samples, sample_rate)
        finally:
            self.busy = False

    def reset(self):
        self.reset_while_busy |= self.busy
        super().reset()


def test_cancelled_stream_gives_the_denoiser_back_only_after_the_running_call():
    async def go():
        svc = noise_service(size=1, factory=BlockingDenoiser)
        model = svc.pool._free[0]
        try:
            service = SpeechStreamService(None, noise=svc)

            async def consume():
                return [r async for r in service.Recognize(as_stream(noise_chunks()), FakeContext())]
            task = asyncio.create_task(consume())
            await asyncio.to_thread(model.started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert svc.pool.free == 0                   # the model call is still running
            model.go.set()
            await wait_until(lambda: svc.pool.free == 1)
            assert model.resets == 1 and not model.reset_while_busy
        finally:
            model.go.set()
            svc.close()
    asyncio.run(go())


def test_final_not_taken_by_the_client_is_logged_as_cancelled(caplog):
    # review L3: outcome=final was set before the final was handed over
    async def go():
        svc = noise_service()
        try:
            results = SpeechStreamService(None, noise=svc).Recognize(as_stream(noise_chunks()),
                                                                     FakeContext())
            async for r in results:
                if r.is_final:
                    break
            await results.aclose()
        finally:
            svc.close()
    with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.noise'):
        asyncio.run(go())
    assert 'outcome=cancelled' in caplog.text


# ----------------------------------------------------------------- backlog limit
async def paced(messages, interval_s, burst=1):
    """The messages at real-time pace; the first `burst` of them at once."""
    for n, message in enumerate(messages):
        yield message
        if n + 1 >= burst:
            await asyncio.sleep(interval_s)


class SlowDenoiser(FakeDenoiser):
    """200 ms of model time per call: 2.5 times slower than 80 ms chunks arrive."""

    def run(self, samples, sample_rate):
        time.sleep(0.2)
        return super().run(samples, sample_rate)


async def _outcome(service, requests):
    try:
        out = [r async for r in service.Recognize(requests, FakeContext())]
    except Aborted as err:
        return err.code, None
    return grpc.StatusCode.OK, out


def test_model_slower_than_real_time_ends_the_stream_at_the_backlog_limit(caplog):
    svc = noise_service(size=1, factory=SlowDenoiser, max_backlog_ms=300)

    async def go():
        code, _ = await _outcome(SpeechStreamService(None, noise=svc), paced(noise_chunks(), 0.08))
        await wait_until(lambda: svc.pool.free == 1)
        return code
    try:
        with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.noise'):
            assert asyncio.run(go()) == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        svc.close()
    assert 'noise_reduction backlog session=call-1' in caplog.text
    summary = [r.getMessage() for r in caplog.records if 'stream ended' in r.getMessage()][0]
    assert 'outcome=backlog' in summary
    assert int(summary.split('backlog_ms_max=')[1].split()[0]) > 300


def test_real_time_client_with_a_fast_model_stays_under_the_limit():
    svc = noise_service(max_backlog_ms=300)

    async def go():
        return await _outcome(SpeechStreamService(None, noise=svc), paced(noise_chunks(DATA[:16000]), 0.08))
    try:
        code, out = asyncio.run(go())
    finally:
        svc.close()
    assert code == grpc.StatusCode.OK and out[-1].is_final


@pytest.mark.parametrize('burst,code,backlog_ms', [(7, grpc.StatusCode.OK, 480),
                                                   (8, grpc.StatusCode.RESOURCE_EXHAUSTED, 560)])
def test_default_limit_and_a_burst_of_chunks(burst, code, backlog_ms, caplog):
    # A gateway that stalled sends what it queued at once. While the first chunk is in the
    # model, the others wait: 7 chunks -> 6 waiting = 480 ms, under 500; 8 -> 560 ms.
    # (review L5: this test claimed 480 ms but measured 320)
    svc = noise_service(max_backlog_ms=500)

    async def go():
        return await _outcome(SpeechStreamService(None, noise=svc),
                              paced(noise_chunks(DATA[:2560 * 10]), 0.08, burst=burst))
    try:
        with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.noise'):
            got, _ = asyncio.run(go())
    finally:
        svc.close()
    assert got == code
    summary = [r.getMessage() for r in caplog.records if 'stream ended' in r.getMessage()][0]
    assert f'backlog_ms_max={backlog_ms} ' in summary


def test_reading_stops_at_the_backlog_limit_while_the_model_is_busy():
    # review L5 (important): all the client's audio was read into memory during one long model
    # call; the limit was only checked between calls
    read = []

    async def fast_client():
        for message in noise_chunks(DATA * 10):         # 10 s, sent as fast as possible
            read.append(message.chunk_seq)
            yield message
    svc = noise_service(size=1, factory=BlockingDenoiser, max_backlog_ms=500)
    model = svc.pool._free[0]

    async def go():
        task = asyncio.create_task(_outcome(SpeechStreamService(None, noise=svc), fast_client()))
        await asyncio.to_thread(model.started.wait, 5)
        await asyncio.sleep(0.2)                        # time enough to read everything
        chunks_read = len(read)
        model.go.set()
        code, _ = await asyncio.wait_for(task, 10)
        return chunks_read, code
    try:
        chunks_read, code = asyncio.run(go())
    finally:
        model.go.set()
        svc.close()
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert chunks_read <= 10                            # the first, 7 waiting (560 ms), and one more


def test_without_a_limit_reading_still_stops_at_64_chunks():
    # review L5: max_backlog_ms 0 must not mean unlimited memory; the client is held by gRPC
    read = []

    async def fast_client():
        for message in noise_chunks(DATA * 10):
            read.append(message.chunk_seq)
            yield message
    svc = noise_service(size=1, factory=BlockingDenoiser, max_backlog_ms=0)
    model = svc.pool._free[0]

    async def go():
        task = asyncio.create_task(_outcome(SpeechStreamService(None, noise=svc), fast_client()))
        await asyncio.to_thread(model.started.wait, 5)
        await asyncio.sleep(0.2)
        chunks_read = len(read)
        model.go.set()
        code, out = await asyncio.wait_for(task, 20)
        return chunks_read, code, out
    try:
        chunks_read, code, out = asyncio.run(go())
    finally:
        model.go.set()
        svc.close()
    assert chunks_read <= 1 + 64 + 1
    assert code == grpc.StatusCode.OK and sum(len(r.audio_payload) for r in out) == len(DATA) * 10


def test_backlog_warnings_are_not_logged_one_by_one(caplog):
    # review L5: under a CPU overload every stream logs at once
    svc = noise_service(max_backlog_ms=100)

    async def go():
        service = SpeechStreamService(None, noise=svc)
        for n in range(10):                             # 5 chunks at once: 320 ms waiting
            code, _ = await _outcome(service, as_stream(noise_chunks(stream=f's{n}')[:5]))
            assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    try:
        with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.noise'):
            asyncio.run(go())
    finally:
        svc.close()
    assert len([r for r in caplog.records if 'noise_reduction backlog' in r.getMessage()]) == 3


class _ReaderKilled(BaseException):
    pass


def test_reader_ending_on_a_base_exception_does_not_leave_the_stream_waiting():
    # review L5: the reader put nothing in the queue and the stream waited forever
    async def killed():
        for message in noise_chunks()[:2]:
            yield message
        raise _ReaderKilled()
    svc = noise_service(size=1)

    async def go():
        stream = SpeechStreamService(None, noise=svc).Recognize(killed(), FakeContext())
        await asyncio.wait_for(_drain_all(stream), 5)
    try:
        with pytest.raises(_ReaderKilled):
            asyncio.run(go())
        assert svc.pool.free == 1
    finally:
        svc.close()


async def _drain_all(results):
    return [r async for r in results]


@pytest.mark.parametrize('index', [0, 2])
def test_chunk_above_64kb_is_refused_as_for_speech_to_text(index):
    # review L5: one huge chunk held a model thread for seconds, unseen by the backlog limit
    assert noise_rejected(noise_chunks(mutate=_set(index, pcm=b'\x00' * 64002))) == \
        grpc.StatusCode.RESOURCE_EXHAUSTED


def test_error_while_reading_requests_ends_the_stream_and_frees_the_slot(caplog):
    async def broken():
        for message in noise_chunks()[:2]:
            yield message
        raise RuntimeError('connection reset')
    svc = noise_service(size=1)

    async def go():
        [r async for r in SpeechStreamService(None, noise=svc).Recognize(broken(), FakeContext())]
    try:
        with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.noise'), \
                pytest.raises(RuntimeError, match='connection reset'):
            asyncio.run(go())
        assert svc.pool.free == 1
    finally:
        svc.close()
    # review L5: the outcome of the summary line was not checked
    assert 'outcome=read_error' in caplog.text


# -------------------------------------------------------------------------- routing
def test_noise_reduction_is_unimplemented_when_the_service_is_off():
    service = SpeechStreamService(make_service())

    async def go():
        return [r async for r in service.Recognize(as_stream(noise_chunks()), FakeContext())]
    with pytest.raises(Aborted) as err:
        asyncio.run(go())
    assert err.value.code == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.parametrize('task', ['', 'speech_to_text'])
def test_speech_to_text_is_unimplemented_when_asr_is_off(task):
    assert noise_rejected(chunks(ONE, task=task)) == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.parametrize('task,code', [('keyword', grpc.StatusCode.UNIMPLEMENTED),
                                       ('dance', grpc.StatusCode.INVALID_ARGUMENT)])
def test_other_tasks_keep_their_codes_when_asr_is_off(task, code):
    assert noise_rejected(chunks(ONE, task=task)) == code


def test_speech_to_text_results_do_not_change_when_noise_reduction_is_on():
    svc = noise_service()
    try:
        assert recognise(chunks(ONE + ONE), noise=svc) == recognise(chunks(ONE + ONE))
    finally:
        svc.close()


def test_both_tasks_run_side_by_side():
    svc = noise_service()
    service = SpeechStreamService(make_service(), noise=svc)

    async def one(messages):
        return [r async for r in service.Recognize(as_stream(messages), FakeContext())]

    async def go():
        return await asyncio.gather(one(chunks(ONE)), one(noise_chunks()))
    try:
        stt, noise = asyncio.run(go())
    finally:
        svc.close()
    assert stt[-1].text and not stt[-1].audio_payload
    assert noise[-1].is_final and sum(len(r.audio_payload) for r in noise) == len(DATA)


# ---------------------------------------------------------------- real gRPC server
async def with_noise_server(testfn, svc, core=None):
    server = grpc.aio.server()
    speech_pb2_grpc.add_SpeechStreamServicer_to_server(SpeechStreamService(core, noise=svc), server)
    port = server.add_insecure_port('127.0.0.1:0')
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
            await testfn(speech_pb2_grpc.SpeechStreamStub(channel))
    finally:
        await server.stop(0)


def test_over_grpc_results_and_client_cancel():
    svc = noise_service(size=1)

    async def check(stub):
        out = [r async for r in stub.Recognize(as_stream(noise_chunks()))]
        assert out[-1].is_final and sum(len(r.audio_payload) for r in out) == len(DATA)
        assert svc.pool.free == 1
        release = asyncio.Event()

        async def held_open():
            for message in noise_chunks(end=False)[:2]:
                yield message
            await release.wait()
        call = stub.Recognize(held_open())
        await call.read()
        assert svc.pool.free == 0
        call.cancel()
        await wait_until(lambda: svc.pool.free == 1)

        # Only the refused chunk: a Python aio client that is still writing after the server
        # ended the RPC reports INTERNAL instead of the server's status (speech_to_text too).
        bad = noise_chunks(mutate=_set(0, output_audio_format=5))[:1]
        with pytest.raises(grpc.aio.AioRpcError) as err:
            [r async for r in stub.Recognize(as_stream(bad))]
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    try:
        asyncio.run(with_noise_server(check, svc))
    finally:
        svc.close()


def test_stream_summary_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.noise'):
        run_noise(noise_chunks())
    line = [r.getMessage() for r in caplog.records if 'noise_reduction stream' in r.getMessage()]
    assert len(line) == 1
    assert 'outcome=final' in line[0] and 'samples_in=16000' in line[0] and 'samples_out=16000' in line[0]
    # review L3: time in the model and time waiting for a worker thread are told apart
    assert 'model_ms_p95=' in line[0] and 'queue_ms_p95=' in line[0] and 'samples_sent=16000' in line[0]
    assert 'backlog_ms_max=' in line[0]
