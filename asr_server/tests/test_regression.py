"""Regression tests for the defects fixed per docs/Giai_phap_fix_bug_server_hien_tai.md.

Each test fails on the server as it was before the fixes. No GPU or checkpoints:
the ASR and punctuation models are fakes, the VAD is the real endpointer with an
amplitude classifier.
"""
import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import grpc
import numpy as np

from asr_server import messages as pb
from asr_server.proto import speech_pb2
from asr_server.proto import speech_pb2_grpc
from asr_server.speech_service import SpeechStreamService
from asr_server.punctuation import PunctResult
from asr_server.server import ASRService
from asr_server.vad import WebRTCEndpointer


class FakeASR:
    """Contract of Qwen3ASRModel.LLM streaming calls, with delay / failure hooks."""

    def __init__(self, delay=0.0, fail_calls=(), empty_on_finish=False, text=None):
        self.infer_calls = 0
        self.states = 0
        self.delay = delay
        self.fail_calls = set(fail_calls)       # 1-based inference numbers that raise
        self.empty_on_finish = empty_on_finish
        self.text = text                        # None -> text derived from the state's own audio
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.threads = set()

    def init_streaming_state(self, language, unfixed_chunk_num, unfixed_token_num, chunk_size_sec):
        self.states += 1
        return SimpleNamespace(text='', language=language, chunk_id=0,
                               buffer=np.empty(0, dtype=np.float32), level=0.0,
                               chunk_size_samples=round(chunk_size_sec * 16000))

    def _infer(self, state):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.infer_calls += 1
            n = self.infer_calls
            self.threads.add(threading.current_thread().name)
        try:
            if self.delay:
                time.sleep(self.delay)
            if n in self.fail_calls:
                raise RuntimeError('simulated CUDA/vLLM failure')
            state.chunk_id += 1
            state.text = self.text if self.text is not None else f'level {round(state.level * 100)}'
        finally:
            with self._lock:
                self.active -= 1

    def streaming_transcribe(self, samples, state):
        if samples.size:
            state.level = max(state.level, float(np.max(np.abs(samples))))
        state.buffer = np.concatenate([state.buffer, samples])
        while len(state.buffer) >= state.chunk_size_samples:
            state.buffer = state.buffer[state.chunk_size_samples:]
            self._infer(state)

    def finish_streaming_transcribe(self, state):
        if len(state.buffer):
            state.buffer = np.empty(0, np.float32)
            self._infer(state)
        if self.empty_on_finish:
            state.text = ''


class FakePunctuator:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0

    def restore(self, text):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return PunctResult(text + '?', '?', 0.999) if text else PunctResult('')


class Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(f'{code}: {details}')
        self.code = code


class FakeContext:
    def cancelled(self):
        return False

    async def abort(self, code, details):
        raise Aborted(code, details)


def fake_vad(sr):
    return WebRTCEndpointer(sr, classifier=lambda x, r: float(np.max(np.abs(x))) > 0.05)


def pcm(seconds, voiced, level=0.20):
    value = int(32767 * level) if voiced else 0
    return (np.ones(int(round(seconds * 16000))) * value).astype('<i2').tobytes()


def cfg_msg(call='c1', leg='a'):
    return pb.StreamRequest(config=pb.StreamConfig(
        call_id=call, leg_id=leg, sample_rate=16000, encoding=pb.PCM_S16LE, language='Vietnamese'))


async def frames_req(frames, call='c1', end=True, chunk_bytes=640):
    yield cfg_msg(call)
    n = 0
    for payload in frames:
        for p in range(0, len(payload), chunk_bytes):
            n += 1
            yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n, data=payload[p:p + chunk_bytes]))
    if end:
        yield pb.StreamRequest(end_stream=pb.EndStream())


def make_service(model=None, punct=None, **options):
    # Tests send audio much faster than real time, which the ingress limit is meant to reject.
    options.setdefault('ingress_max_ms', 0)
    # ... and would make every PARTIAL inference be skipped as backlog.
    options.setdefault('partial_max_backlog_ms', 0)
    return ASRService(model or FakeASR(text='xin chao anh'), punct or FakePunctuator(),
                      vad_factory=fake_vad, chunk_sec=options.pop('chunk_sec', 0.4), **options)


async def collect(service, requests):
    return [r async for r in service.StreamRecognize(requests, FakeContext())]


async def with_server(testfn, service):
    """Real gRPC server speaking speech.proto in front of the core `service`."""
    server = grpc.aio.server()
    speech_pb2_grpc.add_SpeechStreamServicer_to_server(SpeechStreamService(service), server)
    port = server.add_insecure_port('127.0.0.1:0')
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
            await testfn(speech_pb2_grpc.SpeechStreamStub(channel))
    finally:
        await server.stop(0)


async def speech_req(frames, session='c1', end=True, chunk_bytes=2560):
    """AudioChunk stream: PCM16 16 kHz mono, 80 ms chunks, RTP clock 8000."""
    data = b''.join(frames)
    pieces = [data[p:p + chunk_bytes] for p in range(0, len(data), chunk_bytes)]
    for n, piece in enumerate(pieces):
        yield speech_pb2.AudioChunk(
            session_id=session, stream_id=session + '-a2b', pcm=piece, sample_rate=16000,
            channels=1, rtp_timestamp=n * 640, duration_ms=80, rtp_clock_rate=8000,
            chunk_seq=n, language='vi', task='speech_to_text',
            end_of_stream=(end and n == len(pieces) - 1))


def kinds(results):
    return ['PARTIAL' if r.event == pb.PARTIAL else 'COMMIT' for r in results]


# --------------------------------------------------------------------------- B1
def test_b1_reading_does_not_wait_for_the_model():
    async def run():
        model = FakeASR(delay=0.30, text='xin chao anh')
        service = make_service(model)
        pulls = []

        async def reqs():
            async for r in frames_req([pcm(1.2, True), pcm(0.8, False)]):
                pulls.append(time.perf_counter())
                yield r

        out = await collect(service, reqs())
        assert float(np.max(np.diff(pulls))) < 0.10        # was ~0.30 at every inference
        assert kinds(out)[-1] == 'COMMIT'
    asyncio.run(run())


def test_b1_overflow_closes_stream_with_resource_exhausted():
    async def run():
        service = make_service(FakeASR(delay=0.5, text='xin chao anh'), ingress_max_ms=1000)
        try:
            await collect(service, frames_req([pcm(6.0, True)]))
        except Aborted as err:
            assert err.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        else:
            raise AssertionError('expected RESOURCE_EXHAUSTED')
    asyncio.run(run())


def test_b1_unlimited_ingress_when_zero():
    async def run():
        service = make_service(FakeASR(delay=0.05, text='xin chao anh'), ingress_max_ms=0)
        out = await collect(service, frames_req([pcm(6.0, True), pcm(0.8, False)]))
        assert kinds(out)[-1] == 'COMMIT'
    asyncio.run(run())


# --------------------------------------------------------------------------- B2
async def _drain(call):
    try:
        return [r async for r in call]
    except (grpc.aio.AioRpcError, asyncio.CancelledError):
        return []


def test_b2_cancel_during_inference_never_overlaps_model_calls():
    model = FakeASR(delay=0.6, text='xin chao anh')

    async def test(stub):
        call_a = stub.Recognize(speech_req([pcm(3.0, True)], session='A', end=False))
        task_a = asyncio.ensure_future(_drain(call_a))
        await asyncio.sleep(0.25)          # A is inside the first slow inference
        call_b = stub.Recognize(speech_req([pcm(1.0, True), pcm(0.8, False)], session='B'))
        task_b = asyncio.ensure_future(_drain(call_b))
        await asyncio.sleep(0.10)
        call_a.cancel()
        _, res_b = await asyncio.gather(task_a, task_b, return_exceptions=True)
        assert model.max_active == 1       # was 2
        assert [(r.session_id, r.raw_text) for r in res_b if r.is_final] == [('B', 'xin chao anh')]
    asyncio.run(with_server(test, make_service(model)))


def test_b2_all_model_calls_use_one_thread_and_results_stay_with_their_stream():
    model = FakeASR(delay=0.01)            # text = level of the state's own audio

    async def run():
        service = make_service(model)

        async def one(i):
            level = 0.10 + 0.05 * i
            out = await collect(service, frames_req(
                [pcm(1.0, True, level), pcm(0.8, False), pcm(0.9, True, level), pcm(0.8, False)],
                call=f'c{i}'))
            return level, out
        for level, out in await asyncio.gather(*[one(i) for i in range(12)]):
            commits = [r for r in out if r.event == pb.COMMIT]
            assert len(commits) == 2
            assert all(r.raw_text == f'level {round(level * 100)}' for r in out if r.raw_text)
            assert {r.call_id for r in out} == {out[0].call_id}
        assert model.max_active == 1
        assert len(model.threads) == 1 and next(iter(model.threads)).startswith('asr-infer')
    asyncio.run(run())


# --------------------------------------------------------------------------- B3
def test_b3_commit_when_audio_stops_arriving(caplog):
    async def run():
        service = make_service(audio_timeout_ms=300)
        marks = {}

        async def reqs():
            async for r in frames_req([pcm(1.0, True)], end=False):
                yield r
            marks['last_audio'] = time.perf_counter()
            await asyncio.sleep(1.5)       # sender is silent and sends no frames
            yield pb.StreamRequest(end_stream=pb.EndStream())

        commits = []
        async for r in service.StreamRecognize(reqs(), FakeContext()):
            if r.event == pb.COMMIT:
                commits.append((time.perf_counter() - marks['last_audio'], r.reason, r.raw_text))
        assert len(commits) == 1
        delay, reason, raw = commits[0]
        assert 0.25 <= delay < 1.0         # was 1.5 s, only at end_stream
        assert reason == 'vad_endpoint' and raw == 'xin chao anh'
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        asyncio.run(run())
    assert any('audio_timeout' in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_b3_timer_disabled_when_zero():
    async def run():
        service = make_service(audio_timeout_ms=0)

        async def reqs():
            async for r in frames_req([pcm(1.0, True)], end=False):
                yield r
            await asyncio.sleep(0.6)
            yield pb.StreamRequest(end_stream=pb.EndStream())

        out = await collect(service, reqs())
        assert [r.reason for r in out if r.event == pb.COMMIT] == ['end_stream']
    asyncio.run(run())


# --------------------------------------------------------------------------- B4
def test_b4_forced_split_pieces_are_contiguous():
    async def run():
        service = make_service(max_utterance_sec=1.0)
        out = await collect(service, frames_req([pcm(3.0, True), pcm(0.8, False)]))
        commits = [(r.reason, r.start_ms, r.end_ms) for r in out if r.event == pb.COMMIT]
        assert [c[0] for c in commits[:3]] == ['max_duration'] * 3
        assert commits[0][1:] == (0, 1000)
        for prev, cur in zip(commits, commits[1:]):
            assert cur[1] == prev[2]       # was: every piece started at 0
        assert [r.utterance_index for r in out if r.event == pb.COMMIT] == [1, 2, 3, 4]
    asyncio.run(run())


# --------------------------------------------------------------------------- B5
CLICKS = [pcm(0.3, False), pcm(0.06, True), pcm(0.5, False), pcm(0.06, True), pcm(1.0, False)]


def test_b5_no_partial_for_an_utterance_the_vad_discards():
    async def run():
        model = FakeASR(text='xin chao anh')
        out = await collect(make_service(model, chunk_sec=1.0), frames_req(CLICKS))
        assert out == []                   # was: one PARTIAL and no COMMIT
        assert model.infer_calls >= 1      # the model did run; only the PARTIAL is withheld
    asyncio.run(run())


def test_b5_empty_final_clears_the_partial_with_an_empty_partial(caplog):
    async def run():
        service = make_service(FakeASR(text='xin chao anh', empty_on_finish=True))
        out = await collect(service, frames_req(
            [pcm(1.2, True), pcm(0.8, False), pcm(1.2, True), pcm(0.8, False)]))
        assert 'COMMIT' not in kinds(out)
        first = [r for r in out if r.utterance_index == 1]
        assert first[0].text == 'xin chao anh?'
        # every utterance ends with an empty PARTIAL; the index is reused, revision keeps growing
        texts = [r.text for r in out]
        assert texts[-1] == '' and texts.count('') == 2
        assert {r.utterance_index for r in out} == {1}
        revisions = [r.revision for r in out]
        assert revisions == sorted(set(revisions))     # strictly increasing, was 1 -> 1
        assert all(r.reason == 'partial' for r in out)
    with caplog.at_level(logging.DEBUG, logger='qwen3_asr.grpc.vad'):
        asyncio.run(run())
    assert any('empty final text' in r.getMessage() and r.levelno == logging.DEBUG
               for r in caplog.records)


# --------------------------------------------------------------------------- B7
TWO_UTTERANCES = [pcm(1.2, True), pcm(0.8, False), pcm(1.2, True), pcm(0.8, False)]


def test_b7_model_error_loses_one_utterance_not_the_stream(caplog):
    async def run():
        model = FakeASR(text='xin chao anh', fail_calls={2})   # fails inside the first utterance
        out = await collect(make_service(model), frames_req(TWO_UTTERANCES))
        commits = [r for r in out if r.event == pb.COMMIT]
        assert len(commits) == 1 and commits[0].raw_text == 'xin chao anh'
        assert commits[0].reason == 'vad_endpoint'
        # the failed utterance had shown a PARTIAL: it is cleared by an empty PARTIAL
        before = [r for r in out if r.event == pb.PARTIAL][:2]
        assert [r.text for r in before] == ['xin chao anh?', '']
        assert all(r.reason in ('partial', 'vad_endpoint') for r in out)   # no new reason values
    with caplog.at_level(logging.DEBUG, logger='qwen3_asr.grpc.vad'):
        asyncio.run(run())
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and 'asr error' in errors[0].getMessage()


def test_b7_stream_closes_unavailable_after_consecutive_errors():
    async def run():
        model = FakeASR(text='xin chao anh', fail_calls=set(range(1, 100)))
        service = make_service(model, max_consecutive_errors=3)
        frames = []
        for _ in range(5):
            frames += [pcm(1.0, True), pcm(0.8, False)]
        try:
            await collect(service, frames_req(frames))
        except Aborted as err:
            assert err.code == grpc.StatusCode.UNAVAILABLE
            assert model.infer_calls == 3
        else:
            raise AssertionError('expected UNAVAILABLE')
    asyncio.run(run())


def test_b7_error_counter_resets_after_a_good_inference():
    async def run():
        # utterances 1, 2 and 4 fail, utterance 3 succeeds: never three failures in a row
        model = FakeASR(text='xin chao anh', fail_calls={1, 2, 6})
        service = make_service(model, max_consecutive_errors=3, chunk_sec=0.9)
        frames = []
        for _ in range(4):
            frames += [pcm(1.0, True), pcm(0.8, False)]
        out = await collect(service, frames_req(frames))
        assert len([r for r in out if r.event == pb.COMMIT]) >= 1
    asyncio.run(run())


# --------------------------------------------------------------------------- B8
def test_b8_slow_punctuation_does_not_hold_results_back():
    async def run():
        punct = FakePunctuator(delay=0.6)
        service = make_service(punct=punct, punct_partial_timeout_ms=80, punct_commit_timeout_ms=150)
        t0 = time.perf_counter()
        out = await collect(service, frames_req([pcm(1.2, True), pcm(0.8, False)]))
        elapsed = time.perf_counter() - t0
        partials = [r for r in out if r.event == pb.PARTIAL]
        commit = [r for r in out if r.event == pb.COMMIT][0]
        assert partials and all(r.text == 'xin chao anh' for r in partials)   # raw text, on time
        assert commit.text == commit.raw_text == 'xin chao anh'
        assert elapsed < 0.6               # was: every result waited for punctuation
    asyncio.run(run())


def test_b8_punctuation_error_falls_back_to_raw_text():
    class Broken:
        def restore(self, text):
            raise RuntimeError('tokenizer exploded')

    async def run():
        out = await collect(make_service(punct=Broken()), frames_req([pcm(1.2, True), pcm(0.8, False)]))
        commit = [r for r in out if r.event == pb.COMMIT][0]
        assert commit.text == commit.raw_text == 'xin chao anh'
    asyncio.run(run())


def test_b8_fast_punctuation_still_formats_partial_and_commit():
    async def run():
        out = await collect(make_service(), frames_req([pcm(1.2, True), pcm(0.8, False)]))
        assert [r.text for r in out] == ['xin chao anh?'] * len(out)
        assert kinds(out)[-1] == 'COMMIT' and 'PARTIAL' in kinds(out)
    asyncio.run(run())


# ------------------------------------------------------------- marker stripping
def test_asr_markers_are_removed_before_punctuation_and_commit():
    async def run():
        model = FakeASR(text='language Vietnamese<asr_text>xin chao language: Vietnamese<asr_text>anh')
        out = await collect(make_service(model), frames_req([pcm(1.2, True), pcm(0.8, False)]))
        assert out
        for r in out:
            assert r.raw_text == 'xin chao anh' and r.text == 'xin chao anh?'
    asyncio.run(run())


def test_marker_only_output_is_treated_as_empty():
    async def run():
        model = FakeASR(text='language Vietnamese<asr_text>')
        out = await collect(make_service(model), frames_req([pcm(1.2, True), pcm(0.8, False)]))
        assert out == []
    asyncio.run(run())


# ------------------------------------------------- language passed to punctuation
class LanguagePunctuator:
    def __init__(self):
        self.seen = []

    def restore(self, text, language=None):
        self.seen.append(language)
        return PunctResult(text + '.', '.', 0.99) if text else PunctResult('')


async def lang_req(language, frames):
    yield pb.StreamRequest(config=pb.StreamConfig(
        call_id='c1', leg_id='a', sample_rate=16000, encoding=pb.PCM_S16LE, language=language))
    n = 0
    for payload in frames:
        for p in range(0, len(payload), 640):
            n += 1
            yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n, data=payload[p:p + 640]))
    yield pb.StreamRequest(end_stream=pb.EndStream())


def test_punctuation_receives_the_language_of_the_utterance():
    async def run(language, expected):
        punct = LanguagePunctuator()
        out = await collect(make_service(punct=punct), lang_req(language, [pcm(1.2, True), pcm(0.8, False)]))
        assert kinds(out)[-1] == 'COMMIT' and 'PARTIAL' in kinds(out)
        assert punct.seen and set(punct.seen) == {expected}       # PARTIAL and COMMIT alike
        assert {r.language for r in out} == {expected}            # same value the client sees
    asyncio.run(run('English', 'English'))
    asyncio.run(run('', 'Vietnamese'))       # no language from the client: server default


def test_punctuation_receives_the_language_the_model_detected():
    class DetectingASR(FakeASR):
        def _infer(self, state):
            super()._infer(state)
            state.language = 'English'       # the model reports what it heard

    async def run():
        punct = LanguagePunctuator()
        out = await collect(make_service(DetectingASR(text='hello there friend'), punct=punct),
                            lang_req('Vietnamese', [pcm(1.2, True), pcm(0.8, False)]))
        assert set(punct.seen) == {'English'}
        assert {r.language for r in out} == {'English'}
    asyncio.run(run())


def test_punctuator_without_language_argument_still_works():
    punct = FakePunctuator()                 # restore(self, text): no language parameter
    assert make_service(punct=punct)._punct_takes_language is False

    async def run():
        out = await collect(make_service(punct=punct), frames_req([pcm(1.2, True), pcm(0.8, False)]))
        assert [r.text for r in out if r.event == pb.COMMIT] == ['xin chao anh?']
    asyncio.run(run())
