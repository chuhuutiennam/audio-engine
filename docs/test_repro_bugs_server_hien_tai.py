"""Reproduction tests against the UNMODIFIED qwen3_asr_grpc_vad_commit server.

Each test asserts the CURRENT (buggy) behaviour, so a pass = bug reproduced.
Run:  PYTHONPATH=<server dir> python -m pytest -q -s test_repro.py
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import grpc
import numpy as np

import asr_pb2 as pb
import asr_pb2_grpc as pb_grpc
from punctuation import PunctResult
from server import ASRService
from vad import WebRTCEndpointer


class FakeASR:
    """Same contract as tests/test_grpc.py FakeASR, with optional delay/failure hooks."""

    def __init__(self, delay=0.0, fail_on_call=None, empty_on_finish=False):
        self.infer_calls = 0
        self.states = 0
        self.delay = delay
        self.fail_on_call = fail_on_call
        self.empty_on_finish = empty_on_finish
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def init_streaming_state(self, language, unfixed_chunk_num, unfixed_token_num, chunk_size_sec):
        self.states += 1
        return SimpleNamespace(text='', language=language, chunk_id=0,
                               buffer=np.empty(0, dtype=np.float32),
                               chunk_size_samples=round(chunk_size_sec * 16000))

    def _infer(self, state):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.infer_calls += 1
            n = self.infer_calls
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail_on_call is not None and n >= self.fail_on_call:
                raise RuntimeError('simulated CUDA/vLLM failure')
            state.chunk_id += 1
            state.text = 'xin chao anh'
        finally:
            with self._lock:
                self.active -= 1

    def streaming_transcribe(self, samples, state):
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
    def restore(self, text):
        return PunctResult(text + '?', '?', 0.999) if text else PunctResult('')


def fake_vad(sr):
    return WebRTCEndpointer(sr, classifier=lambda x, r: float(np.max(np.abs(x))) > 0.05)


def pcm(seconds, voiced):
    return (np.ones(int(round(seconds * 16000))) * int(32767 * 0.20 if voiced else 0)).astype('<i2').tobytes()


def cfg_msg(call='c1', leg='a'):
    return pb.StreamRequest(config=pb.StreamConfig(
        call_id=call, leg_id=leg, sample_rate=16000, encoding=pb.PCM_S16LE, language='Vietnamese'))


async def frames_req(frames, call='c1', end=True, chunk_bytes=640, pace=0.0):
    yield cfg_msg(call)
    n = 0
    for payload in frames:
        for p in range(0, len(payload), chunk_bytes):
            n += 1
            yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n, data=payload[p:p + chunk_bytes]))
            if pace:
                await asyncio.sleep(pace)
    if end:
        yield pb.StreamRequest(end_stream=pb.EndStream())


async def with_server(testfn, model=None, **options):
    model = model or FakeASR()
    service = ASRService(model, FakePunctuator(), vad_factory=fake_vad,
                         chunk_sec=options.pop('chunk_sec', 0.4), **options)
    server = grpc.aio.server()
    pb_grpc.add_ASRServicer_to_server(service, server)
    port = server.add_insecure_port('127.0.0.1:0')
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
            await testfn(pb_grpc.ASRStub(channel), model, service)
    finally:
        await server.stop(0)


class FakeContext:
    def cancelled(self):
        return False

    async def abort(self, code, details):
        raise RuntimeError(f'abort {code}: {details}')


# --------------------------------------------------------------------------- B1
def test_B1_reader_is_blocked_while_model_runs():
    """The gRPC read loop does not pull the next chunk while inference runs."""
    async def run():
        model = FakeASR(delay=0.30)
        service = ASRService(model, FakePunctuator(), vad_factory=fake_vad, chunk_sec=0.4)
        pulls = []

        async def reqs():
            async for r in frames_req([pcm(1.2, True), pcm(0.8, False)]):
                pulls.append(time.perf_counter())
                yield r

        out = [r async for r in service.StreamRecognize(reqs(), FakeContext())]
        gaps = np.diff(pulls)
        stalls = [round(float(g), 2) for g in gaps if g > 0.2]
        print(f'\n[B1] chunks pulled={len(pulls)}, stalls >200ms between two reads: {stalls}')
        assert len(stalls) >= 2 and out
    asyncio.run(run())


def test_B1_streams_are_serialized_by_global_lock():
    async def test(stub, model, service):
        async def one(i):
            t0 = time.perf_counter()
            res = [r async for r in stub.StreamRecognize(
                frames_req([pcm(1.2, True), pcm(0.8, False)], call=f'c{i}'))]
            return time.perf_counter() - t0, res
        t1, _ = await one(0)
        t0 = time.perf_counter()
        many = await asyncio.gather(*[one(i) for i in range(1, 5)])
        t4 = time.perf_counter() - t0
        print(f'\n[B1] 1 stream: {t1:.2f}s; 4 streams in parallel: {t4:.2f}s (max_active={model.max_active})')
        assert model.max_active == 1 and t4 > 3.0 * t1
    asyncio.run(with_server(test, model=FakeASR(delay=0.15)))


# --------------------------------------------------------------------------- B2
def test_B2_cancel_during_inference_allows_concurrent_model_calls():
    async def test(stub, model, service):
        call_a = stub.StreamRecognize(frames_req([pcm(3.0, True)], call='A', end=False))
        task_a = asyncio.ensure_future(_drain(call_a))
        await asyncio.sleep(0.25)          # A is inside the first slow inference
        call_b = stub.StreamRecognize(frames_req([pcm(1.0, True), pcm(0.8, False)], call='B'))
        task_b = asyncio.ensure_future(_drain(call_b))
        await asyncio.sleep(0.10)          # B is now waiting for infer_lock
        call_a.cancel()
        await asyncio.gather(task_a, task_b, return_exceptions=True)
        print(f'\n[B2] max concurrent model calls observed: {model.max_active}')
        assert model.max_active >= 2
    asyncio.run(with_server(test, model=FakeASR(delay=0.6)))


async def _drain(call):
    try:
        return [r async for r in call]
    except (grpc.aio.AioRpcError, asyncio.CancelledError):
        return []


# --------------------------------------------------------------------------- B3
def test_B3_no_commit_while_audio_stops_arriving():
    """DTX-like: speech, then the sender goes silent WITHOUT sending frames."""
    async def run():
        model = FakeASR()
        service = ASRService(model, FakePunctuator(), vad_factory=fake_vad, chunk_sec=0.4)
        t_last_audio = None

        async def reqs():
            nonlocal t_last_audio
            async for r in frames_req([pcm(1.0, True)], end=False):
                yield r
            t_last_audio = time.perf_counter()
            await asyncio.sleep(2.0)       # no frames at all for 2 s
            yield pb.StreamRequest(end_stream=pb.EndStream())

        commits = []
        async for r in service.StreamRecognize(reqs(), FakeContext()):
            if r.event == pb.COMMIT:
                commits.append((time.perf_counter() - t_last_audio, r.reason))
        print(f'\n[B3] COMMIT arrived {commits[0][0]:.2f}s after last audio, reason={commits[0][1]}')
        assert commits[0][1] == 'end_stream' and commits[0][0] >= 1.9
    asyncio.run(run())


# --------------------------------------------------------------------------- B4
def test_B4_forced_split_reuses_original_start_ms():
    async def test(stub, model, service):
        res = [r async for r in stub.StreamRecognize(frames_req([pcm(3.0, True), pcm(0.8, False)]))]
        commits = [(r.reason, r.start_ms, r.end_ms) for r in res if r.event == pb.COMMIT]
        print(f'\n[B4] commits (reason, start_ms, end_ms): {commits}')
        assert len(commits) >= 3
        assert commits[0][1] == commits[1][1] == commits[2][1]      # same start for every piece
        assert commits[1][1] < commits[0][2]                         # overlaps previous piece
    asyncio.run(with_server(test, max_utterance_sec=1.0))


# --------------------------------------------------------------------------- B5
def test_B5a_partial_then_discard_leaves_orphan_partial():
    async def test(stub, model, service):
        res = [r async for r in stub.StreamRecognize(frames_req(
            [pcm(0.3, False), pcm(0.06, True), pcm(0.5, False), pcm(0.06, True), pcm(1.0, False)]))]
        kinds = [('PARTIAL' if r.event == pb.PARTIAL else 'COMMIT', r.utterance_index, r.text) for r in res]
        print(f'\n[B5a] default chunk_sec=1.0 -> events: {kinds}')
        assert any(k[0] == 'PARTIAL' for k in kinds) and not any(k[0] == 'COMMIT' for k in kinds)
    asyncio.run(with_server(test, chunk_sec=1.0))


def test_B5b_partial_then_empty_final_leaves_orphan_partial():
    async def test(stub, model, service):
        res = [r async for r in stub.StreamRecognize(frames_req([pcm(1.2, True), pcm(0.8, False)]))]
        kinds = [('PARTIAL' if r.event == pb.PARTIAL else 'COMMIT', r.text) for r in res]
        print(f'\n[B5b] events: {kinds}')
        assert any(k[0] == 'PARTIAL' for k in kinds) and not any(k[0] == 'COMMIT' for k in kinds)
    asyncio.run(with_server(test, model=FakeASR(empty_on_finish=True)))


def test_B5c_utterance_index_reused_after_orphan():
    async def test(stub, model, service):
        res = [r async for r in stub.StreamRecognize(frames_req(
            [pcm(0.3, False), pcm(0.06, True), pcm(0.5, False), pcm(0.06, True), pcm(1.0, False),
             pcm(1.2, True), pcm(0.8, False)]))]
        kinds = [('PARTIAL' if r.event == pb.PARTIAL else 'COMMIT', r.utterance_index, r.revision) for r in res]
        print(f'\n[B5c] events (kind, utterance_index, revision): {kinds}')
        idx = {k[1] for k in kinds}
        assert idx == {1}       # the discarded utterance and the next real one share index 1
    asyncio.run(with_server(test, chunk_sec=1.0))


# --------------------------------------------------------------------------- B6
def test_B6_single_sequence_gap_kills_stream():
    async def test(stub, model, service):
        async def reqs():
            yield cfg_msg()
            data = pcm(1.2, True) + pcm(0.8, False)
            n = 0
            for p in range(0, len(data), 640):
                n += 1
                if n == 20:
                    continue               # one 20 ms chunk dropped by the sender
                yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n, data=data[p:p + 640]))
            yield pb.StreamRequest(end_stream=pb.EndStream())
        got = []
        try:
            async for r in stub.StreamRecognize(reqs()):
                got.append(r)
            assert False, 'expected abort'
        except grpc.aio.AioRpcError as err:
            print(f'\n[B6] stream aborted: {err.code().name}; results before abort: {len(got)}; commits: '
                  f'{sum(1 for r in got if r.event == pb.COMMIT)}')
            # Server aborts with INVALID_ARGUMENT, but a client that is still writing may see INTERNAL.
            assert err.code() in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.INTERNAL)
            assert not any(r.event == pb.COMMIT for r in got)
    asyncio.run(with_server(test))


# --------------------------------------------------------------------------- B7
def test_B7_model_exception_kills_stream():
    async def test(stub, model, service):
        got = []
        try:
            async for r in stub.StreamRecognize(frames_req(
                    [pcm(1.2, True), pcm(0.8, False), pcm(1.2, True), pcm(0.8, False)])):
                got.append(r)
            assert False, 'expected failure'
        except grpc.aio.AioRpcError as err:
            print(f'\n[B7] RPC failed with {err.code().name}; commits delivered before failure: '
                  f'{sum(1 for r in got if r.event == pb.COMMIT)}')
            assert err.code() in (grpc.StatusCode.UNKNOWN, grpc.StatusCode.INTERNAL)
    asyncio.run(with_server(test, model=FakeASR(fail_on_call=5)))


# --------------------------------------------------------------------------- B8
def test_B8_partial_punctuation_serialized_and_in_read_path():
    class SlowPunct(FakePunctuator):
        def __init__(self):
            self.calls = 0
        def restore(self, text):
            self.calls += 1
            time.sleep(0.10)
            return super().restore(text)

    async def run():
        punct = SlowPunct()
        services = ASRService(FakeASR(), punct, vad_factory=fake_vad, chunk_sec=0.4)

        async def one(i):
            return [r async for r in services.StreamRecognize(
                frames_req([pcm(1.2, True), pcm(0.8, False)], call=f'c{i}'), FakeContext())]
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(i) for i in range(4)])
        dt = time.perf_counter() - t0
        partials = sum(1 for rs in res for r in rs if r.event == pb.PARTIAL)
        print(f'\n[B8] punct calls={punct.calls} (partials={partials}, commits=4); wall={dt:.2f}s '
              f'~= calls x 0.10s -> serialized')
        assert punct.calls == partials + 4 and dt >= punct.calls * 0.10 * 0.9
    asyncio.run(run())
