"""Native vLLM backend + Silero endpointer inside the fixed ASRService.

No GPU, no vllm, no Silero checkpoint: the LLM is a fake with the vllm.LLM.chat()
shape, and the Silero model is either the amplitude classifier or a tiny
TorchScript stand-in loaded through the real torch.jit path.
"""
import asyncio
import base64
import io
import logging
import threading
import wave
from types import SimpleNamespace

import grpc
import numpy as np

from asr_server.asr_backend import VLLMASRBackend
from asr_server import messages as pb
from asr_server.punctuation import PunctResult
from asr_server.server import ASRService
from asr_server.vad import SileroEndpointer


class FakeVLLM:
    """vllm.LLM.chat() stand-in. `answers` is one string, or a function of the peak
    amplitude (0..1) of the WAV it was given."""

    def __init__(self, answers='language Vietnamese<asr_text>xin chao anh', fail_calls=()):
        self.answers = answers
        self.fail_calls = set(fail_calls)
        self.calls = 0
        self.threads = set()

    def chat(self, conversation, sampling_params, use_tqdm=True):
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        if self.calls in self.fail_calls:
            raise RuntimeError('simulated vLLM failure')
        part = conversation[0]['content'][0]
        assert part['type'] == 'audio_url'
        if isinstance(self.answers, str):
            text = self.answers
        else:
            payload = base64.b64decode(part['audio_url']['url'].split(',', 1)[1])
            with wave.open(io.BytesIO(payload), 'rb') as w:
                samples = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
            text = self.answers(float(np.max(np.abs(samples))) / 32767.0)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]


class LanguagePunctuator:
    def __init__(self):
        self.seen = []

    def restore(self, text, language=None):
        self.seen.append((text, language))
        return PunctResult(text + '.', '.', 0.99) if text else PunctResult('')


class Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(f'{code}: {details}')
        self.code = code


class FakeContext:
    def cancelled(self):
        return False

    async def abort(self, code, details):
        raise Aborted(code, details)


def silero_classifier_vad(sr):
    return SileroEndpointer(sr, classifier=lambda x, r: float(np.max(np.abs(x))) > 0.05)


def pcm(seconds, voiced, level=0.20):
    value = int(32767 * level) if voiced else 0
    return (np.ones(int(round(seconds * 16000))) * value).astype('<i2').tobytes()


async def requests(frames, language='Vietnamese', chunk_bytes=640):
    yield pb.StreamRequest(config=pb.StreamConfig(
        call_id='c1', leg_id='a', sample_rate=16000, encoding=pb.PCM_S16LE, language=language))
    n = 0
    for payload in frames:
        for p in range(0, len(payload), chunk_bytes):
            n += 1
            yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n, data=payload[p:p + chunk_bytes]))
    yield pb.StreamRequest(end_stream=pb.EndStream())


def make_service(llm, punct=None, **options):
    options.setdefault('ingress_max_ms', 0)        # tests send faster than real time
    options.setdefault('vad_factory', silero_classifier_vad)
    return ASRService(VLLMASRBackend(llm=llm), punct or LanguagePunctuator(),
                      chunk_sec=options.pop('chunk_sec', 0.4), **options)


async def collect(service, reqs):
    return [r async for r in service.StreamRecognize(reqs, FakeContext())]


ONE = [pcm(1.2, True), pcm(0.8, False)]


# ------------------------------------------------------------ marker and language
def test_backend_splits_marker_into_language_and_text():
    model = VLLMASRBackend(llm=FakeVLLM('language English<asr_text>hello there friend'))
    state = model.init_streaming_state(language='Vietnamese', chunk_size_sec=1.0)
    assert state.language == 'Vietnamese'              # configured value until the model answers
    model.streaming_transcribe(np.full(16000, 0.25, np.float32), state)
    assert state.text == 'hello there friend'
    assert state.language == 'English'                 # detected, taken from the marker
    assert state.raw_output == 'language English<asr_text>hello there friend'


def test_backend_keeps_configured_language_when_model_names_none():
    model = VLLMASRBackend(llm=FakeVLLM('language None<asr_text>'))
    state = model.init_streaming_state(language='Vietnamese', chunk_size_sec=1.0)
    model.streaming_transcribe(np.full(16000, 0.25, np.float32), state)
    assert state.text == '' and state.language == 'Vietnamese'


def test_detected_language_reaches_client_and_punctuation():
    async def run():
        punct = LanguagePunctuator()
        service = make_service(FakeVLLM('language English<asr_text>hello there friend'), punct)
        out = await collect(service, requests(ONE, language='Vietnamese'))
        assert out and out[-1].event == pb.COMMIT
        for r in out:
            assert r.language == 'English'                      # not the configured 'Vietnamese'
            assert r.raw_text == 'hello there friend'           # no marker
            assert r.text == 'hello there friend.'
        assert punct.seen and all(seen == ('hello there friend', 'English') for seen in punct.seen)
    asyncio.run(run())


def test_language_can_change_between_utterances():
    async def run():
        llm = FakeVLLM(lambda peak: ('language Vietnamese<asr_text>xin chao anh' if peak < 0.3
                                     else 'language English<asr_text>hello there friend'))
        frames = [pcm(1.2, True, 0.20), pcm(0.8, False), pcm(1.2, True, 0.45), pcm(0.8, False)]
        out = await collect(make_service(llm), requests(frames))
        commits = [r for r in out if r.event == pb.COMMIT]
        assert [(r.language, r.raw_text) for r in commits] == [
            ('Vietnamese', 'xin chao anh'), ('English', 'hello there friend')]
    asyncio.run(run())


def test_nothing_recognised_gives_no_event():
    async def run():
        out = await collect(make_service(FakeVLLM('language None<asr_text>')), requests(ONE))
        assert out == []
    asyncio.run(run())


# ------------------------------------------- fixes still hold with this backend/VAD
def test_model_calls_use_the_single_inference_thread():
    async def run():
        llm = FakeVLLM()
        service = make_service(llm)
        await asyncio.gather(*[collect(service, requests(ONE)) for _ in range(6)])
        assert len(llm.threads) == 1 and next(iter(llm.threads)).startswith('asr-infer')
    asyncio.run(run())


def test_forced_split_is_contiguous_with_32ms_frames():
    async def run():
        service = make_service(FakeVLLM(), max_utterance_sec=1.0)
        out = await collect(service, requests([pcm(3.0, True), pcm(0.8, False)]))
        commits = [(r.reason, r.start_ms, r.end_ms) for r in out if r.event == pb.COMMIT]
        assert len(commits) >= 3 and commits[0][0] == 'max_duration' and commits[0][1] == 0
        for prev, cur in zip(commits, commits[1:]):
            assert cur[1] == prev[2]
    asyncio.run(run())


def test_no_partial_for_a_discarded_utterance_with_32ms_frames():
    async def run():
        llm = FakeVLLM()
        clicks = [pcm(0.3, False), pcm(0.04, True), pcm(0.5, False), pcm(0.04, True), pcm(1.0, False)]
        out = await collect(make_service(llm, chunk_sec=1.0), requests(clicks))
        assert out == [] and llm.calls >= 1        # was: PARTIAL without COMMIT
    asyncio.run(run())


def test_vllm_error_loses_one_utterance_not_the_stream(caplog):
    async def run():
        out = await collect(make_service(FakeVLLM(fail_calls={2})), requests(ONE + ONE))
        commits = [r for r in out if r.event == pb.COMMIT]
        assert len(commits) == 1 and commits[0].raw_text == 'xin chao anh'
    with caplog.at_level(logging.ERROR, logger='qwen3_asr.grpc.vad'):
        asyncio.run(run())
    assert sum(1 for r in caplog.records if 'asr error' in r.getMessage()) == 1


# -------------------------------------------- Silero through the real torch.jit path
def _stub_checkpoint(tmp_path):
    import torch

    class StubVad(torch.nn.Module):
        """Speech probability from amplitude; same call shape as silero_vad.jit."""

        @torch.jit.export
        def reset_states(self):
            pass

        def forward(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
            return torch.clamp(samples.abs().mean() * 10, 0, 1).reshape(1)

    path = tmp_path / 'vad_stub.jit'
    torch.jit.script(StubVad()).save(str(path))
    return path


def test_silero_is_loaded_and_run_off_the_event_loop(tmp_path, monkeypatch):
    checkpoint = _stub_checkpoint(tmp_path)
    loop_thread = threading.current_thread().name
    built, pushed = set(), set()
    original_init, original_push = SileroEndpointer.__init__, SileroEndpointer.push

    def spy_init(self, *args, **kwargs):
        built.add(threading.current_thread().name)
        original_init(self, *args, **kwargs)

    def spy_push(self, samples):
        pushed.add(threading.current_thread().name)
        return original_push(self, samples)

    monkeypatch.setattr(SileroEndpointer, '__init__', spy_init)
    monkeypatch.setattr(SileroEndpointer, 'push', spy_push)

    async def run():
        service = ASRService(VLLMASRBackend(llm=FakeVLLM()), LanguagePunctuator(), chunk_sec=0.4,
                             vad_backend='silero', silero_model=str(checkpoint), ingress_max_ms=0)
        out = await collect(service, requests(ONE))
        assert [r.event for r in out][-1] == pb.COMMIT and out[-1].raw_text == 'xin chao anh'
        assert any(r.event == pb.PARTIAL for r in out)
    asyncio.run(run())
    assert built and pushed
    assert all(name.startswith('vad') for name in built | pushed)
    assert loop_thread not in built | pushed


def test_each_stream_gets_its_own_silero_model(tmp_path):
    checkpoint = _stub_checkpoint(tmp_path)

    async def run():
        service = ASRService(VLLMASRBackend(llm=FakeVLLM()), LanguagePunctuator(), chunk_sec=0.4,
                             vad_backend='silero', silero_model=str(checkpoint), ingress_max_ms=0)
        a, b = service.build_vad(16000), service.build_vad(8000)
        assert a._model is not b._model and a.offload and b.offload
        results = await asyncio.gather(*[collect(service, requests(ONE)) for _ in range(5)])
        assert all(out and out[-1].event == pb.COMMIT for out in results)
    asyncio.run(run())


def test_missing_silero_checkpoint_is_reported(tmp_path):
    async def run():
        service = ASRService(VLLMASRBackend(llm=FakeVLLM()), LanguagePunctuator(), chunk_sec=0.4,
                             vad_backend='silero', silero_model=str(tmp_path / 'nope.jit'))
        try:
            await collect(service, requests(ONE))
        except FileNotFoundError as exc:
            assert 'Silero TorchScript checkpoint not found' in str(exc)
        else:
            raise AssertionError('expected FileNotFoundError')
    asyncio.run(run())
