"""Native vLLM inference adapter tests; no GPU/vllm installation required."""
import asyncio
import base64
import io
import wave
from types import SimpleNamespace

import numpy as np

from asr_server.asr_backend import VLLMASRBackend
from asr_server import messages as pb
from asr_server.punctuation import PunctResult
from asr_server.server import ASRService
from asr_server.vad import SileroEndpointer


class FakeVLLM:
    def __init__(self):
        self.calls = []

    def chat(self, conversation, sampling_params, use_tqdm=True):
        assert use_tqdm is False
        assert sampling_params is not None
        message = conversation[0]['content'][0]
        assert message['type'] == 'audio_url'
        url = message['audio_url']['url']
        assert url.startswith('data:audio/wav;base64,')
        with wave.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1])), 'rb') as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            self.calls.append(w.getnframes())
        return [SimpleNamespace(outputs=[SimpleNamespace(text='xin chao anh')])]


class FakePunctuator:
    def restore(self, text):
        return PunctResult(text + '?', '?', 0.999)


def fake_vad(sr):
    return SileroEndpointer(sr, classifier=lambda x, r: float(np.max(np.abs(x))) > .05)


def test_partial_and_final_use_accumulated_wav_and_no_repeat_final():
    llm = FakeVLLM()
    model = VLLMASRBackend(llm=llm)
    state = model.init_streaming_state(language='Vietnamese', chunk_size_sec=1.0)
    samples = np.full(4000, 0.25, dtype=np.float32)
    for _ in range(3):
        model.streaming_transcribe(samples, state)
    assert not llm.calls and state.chunk_id == 0
    model.streaming_transcribe(samples, state)
    assert llm.calls == [16000]
    assert state.chunk_id == 1 and state.text == 'xin chao anh'
    model.streaming_transcribe(samples, state)
    assert llm.calls == [16000]
    model.finish_streaming_transcribe(state)
    assert llm.calls == [16000, 20000]
    assert state.chunk_id == 2
    model.finish_streaming_transcribe(state)
    assert llm.calls == [16000, 20000]  # no duplicated final call


def test_short_utterance_only_transcribed_on_finish():
    llm = FakeVLLM()
    model = VLLMASRBackend(llm=llm)
    state = model.init_streaming_state(chunk_size_sec=1.0)
    model.streaming_transcribe(np.ones(4800, dtype=np.float32) / 4, state)
    assert not llm.calls
    model.finish_streaming_transcribe(state)
    assert llm.calls == [4800] and state.text == 'xin chao anh'


def test_grpc_silero_partial_and_commit_using_native_vllm():
    async def run():
        llm = FakeVLLM()
        asr = VLLMASRBackend(llm=llm)
        svc = ASRService(asr, FakePunctuator(), vad_factory=fake_vad, chunk_sec=0.4)
        async def reqs():
            yield pb.StreamRequest(config=pb.StreamConfig(
                call_id='call1', leg_id='callee', sample_rate=16000, encoding=pb.PCM_S16LE))
            audio = (np.full(19200, 8000, dtype='<i2')).tobytes() + bytes(2 * 12800)
            for n, i in enumerate(range(0, len(audio), 3200), start=1):
                yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n, data=audio[i:i+3200]))
            yield pb.StreamRequest(end_stream=pb.EndStream())
        class Context:
            def cancelled(self):
                return False
        events = [e async for e in svc.StreamRecognize(reqs(), Context())]
        partials = [e for e in events if e.event == pb.PARTIAL]
        commits = [e for e in events if e.event == pb.COMMIT]
        assert partials and len(commits) == 1
        assert commits[0].text == 'xin chao anh?'
        assert commits[0].reason == 'vad_endpoint'
        assert llm.calls and max(llm.calls) > 16000
    asyncio.run(run())
