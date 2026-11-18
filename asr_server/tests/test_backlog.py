"""PARTIAL inference is skipped while audio of the stream is waiting for the model.

Overload must thin out PARTIALs, not grow the ingress queue until the stream is closed.
A COMMIT is always inferred on all the audio of the utterance.

The model is the native vLLM backend (one inference per call, on everything received so
far) with a fake LLM. The qwen_asr package infers once per chunk inside each call, so
skipping only delays its work; see README, "Failure and overload behaviour".
"""
import asyncio
import base64
import io
import logging
import time
import wave
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from asr_server import messages as pb
from asr_server.asr_backend import VLLMASRBackend

from test_regression import Aborted, FakeContext, cfg_msg, collect, frames_req, make_service, pcm


class SlowLLM:
    """vllm.LLM.chat() stand-in: answers `level NN` from the peak of the WAV, after `delay`."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.seconds = []            # audio length of every inference, in order

    @property
    def infer_calls(self):
        return len(self.seconds)

    def chat(self, conversation, sampling_params, use_tqdm=True):
        payload = base64.b64decode(conversation[0]['content'][0]['audio_url']['url'].split(',', 1)[1])
        with wave.open(io.BytesIO(payload), 'rb') as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
        self.seconds.append(samples.size / 16000)
        if self.delay:
            time.sleep(self.delay)
        level = round(float(np.max(np.abs(samples))) / 32767.0 * 100)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=f'language Vietnamese<asr_text>level {level}')])]


def native(llm, **options):
    return make_service(VLLMASRBackend(llm=llm), **options)

VOICE_THEN_SILENCE = [pcm(2.4, True), pcm(0.8, False)]


def run_fast(service):
    """Everything is sent at once: the whole call is waiting when processing starts."""
    return asyncio.run(collect(service, frames_req(VOICE_THEN_SILENCE)))


def test_backlog_skips_partial_inferences_and_the_commit_is_unchanged(caplog):
    plain_asr, skipping_asr = SlowLLM(), SlowLLM()
    plain = run_fast(native(plain_asr, partial_max_backlog_ms=0))
    with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.vad'):
        skipping = run_fast(native(skipping_asr, partial_max_backlog_ms=500))

    assert plain_asr.infer_calls >= 5                       # one per 0.4 s chunk, as before
    assert skipping_asr.infer_calls < plain_asr.infer_calls / 2
    commits = [r for r in skipping if r.is_final]
    assert len(commits) == 1 and skipping[-1].is_final
    reference = [r for r in plain if r.is_final][0]
    assert (commits[0].text, commits[0].raw_text) == (reference.text, reference.raw_text)
    assert (commits[0].start_ms, commits[0].end_ms) == (reference.start_ms, reference.end_ms)
    assert skipping_asr.seconds[-1] == plain_asr.seconds[-1]      # the COMMIT saw all the audio
    # revision and utterance numbering stay consistent with fewer PARTIALs
    assert [r.revision for r in skipping] == list(range(1, len(skipping) + 1))

    messages = [r.getMessage() for r in caplog.records]
    assert sum('asr backlog' in m for m in messages) == 1    # one WARNING per stream
    commit_line = next(m for m in messages if m.startswith('commit '))
    assert int(commit_line.rsplit('skipped_partials=', 1)[1]) >= 1


def test_no_skipping_when_the_stream_keeps_up(caplog):
    """A paced sender with a fast model: nothing waits, every chunk gets its inference."""
    asr = SlowLLM()
    service = native(asr, partial_max_backlog_ms=400, chunk_sec=0.2)

    async def paced():
        start = time.monotonic()
        yield cfg_msg()
        data = pcm(0.8, True) + pcm(0.7, False)
        for n, pos in enumerate(range(0, len(data), 2560)):
            await asyncio.sleep(max(0.0, start + n * 0.08 - time.monotonic()))
            yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n + 1, data=data[pos:pos + 2560]))
        yield pb.StreamRequest(end_stream=pb.EndStream())

    with caplog.at_level(logging.INFO, logger='qwen3_asr.grpc.vad'):
        out = asyncio.run(collect(service, paced()))
    assert out[-1].is_final
    messages = [r.getMessage() for r in caplog.records]
    assert not any('asr backlog' in m for m in messages)
    assert next(m for m in messages if m.startswith('commit ')).endswith('skipped_partials=0')
    assert asr.infer_calls >= 4


async def _paced_call(service, voiced_sec):
    """One real-time call in 80 ms chunks; returns the results or raises Aborted."""
    async def paced():
        start = time.monotonic()
        yield cfg_msg()
        data = pcm(voiced_sec, True) + pcm(0.8, False)
        for n, pos in enumerate(range(0, len(data), 2560)):
            await asyncio.sleep(max(0.0, start + n * 0.08 - time.monotonic()))
            yield pb.StreamRequest(audio=pb.AudioChunk(sequence=n + 1, data=data[pos:pos + 2560]))
        yield pb.StreamRequest(end_stream=pb.EndStream())
    return [r async for r in service.StreamRecognize(paced(), FakeContext())]


def test_slow_model_closes_the_stream_without_skipping():
    """The behaviour being fixed: every 0.2 s of audio costs a 0.5 s model call."""
    service = native(SlowLLM(delay=0.5), chunk_sec=0.2, ingress_max_ms=1000, partial_max_backlog_ms=0)
    with pytest.raises(Aborted) as err:
        asyncio.run(_paced_call(service, 3.0))
    assert err.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_slow_model_keeps_the_stream_with_skipping():
    asr = SlowLLM(delay=0.5)
    service = native(asr, chunk_sec=0.2, ingress_max_ms=1500, partial_max_backlog_ms=200)
    out = asyncio.run(_paced_call(service, 3.0))
    commits = [r for r in out if r.is_final]
    assert len(commits) == 1 and out[-1].is_final
    assert commits[0].raw_text == 'level 20'
    assert commits[0].end_ms - commits[0].start_ms >= 3000       # the whole utterance was committed
    assert asr.infer_calls < 15                                   # 3.0 s / 0.2 s without skipping
    assert asr.seconds[-1] >= 3.0                                 # the COMMIT saw all the audio


def test_forced_split_is_still_inferred_under_backlog():
    """max_utterance cuts are COMMITs: they are never skipped, whatever is waiting."""
    service = native(SlowLLM(), partial_max_backlog_ms=500, max_utterance_sec=1.0)
    out = asyncio.run(collect(service, frames_req([pcm(3.2, True), pcm(0.8, False)])))
    commits = [r for r in out if r.is_final]
    assert [c.reason for c in commits][:3] == ['max_duration'] * 3
    assert all(c.raw_text == 'level 20' for c in commits)
    for before, after in zip(commits, commits[1:]):
        assert after.start_ms == before.end_ms
