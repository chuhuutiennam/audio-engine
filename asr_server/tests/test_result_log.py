"""asr.log_results: one log line per PARTIAL and COMMIT, off by default."""
import asyncio
import logging

import pytest

from asr_server import asr_backend, messages as pb, server
from asr_server.config import Settings
from asr_server.text_clean import count_han

from test_native_backend import FakeVLLM, ONE, collect, make_service, requests
from test_regression import FakeASR, FakeContext, frames_req, pcm
from test_regression import make_service as make_fake_asr_service

LOGGER = 'qwen3_asr.grpc.vad'


def result_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith('asr_result ')]


def run(llm, caplog, **options):
    async def go():
        service = make_service(llm, **options)
        try:
            return await collect(service, requests(ONE))
        finally:
            service.close()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        return asyncio.run(go())


def expected_line(r, model_output, han_chars):
    event = 'COMMIT' if r.event == pb.COMMIT else 'PARTIAL'
    return (f'asr_result call=c1 leg=a utterance={r.utterance_index} revision={r.revision} '
            f'event={event} language={r.language} han_chars={han_chars} '
            f'model_output={model_output!r} raw={r.raw_text!r} text={r.text!r}')


def test_no_result_line_and_no_call_text_by_default(caplog):
    out = run(FakeVLLM('language Vietnamese<asr_text>xin chao anh'), caplog)
    assert out and result_lines(caplog) == []
    assert not any('xin chao' in r.getMessage() for r in caplog.records)


def test_every_partial_and_commit_is_logged_with_what_the_client_receives(caplog):
    answer = 'language Vietnamese<asr_text>嗯 nhà mình'
    out = run(FakeVLLM(answer), caplog, log_results=True, partial_max_backlog_ms=0)
    assert [r.event for r in out] == [pb.PARTIAL, pb.COMMIT]
    assert result_lines(caplog) == [expected_line(r, answer, 1) for r in out]


def test_line_shows_a_language_the_result_does_not_report(caplog):
    answer = 'language Vietnamese<asr_text>xin chào language Chinese<asr_text>你好'
    out = run(FakeVLLM(answer), caplog, log_results=True)
    assert out[-1].language == 'Vietnamese'
    line = result_lines(caplog)[-1]
    assert 'language=Vietnamese han_chars=2 ' in line and 'language Chinese<asr_text>你好' in line


def test_text_that_became_empty_is_logged(caplog):
    answers = iter(['language Vietnamese<asr_text>xin chao anh'])
    llm = FakeVLLM(lambda peak: next(answers, 'language None<asr_text>'))
    out = run(llm, caplog, log_results=True, partial_max_backlog_ms=0)
    assert out[-1].event == pb.PARTIAL and out[-1].text == ''
    lines = result_lines(caplog)
    assert len(lines) == len(out)
    assert lines[-1].endswith("han_chars=0 model_output='language None<asr_text>' raw='' text=''")


def test_a_failed_model_call_logs_no_model_answer(caplog):
    llm = FakeVLLM('language Vietnamese<asr_text>xin chao anh', fail_calls={2})
    out = run(llm, caplog, log_results=True, partial_max_backlog_ms=0)
    assert out[-1].event == pb.PARTIAL and out[-1].text == ''
    assert result_lines(caplog)[-1].endswith("han_chars=0 model_output='' raw='' text=''")


def test_a_model_answer_with_a_line_break_stays_on_one_log_line(caplog):
    run(FakeVLLM('language Vietnamese<asr_text>xin\nasr_result forged'), caplog, log_results=True)
    lines = result_lines(caplog)
    assert lines and all('\n' not in line for line in lines)
    assert all(line.startswith('asr_result call=c1 ') for line in lines)


def test_backend_without_raw_output_logs_it_empty(caplog):
    async def go():
        service = make_fake_asr_service(FakeASR(), log_results=True)
        frames = [pcm(1.2, True), pcm(0.8, False)]
        try:
            return [r async for r in service.StreamRecognize(frames_req(frames), FakeContext())]
        finally:
            service.close()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = asyncio.run(go())
    lines = result_lines(caplog)
    assert out and len(lines) == len(out)
    assert all(" model_output='' " in line for line in lines)


@pytest.mark.parametrize('text, count', [
    ('', 0),
    ('cho mình gặp bộ phận kỹ thuật với ạ.', 0),
    ('internet 123 .,?!', 0),
    ('。,、', 0),                     # Chinese punctuation is not a Han character
    ('こんにちは 안녕', 0),
    ('你好', 2),
    ('嗯 nhà mình', 1),
    ('㐀 龥 豈 𠀀', 4),                # extension A, end of the main block, compatibility, extension B
    ('〇々', 2),              # ideographic zero, iteration mark
    ('⺀⼀', 2),              # radicals
    ('〡〸', 2),              # Hangzhou numerals
])
def test_count_han(text, count):
    assert count_han(text) == count


def test_serve_passes_the_flag_and_warns_that_call_text_is_logged(tmp_path, monkeypatch, caplog):
    class Stop(Exception):
        pass

    received = {}

    def fake_service(*args, **kwargs):
        received.update(kwargs)
        raise Stop

    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                 'VLLM_WORKER_MULTIPROC_METHOD'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(server, 'ASRService', fake_service)
    monkeypatch.setattr(server, 'HFPunctuator', lambda *a, **k: object())
    monkeypatch.setattr(asr_backend, 'VLLMASRBackend', lambda **k: object())
    settings = Settings(model=str(tmp_path), punct_model=str(tmp_path), vad_backend='webrtc',
                        asr_log_results=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(Stop):
        asyncio.run(server.serve(settings))
    assert received['log_results'] is True
    assert any('asr.log_results is on' in r.getMessage() for r in caplog.records)
