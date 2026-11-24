"""asr.force_language and asr.default_language.

With force_language on, the native vLLM backend appends `language <name><asr_text>` to the
generation prompt of the model's chat template, as qwen_asr builds a forced-language prompt,
so the model writes text only. default_language replaces the fixed 'Vietnamese' of a stream
that names no language; `auto` lets the model detect it.
"""
import asyncio
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from asr_server import asr_backend, messages as pb, server
from asr_server.asr_backend import VLLMASRBackend, forced_language_template, load_chat_template
from asr_server.config import Settings
from asr_server.server import ASRService

from test_native_backend import LanguagePunctuator, ONE, collect, requests, silero_classifier_vad
from test_speech_grpc import chunks, recognise

LOGGER = 'qwen3_asr.grpc.vad'

# Same structure as the template of Qwen3-ASR: one system turn, one user turn holding only the
# audio tokens, and the generation prompt last.
BASE = (
    "{%- set ns = namespace(system_text='') -%}"
    "{%- for m in messages -%}{%- if m.role == 'system' -%}"
    "{%- set ns.system_text = ns.system_text + m.content -%}{%- endif -%}{%- endfor -%}"
    "{{- '<|im_start|>system\\n' + ns.system_text + '<|im_end|>\\n' -}}"
    "{{- '<|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\\n' -}}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}\n"
)


class RecordingLLM:
    """vllm.LLM.chat() stand-in that records the chat template of every call.

    `answer` is a string or a function of that template (None = the model's own template).
    Like vLLM, the result carries the prompt the template gave."""

    def __init__(self, answer):
        self.answer = answer
        self.templates = []

    def chat(self, conversation, sampling_params, use_tqdm=True, chat_template=None):
        assert conversation[0]['content'][0]['type'] == 'audio_url'
        self.templates.append(chat_template)
        text = self.answer(chat_template) if callable(self.answer) else self.answer
        return [SimpleNamespace(prompt=render(chat_template or BASE),
                                outputs=[SimpleNamespace(text=text)])]


def like_the_model(template):
    """A forced prompt gets text only; on its own the model names a language, wrongly here."""
    return 'language Portuguese<asr_text>xin chao' if template is None else 'xin chao'


def render(template, add_generation_prompt=True):
    """The prompt the template gives, rendered as transformers renders chat templates."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    messages = [{'role': 'system', 'content': ''},
                {'role': 'user', 'content': [{'type': 'audio_url', 'audio_url': {'url': 'x'}}]}]
    return env.from_string(template).render(messages=messages,
                                            add_generation_prompt=add_generation_prompt)


def forced(language):
    return forced_language_template(BASE, language)


# ------------------------------------------------------------------------ template
@pytest.mark.parametrize('base', [BASE, BASE.rstrip('\n'), BASE + '  \n\n'])
@pytest.mark.parametrize('language', ['Vietnamese', 'English'])
def test_forced_template_is_the_qwen_prompt_followed_by_the_language(base, language):
    assert render(base).endswith('<|im_start|>assistant\n')
    assert render(forced_language_template(base, language)) == render(base) + f'language {language}<asr_text>'
    assert render(forced_language_template(base, language), False) == render(base, False)


@pytest.mark.parametrize('name', ['', 'Viet{{ 7*7 }}', 'Vietnamese<asr_text>', "Viet{% raw %}"])
def test_a_language_name_that_could_change_the_template_is_refused(name):
    with pytest.raises(ValueError):
        forced_language_template(BASE, name)


def test_chat_template_is_read_like_the_processor_reads_it(tmp_path):
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'TOKENIZER'}))
    assert load_chat_template(tmp_path) == 'TOKENIZER'
    (tmp_path / 'chat_template.json').write_text(json.dumps({'chat_template': 'PROCESSOR'}))
    assert load_chat_template(tmp_path) == 'PROCESSOR'
    (tmp_path / 'chat_template.jinja').write_text('JINJA', encoding='utf-8')
    assert load_chat_template(tmp_path) == 'JINJA'


def test_a_list_of_named_templates_gives_the_default_one(tmp_path):
    named = [{'name': 'tool_use', 'template': 'TOOLS'}, {'name': 'default', 'template': 'DEFAULT'}]
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'chat_template': named}))
    assert load_chat_template(tmp_path) == 'DEFAULT'


def test_missing_chat_template_stops_the_start_before_vllm_is_loaded(tmp_path):
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'model_max_length': 8192}))
    with pytest.raises(FileNotFoundError, match='chat template'):
        VLLMASRBackend(model_path=tmp_path, force_language=True)


# ------------------------------------------------------------------------- backend
def utterance(model, language):
    state = model.init_streaming_state(language=language, chunk_size_sec=1.0)
    model.streaming_transcribe(np.full(16000, 0.25, np.float32), state)
    return state


def test_forced_language_goes_into_the_prompt_and_the_answer_is_text():
    llm = RecordingLLM('xin chào anh')
    state = utterance(VLLMASRBackend(llm=llm, force_language=True, chat_template=BASE), 'Vietnamese')
    assert llm.templates == [forced('Vietnamese')]
    assert (state.text, state.language, state.raw_output) == ('xin chào anh', 'Vietnamese', 'xin chào anh')


def test_a_marker_in_a_forced_answer_does_not_change_the_language():
    llm = RecordingLLM('xin chào language Chinese<asr_text>你好')
    state = utterance(VLLMASRBackend(llm=llm, force_language=True, chat_template=BASE), 'Vietnamese')
    assert (state.text, state.language) == ('xin chào 你好', 'Vietnamese')


def test_no_language_is_detected_even_when_forcing_is_on():
    llm = RecordingLLM('language English<asr_text>hello')
    state = utterance(VLLMASRBackend(llm=llm, force_language=True, chat_template=BASE), None)
    assert llm.templates == [None]
    assert (state.text, state.language) == ('hello', 'English')


def test_the_end_of_the_prompt_vllm_built_is_logged_once_per_language(caplog):
    # Evidence from the real model that the template was applied: auto and forced prompts
    # must be the same up to the forced suffix.
    model = VLLMASRBackend(llm=RecordingLLM('xin chao'), force_language=True, chat_template=BASE)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        for language in ('Vietnamese', 'Vietnamese', None, 'English', None):
            utterance(model, language)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('model prompt ')]
    tail = {name: (render(BASE) + suffix)[-90:] for name, suffix in (
        ('Vietnamese', 'language Vietnamese<asr_text>'), ('auto', ''),
        ('English', 'language English<asr_text>'))}
    assert lines == [f'model prompt language={name} ends with {tail[name]!r}'
                     for name in ('Vietnamese', 'auto', 'English')]


def test_forcing_off_keeps_detection_and_the_model_template():
    llm = RecordingLLM('language English<asr_text>hello')
    state = utterance(VLLMASRBackend(llm=llm, chat_template=BASE), 'Vietnamese')
    assert llm.templates == [None]
    assert state.language == 'English'


# ---------------------------------------------------------------- default language
def core(llm, force=False, **options):
    options.setdefault('ingress_max_ms', 0)
    options.setdefault('vad_factory', silero_classifier_vad)
    return ASRService(VLLMASRBackend(llm=llm, force_language=force, chat_template=BASE),
                      LanguagePunctuator(), chunk_sec=0.4, **options)


def languages(service, language=''):
    async def go():
        try:
            return {r.language for r in await collect(service, requests(ONE, language=language))}
        finally:
            service.close()
    return asyncio.run(go())


def test_default_language_replaces_the_fixed_vietnamese():
    # the model names no language, so the result carries the language of the stream
    assert languages(core(RecordingLLM('hello'))) == {'Vietnamese'}
    assert languages(core(RecordingLLM('hello'), default_language='English')) == {'English'}
    assert languages(core(RecordingLLM('hello'), default_language=pb.AUTO_LANGUAGE)) == {''}
    assert languages(core(RecordingLLM('hello'), default_language='English'), 'Vietnamese') == {'Vietnamese'}


def test_default_language_is_forced_for_a_stream_that_names_none():
    llm = RecordingLLM(like_the_model)
    assert languages(core(llm, force=True, default_language='English')) == {'English'}
    assert llm.templates and set(llm.templates) == {forced('English')}


# ------------------------------------------------------------------- speech.proto
@pytest.mark.parametrize('language, source, use_source, default, model_language, result', [
    ('vi', '', False, 'Vietnamese', 'Vietnamese', 'vi'),
    ('en', '', False, 'Vietnamese', 'English', 'en'),
    ('VI', '', False, pb.AUTO_LANGUAGE, 'Vietnamese', 'vi'),
    ('', '', False, 'Vietnamese', 'Vietnamese', 'vi'),
    ('', '', False, 'English', 'English', 'en'),
    ('', '', False, pb.AUTO_LANGUAGE, None, ''),          # detected: Portuguese has no code
    ('auto', '', False, 'Vietnamese', None, ''),
    ('fr', '', False, 'English', 'English', 'en'),        # unknown code: the default
    ('vi', 'en', True, 'Vietnamese', 'English', 'en'),    # language_source wins when enabled
    ('vi', 'en', False, 'Vietnamese', 'Vietnamese', 'vi'),
])
def test_language_of_the_stream_is_forced(caplog, language, source, use_source, default,
                                          model_language, result):
    llm = RecordingLLM(like_the_model)
    messages = chunks(ONE, language=language, language_source=source)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = recognise(messages, core(llm, force=True, default_language=default),
                        use_language_source=use_source)
    assert out and out[-1].is_final and out[-1].raw_text == 'xin chao'
    expected = None if model_language is None else forced(model_language)
    assert llm.templates and set(llm.templates) == {expected}
    assert {r.language for r in out} == {result}
    reported = any("model reported language 'Portuguese'" in r.getMessage() for r in caplog.records)
    assert reported is (model_language is None)
    unknown = any("unknown language code 'fr'" in r.getMessage() for r in caplog.records)
    assert unknown is (language == 'fr')


def test_forcing_off_leaves_the_language_to_the_model():
    llm = RecordingLLM(like_the_model)
    out = recognise(chunks(ONE, language='vi'), core(llm))
    assert set(llm.templates) == {None}
    assert {r.language for r in out} == {''}


# -------------------------------------------------------------------------- serve
@pytest.mark.parametrize('code, default', [('vi', 'Vietnamese'), ('en', 'English'),
                                           ('auto', pb.AUTO_LANGUAGE)])
def test_serve_passes_the_language_settings(tmp_path, monkeypatch, code, default):
    class Stop(Exception):
        pass

    received, backend = {}, {}

    def fake_service(*args, **kwargs):
        received.update(kwargs)
        raise Stop

    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                 'VLLM_WORKER_MULTIPROC_METHOD'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(server, 'ASRService', fake_service)
    monkeypatch.setattr(server, 'HFPunctuator', lambda *a, **k: object())
    monkeypatch.setattr(asr_backend, 'VLLMASRBackend', lambda **k: backend.update(k) or object())
    settings = Settings(model=str(tmp_path), punct_model=str(tmp_path), vad_backend='webrtc',
                        default_language=code, force_language=True)
    with pytest.raises(Stop):
        asyncio.run(server.serve(settings))
    assert backend['force_language'] is True
    assert received['default_language'] == default
