import logging
from pathlib import Path

import pytest

from asr_server.config import ConfigError, Settings, load_settings, setup_logging

ROOT = Path(__file__).resolve().parents[1]

# Defaults of the command-line flags of the native-vLLM + Silero server.py
# (docs/qwen3_asr_grpc_native_vllm_silero.zip), which config/server.yaml replaces.
FORMER_CLI_DEFAULTS = dict(
    model='/opt/models/Qwen3-ASR-1.7B', punct_model='/opt/models/punctuation-restoration-small',
    punct_device='cpu', punct_threshold=0.55, host='0.0.0.0', port=50051,
    chunk_sec=1.0, max_utterance_sec=15.0, vad_endpoint_ms=600, vad_pause_ms=300,
    silero_model='/opt/models/silero_vad.jit', vad_threshold=0.5, vad_neg_threshold=None,
    min_voiced_ms=160, semantic_early_commit=False, semantic_confidence=0.90,
    stable_updates=2, max_new_tokens=256, gpu_memory_utilization=0.75, max_model_len=8192,
    max_num_seqs=1, dtype='float16', enforce_eager=False, max_streams=64)


def write(tmp_path, text):
    path = tmp_path / 'server.yaml'
    path.write_text(text, encoding='utf-8')
    return path


# Keys added with the bug fixes (docs/Giai_phap_fix_bug_server_hien_tai.md, section 5).
NEW_DEFAULTS = dict(ingress_max_ms=3000, audio_timeout_ms=2000, asr_max_consecutive_errors=3, partial_max_backlog_ms=500,
                    punct_workers=1, punct_partial_timeout_ms=300, punct_commit_timeout_ms=1000,
                    # backend selection; vad_mode only applies to vad.backend = webrtc
                    asr_backend='native_vllm', vad_backend='silero', vad_workers=2, vad_mode=2,
                    use_language_source=False,
                    # language of a stream that names none (was fixed to Vietnamese); not forced
                    default_language='vi', force_language=False,
                    # the text of the calls stays out of the log unless asked for
                    asr_log_results=False,
                    # word-by-word reveal of PARTIAL results, off by default
                    word_reveal_enabled=False, word_reveal_window_ms=400, word_reveal_min_interval_ms=120,
                    # task noise_reduction (DPDFNet-2), off by default; speech_to_text stays on
                    asr_enabled=True, noise_enabled=False, noise_model='/opt/models/dpdfnet2.onnx',
                    noise_frame_ms=20, noise_num_threads=1, noise_workers=2, noise_max_streams=16)
ALL_DEFAULTS = {**FORMER_CLI_DEFAULTS, **NEW_DEFAULTS}


def test_defaults_match_former_cli_flags():
    assert vars(Settings()) == ALL_DEFAULTS


def test_shipped_server_yaml_equals_defaults():
    assert vars(load_settings(ROOT / 'config' / 'server.yaml')) == ALL_DEFAULTS


def test_empty_file_gives_defaults(tmp_path):
    assert load_settings(write(tmp_path, '')) == Settings()


def test_env_example_sets_the_offline_runtime_variables():
    lines = (ROOT / 'bin' / 'env.sh.example').read_text(encoding='utf-8').splitlines()
    exported = dict(line[len('export '):].split('=', 1) for line in lines if line.startswith('export '))
    assert exported == {
        'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'CUDA_VISIBLE_DEVICES': '0'}


def test_word_reveal_is_off_unless_enabled(tmp_path):
    assert Settings().word_reveal is None
    path = write(tmp_path, 'asr:\n  chunk_sec: 0.5\nword_reveal:\n  enabled: true\n')
    assert load_settings(path).word_reveal == (400, 120)
    path = write(tmp_path, 'word_reveal:\n  enabled: true\n  window_ms: 300\n  min_interval_ms: 100\n')
    assert load_settings(path).word_reveal == (300, 100)


def test_log_results_is_read_from_the_asr_section(tmp_path):
    assert load_settings(write(tmp_path, 'asr:\n  log_results: true\n')).asr_log_results is True


@pytest.mark.parametrize('code', ['vi', 'en', 'auto'])
def test_language_settings_are_read_from_the_asr_section(tmp_path, code):
    s = load_settings(write(tmp_path, f'asr:\n  default_language: {code}\n  force_language: true\n'))
    assert (s.default_language, s.force_language) == (code, True)


def test_noise_reduction_is_read_from_its_section(tmp_path):
    s = load_settings(write(tmp_path, (
        'asr:\n  enabled: false\n'
        'noise_reduction:\n  enabled: true\n  model: /m/dpdfnet2.onnx\n  frame_ms: 40\n'
        '  num_threads: 2\n  workers: 4\n  max_streams: 8\n')))
    assert (s.asr_enabled, s.noise_enabled, s.noise_model, s.noise_frame_ms, s.noise_num_threads,
            s.noise_workers, s.noise_max_streams) == (False, True, '/m/dpdfnet2.onnx', 40, 2, 4, 8)


def test_noise_reduction_limits_apply_only_when_enabled(tmp_path):
    # default noise_reduction.max_streams (16) above server.max_streams is fine while it is off
    assert load_settings(write(tmp_path, 'server:\n  max_streams: 8\n')).max_streams == 8
    s = load_settings(write(tmp_path, 'server:\n  max_streams: 8\nnoise_reduction:\n  enabled: true\n'
                                      '  max_streams: 8\n'))
    assert (s.noise_enabled, s.noise_max_streams) == (True, 8)


def test_values_and_overrides(tmp_path):
    path = write(tmp_path, 'server:\n  port: 6000\nasr:\n  chunk_sec: 2\nsemantic:\n  early_commit: true\n')
    s = load_settings(path)
    assert (s.port, s.chunk_sec, s.semantic_early_commit) == (6000, 2.0, True)
    assert isinstance(s.chunk_sec, float)
    s = load_settings(path, {'port': 7000, 'host': None})
    assert (s.port, s.host) == (7000, '0.0.0.0')


@pytest.mark.parametrize('text', [
    'server:\n  prot: 50051\n',                # typo in key
    'serverr:\n  port: 50051\n',               # typo in section
    'server:\n  port: "50051"\n',              # wrong type
    'semantic:\n  early_commit: 1\n',          # not a boolean
    'server:\n  port: 70000\n',                # out of range
    'vad:\n  mode: 4\n',
    'vad:\n  pause_ms: 600\n  endpoint_ms: 600\n',
    'asr:\n  chunk_sec: 0.1\n',
    'asr:\n  model: ""\n',
    'server:\n  ingress_max_ms: -1\n',
    'asr:\n  max_consecutive_errors: 0\n',
    'punctuation:\n  partial_timeout_ms: 0\n',
    'asr:\n  backend: whisper\n',
    'vad:\n  backend: energy\n',
    'asr:\n  dtype: float64\n',
    'asr:\n  log_results: 1\n',
    'asr:\n  default_language: fr\n',         # vi | en | auto only
    'asr:\n  default_language: Vietnamese\n', # the speech.proto code, not the model name
    'asr:\n  default_language: VI\n',
    'asr:\n  default_language: ""\n',
    'asr:\n  force_language: 1\n',
    'vad:\n  threshold: 1.5\n',
    'vad:\n  threshold: 0.5\n  neg_threshold: 0.7\n',
    'word_reveal:\n  enabled: 1\n',
    'word_reveal:\n  window_ms: 0\n',
    'word_reveal:\n  min_interval_ms: 0\n',
    'word_reveal:\n  window_ms: 100\n  min_interval_ms: 200\n',
    'word_reveal:\n  windows_ms: 400\n',
    # the words of one PARTIAL must be out before the next PARTIAL arrives
    'asr:\n  chunk_sec: 0.5\nword_reveal:\n  enabled: true\n  window_ms: 600\n',
    'asr:\n  enabled: 1\n',
    'asr:\n  enabled: false\n',                # no task left to serve
    'noise_reduction:\n  enabled: 1\n',
    'noise_reduction:\n  model: ""\n',
    'noise_reduction:\n  frame_ms: 10\n',      # the gateway wants multiples of 640 bytes (20 ms)
    'noise_reduction:\n  frame_ms: 30\n',
    'noise_reduction:\n  num_threads: 0\n',
    'noise_reduction:\n  workers: 0\n',
    'noise_reduction:\n  max_streams: 0\n',
    'noise_reduction:\n  max_stream: 4\n',
    # gRPC would refuse the extra RPCs before the denoiser pool does
    'server:\n  max_streams: 8\nnoise_reduction:\n  enabled: true\n',
    '- not\n- a mapping\n',
])
def test_invalid_config_is_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_settings(write(tmp_path, text))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / 'nope.yaml')


def test_logging_yaml_writes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        setup_logging(ROOT / 'config' / 'logging.yaml')
        logging.getLogger('qwen3_asr.grpc.vad').info('hello log file')
        for handler in logging.getLogger().handlers:
            handler.flush()
        text = (tmp_path / 'logs' / 'asr-server.log').read_text(encoding='utf-8')
        assert 'INFO qwen3_asr.grpc.vad: hello log file' in text
    finally:
        for handler in root.handlers[:]:
            if handler not in saved[0]:
                handler.close()
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


def test_check_config_does_not_create_log_files(tmp_path, monkeypatch, capsys):
    from asr_server.__main__ import main
    monkeypatch.chdir(tmp_path)
    code = main(['--config', str(ROOT / 'config' / 'server.yaml'),
                 '--logging-config', str(ROOT / 'config' / 'logging.yaml'), '--check-config'])
    assert code == 0
    assert 'port = 50051' in capsys.readouterr().out
    assert not (tmp_path / 'logs').exists()


def test_main_reports_config_error(tmp_path, capsys):
    from asr_server.__main__ import main
    assert main(['--config', str(write(tmp_path, 'server:\n  prot: 1\n')), '--check-config']) == 2
    assert 'unknown key server.prot' in capsys.readouterr().err


def test_no_console_drops_only_console_handler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        setup_logging(ROOT / 'config' / 'logging.yaml', console=False)
        kinds = [type(h).__name__ for h in logging.getLogger().handlers]
        assert kinds == ['TimedRotatingFileHandler']
    finally:
        for handler in root.handlers[:]:
            if handler not in saved[0]:
                handler.close()
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


def test_optional_setting_does_not_depend_on_the_typing_cache(tmp_path):
    # Imports can push Optional[float] out of typing's cache; an identity check then failed
    # with "vad_neg_threshold: expected a non-empty string, got 0.3".
    import typing
    for clear in typing._cleanups:
        clear()
    assert load_settings(write(tmp_path, 'vad:\n  neg_threshold: 0.3\n')).vad_neg_threshold == 0.3


def test_backend_selection_and_optional_neg_threshold(tmp_path):
    s = load_settings(write(tmp_path, 'asr:\n  backend: qwen_asr\nvad:\n  backend: webrtc\n  neg_threshold: 0.3\n'))
    assert (s.asr_backend, s.vad_backend, s.vad_neg_threshold) == ('qwen_asr', 'webrtc', 0.3)
    s = load_settings(write(tmp_path, 'vad:\n  neg_threshold: null\n'))
    assert s.vad_neg_threshold is None
