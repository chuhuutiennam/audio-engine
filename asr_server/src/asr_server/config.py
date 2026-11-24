"""Load config/server.yaml and config/logging.yaml.

server.yaml replaces the former command-line flags of server.py. Defaults follow
the native-vLLM + Silero server; unknown keys are rejected so a typo cannot
silently fall back to a default.
"""
from dataclasses import dataclass, fields
from typing import Optional
import logging
import logging.config
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    # tasks served (AudioChunk.task); a task that is off ends with UNIMPLEMENTED
    asr_enabled: bool = True                  # speech_to_text: loads Qwen3-ASR, VAD and punctuation
    noise_enabled: bool = False               # noise_reduction: DPDFNet-2 (sherpa-onnx, CPU)
    # server
    host: str = '0.0.0.0'
    port: int = 50051
    max_streams: int = 64
    ingress_max_ms: int = 3000
    audio_timeout_ms: int = 2000
    # asr
    asr_backend: str = 'native_vllm'          # native_vllm | qwen_asr
    model: str = '/opt/models/Qwen3-ASR-1.7B'
    chunk_sec: float = 1.0
    max_utterance_sec: float = 15.0
    max_new_tokens: int = 256
    gpu_memory_utilization: float = 0.75
    max_model_len: int = 8192                 # native_vllm only
    max_num_seqs: int = 1                     # native_vllm only
    dtype: str = 'float16'                    # native_vllm only: auto | float16 | bfloat16
    enforce_eager: bool = False               # native_vllm only: no CUDA graphs, less VRAM
    asr_max_consecutive_errors: int = 3
    partial_max_backlog_ms: int = 500         # 0 = never skip a PARTIAL inference
    use_language_source: bool = False         # read AudioChunk.language_source
    asr_log_results: bool = False             # log language and text of every PARTIAL and COMMIT
    default_language: str = 'vi'              # vi | en | auto: stream that names no language
    force_language: bool = False              # native_vllm: transcribe the stream's language
    # vad
    vad_backend: str = 'silero'               # silero | webrtc
    silero_model: str = '/opt/models/silero_vad.jit'
    vad_threshold: float = 0.5                # silero: speech onset probability
    vad_neg_threshold: Optional[float] = None  # silero: exit probability; None = threshold - 0.15
    vad_workers: int = 2                      # threads that load and run Silero
    vad_mode: int = 2                         # webrtc only: aggressiveness 0..3
    vad_endpoint_ms: int = 600
    vad_pause_ms: int = 300
    min_voiced_ms: int = 160
    # punctuation
    punct_model: str = '/opt/models/punctuation-restoration-small'
    punct_device: str = 'cpu'
    punct_threshold: float = 0.55
    punct_workers: int = 1
    punct_partial_timeout_ms: int = 300
    punct_commit_timeout_ms: int = 1000
    # semantic early commit
    semantic_early_commit: bool = False
    semantic_confidence: float = 0.90
    stable_updates: int = 2
    # word-by-word reveal of PARTIAL results
    word_reveal_enabled: bool = False
    word_reveal_window_ms: int = 400
    word_reveal_min_interval_ms: int = 120
    # noise_reduction (docs/DPDFNet2_Task_Routing_Design_VI.md, section 7.2)
    noise_model: str = '/opt/models/dpdfnet2.onnx'
    noise_frame_ms: int = 20                  # audio per result: 20 | 40 | 80
    noise_num_threads: int = 1                # ONNX threads of each denoiser
    noise_workers: int = 2                    # threads that run the denoisers of all streams
    noise_max_streams: int = 16               # denoisers loaded at start; one per stream
    noise_max_backlog_ms: int = 500           # audio waiting for the model; above it the stream ends

    @property
    def word_reveal(self):
        """(window_ms, min_interval_ms) when the word-by-word reveal is on, else None."""
        if not self.word_reveal_enabled:
            return None
        return self.word_reveal_window_ms, self.word_reveal_min_interval_ms


# (yaml section, yaml key) -> Settings field
_KEYS = {
    ('server', 'host'): 'host',
    ('server', 'port'): 'port',
    ('server', 'max_streams'): 'max_streams',
    ('server', 'ingress_max_ms'): 'ingress_max_ms',
    ('server', 'audio_timeout_ms'): 'audio_timeout_ms',
    ('asr', 'enabled'): 'asr_enabled',
    ('asr', 'backend'): 'asr_backend',
    ('asr', 'model'): 'model',
    ('asr', 'chunk_sec'): 'chunk_sec',
    ('asr', 'max_utterance_sec'): 'max_utterance_sec',
    ('asr', 'max_new_tokens'): 'max_new_tokens',
    ('asr', 'gpu_memory_utilization'): 'gpu_memory_utilization',
    ('asr', 'max_model_len'): 'max_model_len',
    ('asr', 'max_num_seqs'): 'max_num_seqs',
    ('asr', 'dtype'): 'dtype',
    ('asr', 'enforce_eager'): 'enforce_eager',
    ('asr', 'max_consecutive_errors'): 'asr_max_consecutive_errors',
    ('asr', 'partial_max_backlog_ms'): 'partial_max_backlog_ms',
    ('asr', 'use_language_source'): 'use_language_source',
    ('asr', 'log_results'): 'asr_log_results',
    ('asr', 'default_language'): 'default_language',
    ('asr', 'force_language'): 'force_language',
    ('vad', 'backend'): 'vad_backend',
    ('vad', 'silero_model'): 'silero_model',
    ('vad', 'threshold'): 'vad_threshold',
    ('vad', 'neg_threshold'): 'vad_neg_threshold',
    ('vad', 'workers'): 'vad_workers',
    ('vad', 'mode'): 'vad_mode',
    ('vad', 'endpoint_ms'): 'vad_endpoint_ms',
    ('vad', 'pause_ms'): 'vad_pause_ms',
    ('vad', 'min_voiced_ms'): 'min_voiced_ms',
    ('punctuation', 'model'): 'punct_model',
    ('punctuation', 'device'): 'punct_device',
    ('punctuation', 'threshold'): 'punct_threshold',
    ('punctuation', 'workers'): 'punct_workers',
    ('punctuation', 'partial_timeout_ms'): 'punct_partial_timeout_ms',
    ('punctuation', 'commit_timeout_ms'): 'punct_commit_timeout_ms',
    ('semantic', 'early_commit'): 'semantic_early_commit',
    ('semantic', 'confidence'): 'semantic_confidence',
    ('semantic', 'stable_updates'): 'stable_updates',
    ('word_reveal', 'enabled'): 'word_reveal_enabled',
    ('word_reveal', 'window_ms'): 'word_reveal_window_ms',
    ('word_reveal', 'min_interval_ms'): 'word_reveal_min_interval_ms',
    ('noise_reduction', 'enabled'): 'noise_enabled',
    ('noise_reduction', 'model'): 'noise_model',
    ('noise_reduction', 'frame_ms'): 'noise_frame_ms',
    ('noise_reduction', 'num_threads'): 'noise_num_threads',
    ('noise_reduction', 'workers'): 'noise_workers',
    ('noise_reduction', 'max_streams'): 'noise_max_streams',
    ('noise_reduction', 'max_backlog_ms'): 'noise_max_backlog_ms',
}
_SECTIONS = sorted({section for section, _ in _KEYS})
_TYPES = {f.name: f.type for f in fields(Settings)}


class ConfigError(ValueError):
    pass


def _coerce(name, value):
    kind = _TYPES[name]
    # ==, not `is`: Optional[float] is a new object once typing's cache has dropped it.
    if kind == Optional[float]:
        if value is None:
            return None
        kind = float
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigError(f'{name}: expected true/false, got {value!r}')
        return value
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f'{name}: expected an integer, got {value!r}')
        return value
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f'{name}: expected a number, got {value!r}')
        return float(value)
    if not isinstance(value, str) or not value:
        raise ConfigError(f'{name}: expected a non-empty string, got {value!r}')
    return value


def _validate(s: Settings):
    if not 1 <= s.port <= 65535:
        raise ConfigError(f'server.port out of range: {s.port}')
    if s.max_streams < 1:
        raise ConfigError('server.max_streams must be >= 1')
    if s.vad_mode not in (0, 1, 2, 3):
        raise ConfigError('vad.mode must be 0, 1, 2 or 3')
    if s.asr_backend not in ('native_vllm', 'qwen_asr'):
        raise ConfigError("asr.backend must be 'native_vllm' or 'qwen_asr'")
    if s.vad_backend not in ('silero', 'webrtc'):
        raise ConfigError("vad.backend must be 'silero' or 'webrtc'")
    if s.dtype not in ('auto', 'float16', 'bfloat16'):
        raise ConfigError("asr.dtype must be 'auto', 'float16' or 'bfloat16'")
    # speech_service.LANGUAGE_NAMES and AUTO_CODE; not imported, so that config needs no gRPC
    if s.default_language not in ('vi', 'en', 'auto'):
        raise ConfigError("asr.default_language must be 'vi', 'en' or 'auto'")
    if s.max_model_len < 1 or s.max_num_seqs < 1 or s.vad_workers < 1:
        raise ConfigError('asr.max_model_len, asr.max_num_seqs and vad.workers must be >= 1')
    if not 0.0 < s.vad_threshold < 1.0:
        raise ConfigError('vad.threshold must be between 0 and 1')
    if s.vad_neg_threshold is not None and not 0.0 <= s.vad_neg_threshold <= s.vad_threshold:
        raise ConfigError('vad.neg_threshold must be between 0 and vad.threshold')
    # Same constraints ASRService and WebRTCEndpointer enforce, reported at load time.
    if not 0.2 <= s.chunk_sec < s.max_utterance_sec:
        raise ConfigError('require 0.2 <= asr.chunk_sec < asr.max_utterance_sec')
    if not 0 < s.vad_pause_ms < s.vad_endpoint_ms:
        raise ConfigError('require 0 < vad.pause_ms < vad.endpoint_ms')
    if not 0.0 < s.gpu_memory_utilization <= 1.0:
        raise ConfigError('asr.gpu_memory_utilization must be in (0, 1]')
    if s.ingress_max_ms < 0 or s.audio_timeout_ms < 0:
        raise ConfigError('server.ingress_max_ms and server.audio_timeout_ms must be >= 0 (0 disables)')
    if s.asr_max_consecutive_errors < 1 or s.punct_workers < 1:
        raise ConfigError('asr.max_consecutive_errors and punctuation.workers must be >= 1')
    if s.punct_partial_timeout_ms < 1 or s.punct_commit_timeout_ms < 1:
        raise ConfigError('punctuation timeouts must be >= 1 ms')
    if not 1 <= s.word_reveal_min_interval_ms <= s.word_reveal_window_ms:
        raise ConfigError('require 1 <= word_reveal.min_interval_ms <= word_reveal.window_ms')
    if s.word_reveal_enabled and s.word_reveal_window_ms > s.chunk_sec * 1000:
        raise ConfigError('word_reveal.window_ms must not be longer than asr.chunk_sec')
    if not (s.asr_enabled or s.noise_enabled):
        raise ConfigError('asr.enabled and noise_reduction.enabled are both false: no task to serve')
    # The gateway packetizes multiples of 640 bytes (20 ms of PCM16 at 16 kHz).
    if s.noise_frame_ms not in (20, 40, 80):
        raise ConfigError('noise_reduction.frame_ms must be 20, 40 or 80')
    if s.noise_num_threads < 1 or s.noise_workers < 1 or s.noise_max_streams < 1:
        raise ConfigError('noise_reduction.num_threads, workers and max_streams must be >= 1')
    if not 0 <= s.noise_max_backlog_ms <= 5000:
        raise ConfigError('noise_reduction.max_backlog_ms must be between 0 (off) and 5000')
    if s.noise_enabled and s.noise_max_streams > s.max_streams:
        raise ConfigError('noise_reduction.max_streams must not be above server.max_streams')


def load_settings(path, overrides=None) -> Settings:
    """Read server.yaml. `overrides` maps Settings field -> value (None = not set)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f'config file not found: {path}')
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    if not isinstance(raw, dict):
        raise ConfigError(f'{path}: top level must be a mapping')
    values = {}
    for section, body in raw.items():
        if section not in _SECTIONS:
            raise ConfigError(f'{path}: unknown section {section!r} (allowed: {", ".join(_SECTIONS)})')
        if body is None:
            continue
        if not isinstance(body, dict):
            raise ConfigError(f'{path}: section {section!r} must be a mapping')
        for key, value in body.items():
            name = _KEYS.get((section, key))
            if name is None:
                raise ConfigError(f'{path}: unknown key {section}.{key}')
            values[name] = _coerce(name, value)
    for name, value in (overrides or {}).items():
        if value is not None:
            values[name] = _coerce(name, value)
    settings = Settings(**values)
    _validate(settings)
    return settings


def check_logging_config(path):
    """Parse logging.yaml without applying it (no handlers opened, no files created)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f'logging config not found: {path}')
    try:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f'{path}: top level must be a mapping')
    return cfg


def setup_logging(path=None, console=True):
    """Apply logging.yaml (dictConfig). Relative log file paths resolve against the working directory.

    console=False drops the handler named `console`; bin/start.sh uses it so that the
    unrotated logs/asr-server.out does not duplicate the rotated log file.
    """
    if path is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(name)s: %(message)s')
        return
    cfg = check_logging_config(path)
    if not console:
        (cfg.get('handlers') or {}).pop('console', None)
        for logger in [cfg.get('root') or {}, *(cfg.get('loggers') or {}).values()]:
            if 'console' in (logger.get('handlers') or []):
                logger['handlers'] = [h for h in logger['handlers'] if h != 'console']
    for handler in (cfg.get('handlers') or {}).values():
        filename = handler.get('filename') if isinstance(handler, dict) else None
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(cfg)
