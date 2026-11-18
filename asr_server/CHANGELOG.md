# Changelog

## Unreleased

### Added

- Word-by-word reveal of PARTIAL results, off by default. New keys in `config/server.yaml`:
  `word_reveal.enabled`, `word_reveal.window_ms`, `word_reveal.min_interval_ms`.

### Files changed

- New: `src/asr_server/word_pacer.py`, `tests/test_word_pacer.py`.
- Changed: `src/asr_server/speech_service.py`, `src/asr_server/server.py`, `src/asr_server/config.py`,
  `config/server.yaml`, `tests/test_config.py`, `README.md`, `HUONG_DAN_CAI_DAT.md`.
