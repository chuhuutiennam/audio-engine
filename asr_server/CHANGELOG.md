# Changelog

## Unreleased

### Added

- Word-by-word reveal of PARTIAL results, off by default. New keys in `config/server.yaml`:
  `word_reveal.enabled`, `word_reveal.window_ms`, `word_reveal.min_interval_ms`.
- `bin/env.sh.example` sets the offline runtime variables and `CUDA_VISIBLE_DEVICES=0`. An existing
  `bin/env.sh` is kept: add the new lines to it by hand.
- `asr.log_results` in `config/server.yaml`, off by default: one INFO line `asr_result` per PARTIAL and
  COMMIT with the language, the number of Han characters, the model answer and the text before and
  after punctuation. The log then holds the content of the calls.

### Files changed

- New: `src/asr_server/word_pacer.py`, `tests/test_word_pacer.py`, `tests/test_result_log.py`.
- Changed: `src/asr_server/speech_service.py`, `src/asr_server/server.py`, `src/asr_server/config.py`,
  `src/asr_server/text_clean.py`, `config/server.yaml`, `bin/env.sh.example`, `tests/test_config.py`,
  `README.md`, `HUONG_DAN_CAI_DAT.md`.
