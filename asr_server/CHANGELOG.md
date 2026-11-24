# Changelog

## Unreleased

### Added

- Task `noise_reduction`: DPDFNet-2 speech denoising on CPU (sherpa-onnx 2.0.2
  `OnlineSpeechDenoiser`), off by default. The audio goes back in every result as PCM16 16 kHz
  or G.711 PCMU 8 kHz, as `AudioChunk.output_audio_format` (tag 14) asks; one empty `is_final`
  at the end of the stream; `ts_end` is the `rtp_timestamp` of the latest chunk given to the
  model. Denoisers are loaded at start (`noise_reduction.max_streams`); with none free a stream
  gets `RESOURCE_EXHAUSTED` at once. One `noise_reduction stream ended` INFO line per stream.
  When more than `noise_reduction.max_backlog_ms` (500) of audio waits for the model, the stream
  ends with `RESOURCE_EXHAUSTED` so that the gateway plays the original audio.
  New keys in `config/server.yaml`: `noise_reduction.enabled`, `model`, `frame_ms`,
  `num_threads`, `workers`, `max_streams`, `max_backlog_ms`. Needs `sherpa-onnx==2.0.2` and `dpdfnet2.onnx`.
  Design: `docs/DPDFNet2_Task_Routing_Design_VI.md` at the repo root.
- `asr.enabled` (default `true`): `false` loads no ASR, VAD or punctuation model and answers
  `speech_to_text` with `UNIMPLEMENTED`, for a CPU server that only denoises.
- `tests/test_noise_real_model.py` runs the real DPDFNet-2 model when
  `ASR_TEST_DPDFNET2_MODEL` names it (skipped otherwise).
- Word-by-word reveal of PARTIAL results, off by default. New keys in `config/server.yaml`:
  `word_reveal.enabled`, `word_reveal.window_ms`, `word_reveal.min_interval_ms`.
- `bin/env.sh.example` sets the offline runtime variables and `CUDA_VISIBLE_DEVICES=0`. An existing
  `bin/env.sh` is kept: add the new lines to it by hand.
- `asr.log_results` in `config/server.yaml`, off by default: one INFO line `asr_result` per PARTIAL and
  COMMIT with the language, the number of Han characters, the model answer and the text before and
  after punctuation. The log then holds the content of the calls.
- `scripts/update.sh` saves the replaced version in `backup/<date>-<time>/` (last 10 kept,
  `ASR_BACKUP_KEEP`). With `--restart` it waits until the server accepts connections
  (`ASR_START_WAIT_SEC`, default 300 s) and reports a failure without rolling back.
- `bin/rollback.sh [--list | ID]` restores a saved version: code, `bin/`, `config/` and `VERSION`. The
  version it replaces is kept in `backup/rolled-back/`.
- `bin/wait_ready.sh` waits until the started process listens on its port and records
  `health=ok|failed` in `VERSION`.
- `asr.force_language` in `config/server.yaml`, off by default: the native vLLM backend transcribes
  the language of the stream (`vi`, `en`) instead of letting the model detect it, by appending
  `language <name><asr_text>` to the model's chat template. Needs the chat template in the model
  directory. Logs `model prompt language=... ends with ...` once per language.
- `asr.default_language` (`vi`, `en` or `auto`, default `vi`): language of a stream whose
  `AudioChunk.language` is empty or unknown. `AudioChunk.language` also accepts `auto`.

### Changed

- Saved versions in `backup/` (and `backup/rolled-back/`) are named in UTC, always after the
  newest saved name: in local time, a clock going back could make a newer version sort as older
  (wrong rollback default, newest version pruned). Names saved before stay in local time.
- `scripts/update.sh` reports missing models only for the tasks that are on: with
  `asr.enabled: false` it no longer asks for the ASR, punctuation and Silero models (a
  noise-only server is started by `--restart`); with `noise_reduction.enabled: true` it reports
  a missing `dpdfnet2.onnx` and a missing or too old `sherpa-onnx`, with the import error.
  The check runs before anything is changed: a running server whose models are missing is no
  longer stopped and restarted into a failure, the update is refused instead. Relative model
  paths are checked from the run directory, and a path with a quote is no longer skipped.
- `scripts/run_tests.sh` runs each test file that starts a real gRPC server (server and
  `tools/eval` tests) in its own interpreter (a full run in one process sometimes hung on Windows), goes on after a failure
  and accepts pytest options such as `-k`.
- `scripts/update.sh` stops when the installed code cannot read the configuration, because that
  version could not be rolled back to. Add the keys of a new version after the update, or pass
  `--ignore-old-config`.
- `scripts/update.sh --restart` also starts a server that was stopped, unless models are missing.
- A refused update writes the new defaults to `config/*.dist.new` and leaves `config/*.dist` alone.
- A stream without a language took the fixed `Vietnamese`; it now takes `asr.default_language`
  (same result with the default `vi`). The `start` log line shows the language of the stream.
- `speech.proto` comments and README: the engine also serves `noise_reduction`; `speech_to_text`
  ignores `output_audio_format`; `ts_start`/`ts_end` are ticks of `rtp_clock_rate` (8 per ms for
  G.711 and G.722, 16 for L16). Comments only: `speech_pb2.py` is unchanged. Two tests cover
  `speech_to_text` with the chunks the gateway sends for G.722 and L16 sessions.

### Fixed

- `bin/status.sh` reported a running server as failed when the configuration had become invalid, so
  `scripts/update.sh` replaced the code without stopping that server. `scripts/update.sh` now reads
  the pid file itself.
- `vad.neg_threshold` could be refused with "expected a non-empty string" once typing's cache had
  dropped `Optional[float]` (it depended on the modules imported before the configuration was read).

### Files changed

- New: `src/asr_server/word_pacer.py`, `bin/_upgrade.sh`, `bin/rollback.sh`, `bin/wait_ready.sh`,
  `tests/test_word_pacer.py`, `tests/test_result_log.py`, `tests/test_upgrade_rollback.py`,
  `tests/test_force_language.py`.
- Changed: `src/asr_server/speech_service.py`, `src/asr_server/server.py`, `src/asr_server/config.py`,
  `src/asr_server/text_clean.py`, `src/asr_server/asr_backend.py`, `src/asr_server/messages.py`,
  `config/server.yaml`, `bin/env.sh.example`, `bin/status.sh`, `scripts/build.sh`, `scripts/update.sh`,
  `tests/test_config.py`, `README.md`, `HUONG_DAN_CAI_DAT.md`, `docs/Thiet_ke_Speech_Engine_v1.1.md`.
