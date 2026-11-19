# asr_server: Qwen3-ASR 1.7B + native vLLM, gRPC, Silero VAD, punctuation, immutable commit

**Cài đặt và sử dụng trên server GPU (tiếng Việt): [`HUONG_DAN_CAI_DAT.md`](HUONG_DAN_CAI_DAT.md).**

## Project layout

```text
asr_server/
├── pyproject.toml            package metadata, pytest settings
├── requirements.txt          runtime dependencies (vllm[audio] and torch come from the ASR environment)
├── requirements-dev.txt      + pytest, soundfile
├── config/
│   ├── server.yaml           all server parameters (formerly command-line flags)
│   └── logging.yaml          log level, file, daily rotation (14 days)
├── bin/                      start.sh, stop.sh, status.sh (Linux); env.sh.example (local settings)
├── scripts/                  build.sh, update.sh (deploy to the run directory), build_proto.py,
│                             run_tests.sh, client.py (demo client, speech.proto)
├── src/asr_server/           speech_service.py (gRPC, speech.proto), server.py (recognition core),
│                             messages.py, asr_backend.py, vad.py, audio.py, punctuation.py,
│                             text_clean.py, config.py, proto/ (speech.proto + bindings)
├── tests/                    server and deployment-script tests (no GPU or checkpoints needed)
├── tools/eval/               WER / VRAM evaluation client (eval_wav_grpc.py)
├── data/                     sample audio and ground truth for evaluation (not tracked)
└── logs/                     asr-server.log, asr-server.out, asr-server.pid (not tracked)
```

This server uses **native `vllm.LLM.chat()`**; it does **NOT** import
`qwen_asr` or require the `qwen-asr` package. It uses an offline local
Qwen3-ASR checkpoint and local Silero TorchScript VAD checkpoint. The Python
process loads **one** `vllm.LLM` instance into the GPU. Do not start another
copy of the checkpoint on the same GPU without checking VRAM.

Backends are chosen in `config/server.yaml`:

| Key | Default | Alternative |
|---|---|---|
| `asr.backend` | `native_vllm`: `vllm.LLM.chat()` on the audio accumulated for the utterance; language auto-detected | `qwen_asr`: streaming state of the `qwen_asr` package (needs `qwen-asr[vllm]`) |
| `vad.backend` | `silero`: Silero TorchScript on CPU, 32 ms frames | `webrtc`: WebRTC VAD, 20 ms frames (needs `webrtcvad-wheels`) |

The rest of this document describes the defaults.

## Processing

SIP/RTP -> media gateway (RTP header removed, jitter/reorder/PLC, G.711 decoded
and resampled to PCM16 16 kHz) -> `SpeechStream.Recognize` gRPC (`AudioChunk`,
80 ms) -> **Silero TorchScript VAD on CPU at the rate received** -> voiced utterance + pre-roll -> `soxr` 16 kHz
resampler -> accumulated PCM16k WAV (memory-only data URI) -> `vllm.LLM.chat()` -> local HF token
classification punctuation model -> PARTIAL / FINAL `RecognitionResult`.

Each direction of a call has its own gRPC stream, VAD state, audio accumulation
and chunk sequence. Only voiced utterances reach ASR. Every `asr.chunk_sec`
(default: 1 second), the server calls `LLM.chat()` with the **entire utterance
so far**, creating a PARTIAL if the text changes. At the Silero endpoint, it
recognizes any unprocessed remainder and creates a FINAL. This is windowed
*re-inference*, **NOT true KV-cached/token-by-token streaming**: it costs
more GPU work as utterances get longer. Max utterance is bounded to 15s by
default. Audio data URIs require no
network and are created in RAM, not on disk.

Language: native vLLM chat auto-detects the spoken language. The model answers
`language Vietnamese<asr_text>...`; the server removes the marker and uses the
language it names as `Transcript.language` and as the language given to
punctuation. The language sent by the client is only the value reported until
the model has named one. It is not forced on the model.

## gRPC contract (`speech.proto`)

`service SpeechStream { rpc Recognize(stream AudioChunk) returns (stream RecognitionResult); }`
The proto is `src/asr_server/proto/speech.proto`. The gateway (rtpgw, project
`media_ai`, `internal/ai/proto/speech.proto`) owns tags 1-14 of
`RecognitionResult`; tags 12-14 are its `topic_label`, `topic_info` and
`topic_score`, which this engine never sets. The engine adds tags 20-24. The
gateway decodes by tag and skips tags it does not know, so it needs no change.

### Request: `AudioChunk`

One RPC carries one direction of one call. There is no separate configuration
message: the first chunk fixes the stream.

| Field | Rule |
|---|---|
| `session_id`, `stream_id` | Mandatory. Must not change within the RPC. |
| `pcm` | Signed 16-bit little-endian mono. Length must be even. Normally 80 ms (2560 bytes at 16 kHz); other sizes are accepted. |
| `sample_rate` | `16000` (or `8000`). Must not change. |
| `channels` | `1` (unset is accepted). |
| `rtp_timestamp` | RTP timestamp of the **first sample** of the chunk, on the source clock. The first chunk anchors the timeline. A later value that does not continue from the audio received (a jump) re-anchors the timeline at that chunk and is logged at WARNING (the first 3 jumps of a stream, then every 500th); processing goes on. |
| `rtp_clock_rate` | Source RTP clock, `8000` for G.711. `0` means "use `sample_rate`" as the proto defines, and is logged at WARNING. Must not change. |
| `chunk_seq` | Normally increases by 1 from its first value. A gap means the gateway dropped chunks from its send queue: it is logged at WARNING (the first 3 gaps of a stream, then every 500th) and processing goes on with the audio that arrived. `-1` on the first chunk turns the check off. |
| `duration_ms` | Informational; the duration is computed from `pcm`. |
| `end_of_stream` | `true`: the audio in this chunk is processed, the open utterance is committed, the stream is closed. Nothing after it is read. Closing the request stream without it has the same effect. |
| `language` | `vi`, `en` or empty. With the default backend the model auto-detects the language, so this is only the value reported until the model names one. Unknown codes are logged and treated as empty. Read from the first chunk only. |
| `language_source` | Used instead of `language` only when `asr.use_language_source: true`. |
| `task` | `speech_to_text` or empty. The gateway's other tasks (`realtime_translation`, `keyword`, `voice_controller_emoji`) end the RPC with `UNIMPLEMENTED`; anything else with `INVALID_ARGUMENT`. Must not change. |

Send audio at real-time pace. The engine never slows the sender down; if more
than `server.ingress_max_ms` of audio is waiting for ASR the RPC ends with
`RESOURCE_EXHAUSTED`. The gateway's resampler to 16 kHz must be stateful across
chunks: resampling each 80 ms chunk on its own creates discontinuities.

### Response: `RecognitionResult`

| Field | PARTIAL | FINAL |
|---|---|---|
| `is_final` | `false` | `true` |
| `text` | The **entire current** utterance, formatted for display. REPLACE the previous PARTIAL of the same `utterance_id`; never concatenate. Empty `text` means "drop the text shown for this utterance". | Immutable final text. Append once. |
| `raw_text` (22) | ASR text before punctuation | ASR text before punctuation, model markers removed |
| `utterance_id` (20) | Starts at 1. All PARTIALs up to a FINAL carry that FINAL's id. | Advances by 1 after each FINAL. |
| `revision` (21) | Increases with every result of the utterance | Last revision + 1; restarts at 1 for the next utterance |
| `seq` | Starts at 1, +1 for every result of the stream, no gaps | same |
| `ts_start`, `ts_end` | RTP timestamps on the source clock, **modulo 2^32**, half-open `[ts_start, ts_end)`. Derived from the `rtp_timestamp` of the chunk that carried the audio (the latest anchor at or before that position) and the audio position; when `rtp_timestamp` jumped inside an utterance, `ts_start` is on the old timeline and `ts_end` on the new one. From VAD, **not** word timestamps; pre-roll and hangover are included. After a wrap `ts_end` is numerically smaller than `ts_start`. | same; `ts_end` is the end of the utterance |
| `language` | `vi` or `en`: the language the model reports, else the one configured. Empty when the model reports any other language (logged at WARNING, the utterance is processed normally). | same |
| `reason` (23) | `COMMIT_REASON_UNSPECIFIED` | See below |
| `status` (24) | `RESULT_STATUS_OK` | `RESULT_STATUS_OK` (other values are reserved) |
| `confidence` | `0.0`: no confidence is available, not "0 %" | same |
| `audio_payload`, `audio_pt` | Unused | Unused |

`reason` on a FINAL:

- `COMMIT_REASON_SILENCE`: ~600 ms silence after voiced speech. Also used when an
  utterance is open and no audio arrives for `server.audio_timeout_ms` (logged at
  WARNING as `audio_timeout`).
- `COMMIT_REASON_SEMANTIC_PAUSE`: OPTIONAL early boundary: ~300 ms pause, at least
  N repeated ASR updates with identical full text, punctuation classifier
  strong terminal mark `.` / `?` / `!` above threshold; this is heuristic.
  Default is DISABLED. Commas never trigger early commit.
- `COMMIT_REASON_MAX_DURATION`: forced cut at 15 s to bound GPU cost. The next
  piece starts at the cut: its `ts_start` equals the previous `ts_end`.
- `COMMIT_REASON_END_OF_STREAM`: `end_of_stream` or the request stream closed.
- `COMMIT_REASON_AUDIO_GAP` is reserved and not emitted.

Without tags 20-24 (the gateway, or any client built from the gateway's proto)
the pairing rule is positional: every PARTIAL belongs to the next FINAL.

Cases that produce no FINAL (they are logged, not reported in `status`):

- A PARTIAL with **empty `text`** is sent when an utterance that already showed
  a PARTIAL ends without a FINAL: the final text is empty (logged at DEBUG) or
  the model raised (logged at ERROR). The stream stays open and the next
  utterance reuses the `utterance_id`; `revision` keeps increasing.
  The gateway does not pass this one on: it drops every result that has neither
  text nor topic, so MF never sees the clearing PARTIAL (accepted; the text
  shown is replaced by the next result of the stream).
- No PARTIAL is sent before the utterance has `vad.min_voiced_ms` of voiced
  audio, so an utterance the VAD later discards never shows a PARTIAL.

### Word-by-word reveal (optional)

With `word_reveal.enabled: true`, one PARTIAL of the model is sent as several
PARTIALs, each carrying one more word. The contract above does not change: every
PARTIAL is still the entire text to show, and `revision` and `seq` still grow by 1.

- The first new word goes out at once, the others within `word_reveal.window_ms`.
  Two results are at least `word_reveal.min_interval_ms` apart; when more words
  wait than fit, they go out in groups.
- A word the model corrected is replaced at the next send.
- A FINAL or an empty PARTIAL goes out at once and cancels the words still waiting.
  The text and timestamps of a FINAL are the same with the reveal on or off; only
  its `revision` is higher, because more results came before it.

The last word of a PARTIAL reaches the client up to `window_ms` later. With the
default timing one PARTIAL of the model becomes at most 4 results, so at
`asr.chunk_sec: 0.5` a stream sends at most 8 results per second. Use it with
`asr.chunk_sec: 0.5`; `window_ms` must not be longer than `asr.chunk_sec`.

### Status codes

| Code | When |
|---|---|
| `INVALID_ARGUMENT` | Missing ids, bad `sample_rate`/`channels`, unknown `task`, odd `pcm` length, a session field changed |
| `UNIMPLEMENTED` | `task` is `realtime_translation`, `keyword` or `voice_controller_emoji` |
| `RESOURCE_EXHAUSTED` | More than `server.ingress_max_ms` of audio waiting for ASR, or a chunk above 64 KB |
| `UNAVAILABLE` | `asr.max_consecutive_errors` model failures in a row |

What the gateway does with them: `INVALID_ARGUMENT` and `UNIMPLEMENTED` are
permanent for it, it does not reconnect and closes the session. After
`RESOURCE_EXHAUSTED` or `UNAVAILABLE` it reconnects up to `ai.max_retry` (3)
times with a growing backoff, then closes the session; the new stream starts
with a `chunk_seq` other than 0, which is accepted.

A client that is still writing when the engine ends the RPC may see `INTERNAL`
instead of the code above; do not branch on the exact code.

### Failure and overload behaviour

- A model error loses that utterance only.
- When the model is slower than the audio, PARTIALs get sparser before any stream
  is closed: while more than `asr.partial_max_backlog_ms` (500) of a stream's audio
  is waiting, its PARTIAL inference is skipped and the audio is kept for the next
  one. A FINAL is always inferred on all the audio of the utterance. The first skip
  of a stream is logged at WARNING (`asr backlog`), and every `commit` log line
  carries `skipped_partials=N`. This needs `asr.backend: native_vllm`, which
  infers once per call; the `qwen_asr` package infers once per chunk inside each
  call, so there the work is only delayed.
- `server.ingress_max_ms` stays the last resort. A stream waits for its turn on
  the single model thread, so it is still closed with `RESOURCE_EXHAUSTED` when
  one round of model calls over all busy streams takes longer than about
  `ingress_max_ms - partial_max_backlog_ms` (2.5 s).
- A client that sends faster than real time (for example
  `tools/eval/eval_wav_grpc.py --fast`) hits the ingress limit and would have
  its PARTIAL inferences skipped; set `server.ingress_max_ms: 0` and
  `asr.partial_max_backlog_ms: 0` on a test server for such runs.
- Punctuation has a deadline: a PARTIAL falls back to unpunctuated text after
  `punctuation.partial_timeout_ms`, a FINAL falls back to `raw_text` after
  `punctuation.commit_timeout_ms` or when the punctuation model raises.
  Punctuation restoration does not change words or truecase capitalization.

The punctuation step is given the language of the utterance (the model's name
for it, for example `Vietnamese`): `punctuator.restore(text, language=...)`.
The current multilingual model does not use it; it is there so a model or
threshold can be chosen per language.

The punctuation model is not an acoustic endpointer. A question mark is
not enough to commit: VAD or stream termination is the primary signal.
For single-sentence lower latency, enable the conservative semantic pause
heuristic only after validating on G.711 call recordings.

## Deploying to a run directory

The server does not have to run from the source tree. `scripts/update.sh`
installs it into a run directory (default `/u01/ims/audio-engine`, or
`--app-home DIR`, or `$ASR_APP_HOME`):

```text
<app home>/
├── bin/      start.sh stop.sh status.sh   replaced on every update
│             rollback.sh wait_ready.sh _upgrade.sh   replaced on every update, kept by a rollback
│             env.sh                        local settings (PYTHON, CUDA_VISIBLE_DEVICES); created once
├── config/   server.yaml logging.yaml      created once, never overwritten
│             *.dist                        defaults of the installed version
├── lib/      asr_server/                   application code, replaced on every update
│             asr_server.prev/              the version before
├── backup/   <date>-<time>/                replaced versions (code, bin, config), last 10 kept
│             rolled-back/<date>-<time>/    versions replaced by a rollback, last 10 kept
├── logs/  model/  data/                    never touched
└── VERSION
```

```bash
bash scripts/update.sh              # build (scripts/build.sh), then install or update
bash scripts/update.sh --restart    # same, restarting the server and waiting until it is ready
bash scripts/build.sh --test        # run the tests, then build into build/app
<app home>/bin/rollback.sh --list   # show the saved versions
<app home>/bin/rollback.sh [ID]     # go back to a saved version
```

- On the first install the model paths of `config/server.yaml` are set to
  `<app home>/model/...`; missing model files are reported.
- An update validates the configuration found in the run directory against the
  new code and the installed code first. If it is not valid, or the server is
  running and `--restart` was not given, nothing is changed. Add the keys of a
  new version after the update.
- A server that dies or does not open its port within `ASR_START_WAIT_SEC`
  seconds (default 300) is reported with exit code 1 and is **not** rolled back,
  so that the logs can be examined first.
- `ASR_BACKUP_KEEP` and `ASR_START_WAIT_SEC` are read from `bin/env.sh`.
- `bin/*.sh` are the same scripts in both places: they use `lib/` when it
  exists and `src/` otherwise, and read `bin/env.sh` when it exists.

Step-by-step instructions (Vietnamese): `HUONG_DAN_CAI_DAT.md`.

## Prerequisites on offline GPU server

- Linux + compatible NVIDIA driver, CUDA/PyTorch and **`vllm[audio]`**
  (vLLM version supporting Qwen3-ASR; 0.16.0+ is documented in vLLM recipes).
  **`qwen-asr` is not needed**.
- Python deps: grpcio, protobuf, numpy, soxr, transformers, and compatible
  PyTorch. See `requirements.txt`. **No WebRTC VAD package, onnxruntime,
  torch.hub, or Silero Python package is required.**
- Model weights on local disk **before** starting the server:
  `/opt/models/Qwen3-ASR-1.7B`,
  `/opt/models/punctuation-restoration-small`, and a trusted official Silero
  TorchScript file `/opt/models/silero_vad.jit`.
- Both checkpoints must include all configs, tokenizer / preprocessor files
  and weight shards. Model `Horizon-Labs/punctuation-restoration-small` is a
  140M multilingual punctuation token classifier that includes Vietnamese.

### Prepare on a machine WITH Internet

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
hf download Horizon-Labs/punctuation-restoration-small \
  --local-dir ./models/punctuation-restoration-small
# Official Silero TorchScript checkpoint: one standalone file (~2 MB).
curl -fL --retry 3 \
  'https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit' \
  -o ./models/silero_vad.jit
# Download from the official repository only; TorchScript contains executable code.

# Use the SAME OS / Python ABI / CUDA-compatible wheel sources as your server.
python -m pip download -d wheelhouse -r requirements.txt
# Download vLLM audio wheels for the exact offline CUDA/Python environment too:
python -m pip download -d wheelhouse "vllm[audio]>=0.16.0"
# CUDA/Torch/vLLM wheels must match the deployment driver, ABI and platform.
```

Transfer `models/` (including `silero_vad.jit`), `wheelhouse/`, and project
source via SCP/USB/internal artifact registry. Check that the .jit file is
present and readable before launch. For vLLM and CUDA, a built-and-tested container image
(`docker save` / `docker load`) is usually safer than arbitrary wheels.

### Install on offline server (CUDA + vLLM environment)

```bash
python -m pip install --no-index --find-links=/opt/wheelhouse \
  -r requirements.txt
# Only if vLLM/audio extras are missing and all compatible wheels are local:
python -m pip install --no-index --find-links=/opt/wheelhouse "vllm[audio]"
```

### Start

All parameters live in `config/server.yaml` (the values shipped there are the
former command-line defaults). Unknown keys are rejected at startup.

```bash
bin/start.sh          # background; pid in logs/asr-server.pid
bin/status.sh         # exit code 0 = running, 3 = not running
bin/stop.sh           # SIGTERM, waits up to 15 s; --force to SIGKILL
```

`bin/start.sh` needs no installation: it puts `src/` (source tree) or `lib/` (run
directory) on `PYTHONPATH`. Environment overrides: `PYTHON`, `ASR_SERVER_CONFIG`,
`ASR_SERVER_LOGGING_CONFIG`; `bin/env.sh`, when present, is read first and wins.

Foreground run and configuration check:

```bash
PYTHONPATH=src python -m asr_server --config config/server.yaml
PYTHONPATH=src python -m asr_server --check-config     # validate, print, exit
PYTHONPATH=src python -m asr_server --port 50052       # --host / --port override the file
```

The server sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1` itself.

Logs: `logs/asr-server.log`, rotated at midnight, 14 days kept
(`config/logging.yaml`). When started through `bin/start.sh`, anything printed
outside the logging system goes to `logs/asr-server.out`, which is not rotated.

With `asr.log_results: true`, every PARTIAL and COMMIT of the model adds one INFO
line `asr_result` with the language, the number of Han characters, the model answer
and the text before and after punctuation. It is off by default because the log
then holds the content of the calls.

Optional early semantic commits (higher risk of over-segmentation), in
`config/server.yaml`:

```yaml
asr:
  chunk_sec: 0.5
semantic:
  early_commit: true
  confidence: 0.90
  stable_updates: 2
```

Silero is stateful. The server loads an independent CPU VAD model instance
per gRPC stream, so no VAD history leaks between concurrent call legs. Silero
requires **32 ms frames**: 256 samples at 8 kHz and 512 at 16 kHz; incoming
20 ms RTP payloads are buffered automatically. `vad.threshold` controls
speech onset (default 0.5); `vad.neg_threshold` controls exit (default
threshold minus 0.15); `vad.mode` applies only to `vad.backend: webrtc`.
The checkpoint is loaded and run on `vad.workers` threads, not on the event loop.

The server validates the local Silero JIT checkpoint with a silent 32 ms CPU
frame before allocating the Qwen model on GPU. The `.jit` model file is NOT
included in this project; download and transfer it separately from the
official Silero repository as shown above.

`punctuation.model` is REQUIRED at startup; server **fails fast** when
punctuation weights are absent. An identity-only punctuation fallback is
included only for tests and must not be used as if it adds punctuation.

Verify port:

```bash
ss -lntp | grep 50051
```

The demo binds plaintext TCP 50051. Production telecom deployments MUST
restrict network access and add mTLS / authentication, rate limiting,
observability and capacity management. gRPC itself is not SIP/RTP.

## Test client

```bash
# WAV 16-bit PCM, mono, 8 or 16 kHz; sent as 16 kHz, 80 ms chunks
PYTHONPATH=src python scripts/client.py --wav call_audio.wav --realtime

# Raw PCMA payload file (no RTP header): decoded and resampled like the gateway does
PYTHONPATH=src python scripts/client.py --g711 call_audio.alaw --codec pcma --realtime
```

The client sends `end_of_stream` on the last chunk. Expected output:

```text
[PARTIAL seq=1 utt=1:1] xin chao anh
[PARTIAL seq=2 utt=1:2] xin chao anh, em goi ...
[FINAL   seq=3 utt=1 COMMIT_REASON_SILENCE lang=vi rtp 0-21504] xin chao anh, em goi ... .
[PARTIAL seq=4 utt=2:1] anh cho em hoi ...
[FINAL   seq=5 utt=2 COMMIT_REASON_END_OF_STREAM lang=vi rtp 24576-40960] anh cho em hoi ...?
```

All text shown above is illustrative: punctuation depends on model output.

## Bindings

`speech_pb2.py` and `speech_pb2_grpc.py` are supplied, so the GPU server does not
need `protoc` or `grpcio-tools`, and they carry no protobuf/grpcio version
checks. After editing `speech.proto`, run `python scripts/build_proto.py` on a
development machine with `grpcio-tools`; it rewrites `speech_pb2.py` for the
server and for `tools/eval`. `speech_pb2_grpc.py` is written by hand; update it
if the service methods change. For a Go media gateway, generate Go bindings from
the same `.proto`. Send each call direction on its own RPC; do not mix both
speakers before ASR.

## Run tests without a GPU or HF checkpoints

```bash
python -m pip install -r requirements-dev.txt torch
scripts/run_tests.sh            # server tests, then tools/eval tests
python -m pytest tests -q       # server tests only
```

Unit/integration tests use a deterministic fake ASR and local fake punctuation
model; this tests streaming, VAD transitions, event semantics and word
preservation, **not** checkpoint WER, punctuation F1 or GPU latency.

## Deployment limitations

1. vLLM `LLM.chat()` calls are synchronous and this server runs every model
   call of every stream on one worker thread, in order. That prevents
   concurrent model calls (also when an RPC is cancelled mid-inference) but is
   not an optimized multi-session scheduler: N busy streams take N times as
   long. Measure capacity before relying on `server.max_streams`.
2. Partial punctuation is provisional; a FINAL is never revised.
3. Silero VAD can misclassify music, hold tones, noise and very quiet speech.
   Validate with actual RTP G.711 data, silence, burst loss, and crosstalk.
4. Model-based punctuation may be wrong despite correct text; use F1 against
   human references. Tune silence 400/600/800ms and compare 0.5/1s ASR chunk.
5. Keep `raw_text` and `text` for QA and troubleshooting. Never let a
   generative postprocessor silently change phone numbers, service codes
   or content of an immutable FINAL.

## Evaluation client

`tools/eval/` holds the WER / VRAM evaluation client; see `tools/eval/README.md`.
It speaks `speech.proto` like the gateway (16 kHz, 80 ms chunks, `chunk_seq`
from 0, RTP clock 8000, `end_of_stream` on the last chunk) and is standalone: it
has its own copy of the bindings.

```bash
cd tools/eval
python eval_wav_grpc.py --wav ../../data/recording.wav --truth ../../data/ground_truth.txt \
  --target 127.0.0.1:50051 --output-dir ../../data/eval_out
```

The WER before punctuation uses `RecognitionResult.raw_text` (tag 22).

Throughput measurement (`--fast`, reads `processing_rtf` from `metrics.json`)
needs `server.ingress_max_ms: 0` and `asr.partial_max_backlog_ms: 0` on the
server, see "Failure and overload behaviour". Without the second one the
PARTIAL inferences are skipped and the RTF only covers the FINALs.
