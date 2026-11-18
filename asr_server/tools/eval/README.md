# Qwen3-ASR evaluation client: 80 ms chunks over speech.proto, dual WER, GPU VRAM monitoring

> **Protocol (updated).** This client now speaks `speech.proto`
> (`SpeechStream.Recognize`), like the RTP gateway: PCM16 mono, **16 kHz by
> default** (`--rate 8000` to send 8 kHz), 80 ms `AudioChunk` messages,
> `chunk_seq` from 0, RTP timestamps on an 8 kHz clock (`--rtp-clock-rate`,
> `--rtp-start`), `end_of_stream` on the last chunk, `--language vi|en`.
> `--call-id` / `--leg-id` set `session_id` / `stream_id`.
> PARTIAL = `is_final=false`, COMMIT = `is_final=true`. The WER before
> punctuation needs `RecognitionResult.raw_text` (tag 22).
> In `stream_events.jsonl` each row has `seq`, `utterance_id`, `reason` (enum
> name), `ts_start` / `ts_end` (RTP ticks) and `start_ms` / `end_ms` relative to
> `--rtp-start`.
> The engine now strips the model markers itself, so `apply_server_text_fix.py`
> and the "Patch server.py" section below are obsolete; the client still strips
> markers defensively. Sections below that mention `StreamRecognize`,
> `asr.proto`, `commit_id` or `utterance_index` describe the previous protocol.
> A `--fast` run needs `server.ingress_max_ms: 0` and `asr.partial_max_backlog_ms: 0`
> on the engine; otherwise the stream is closed, or its PARTIAL inferences are
> skipped and `processing_rtf` only covers the FINALs.

This package is an update to the previous `qwen3_asr_grpc_wav_eval.zip` client.
It uses the ALREADY LOADED `server.py` model through `StreamRecognize`;
**it does not load a second Qwen checkpoint**.

## What changed

1. Strip repeated `language: Vietnamese<asr_text>` / `language Vietnamese<asr_text>`
   markers in PARTIAL and COMMIT, including mid-utterance. Cleaned text is
   passed to WER, `asr.txt`, and the normal `text` fields in JSON logs.
   The original server text is retained in `text_original` / `raw_text_original`
   for debugging. `raw_text` in regular logs is also sanitized.
2. Optional **NVML GPU VRAM sampling** (default every 100ms) while audio is
   being sent to the ASR server. Save `vram_samples.csv` and a VRAM summary
   under `metrics.json` with baseline, peak, end and post measurements.
3. Include `apply_server_text_fix.py` + `asr_text_clean.py` so the existing
   `server.py` can strip metadata BEFORE punctuation and COMMIT. Its internal
   `ASRStreamingState._raw_decoded` is intentionally left untouched.

## Requirements: client

```bash
python -m pip install -r requirements.txt
# For monitoring VRAM when running the evaluation client ON THE GPU server:
python -m pip install nvidia-ml-py
```

When the server is offline, download and transfer Linux/Python-ABI-compatible
wheels in advance. `nvidia-ml-py` provides `pynvml` and requires the NVIDIA
GPU driver libraries on the local machine.

## Run on SAME MACHINE as the GPU inference server (recommended)

```bash
python eval_wav_grpc.py \
  --wav /data/voice/recording.wav \
  --truth /data/voice/ground_truth.txt \
  --target 127.0.0.1:50051 \
  --chunk-ms 80 \
  --vram-mode required \
  --gpu-indices 0 \
  --vram-interval-ms 100 \
  --output-dir ./results
```

`--vram-mode required` fails early when NVML is not available instead of
reporting bogus zero VRAM. The default is `--vram-mode auto`, which uses NVML
only when the gRPC target looks local; if there is no NVML, the WER evaluation
still works and `metrics.json.vram.status` is `unavailable`.

Other options:

```bash
# Monitor all NVIDIA GPUs visible to NVML, typically including tensor-parallel shards
python eval_wav_grpc.py --wav recording.wav --truth ground_truth.txt \
  --vram-mode required --gpu-indices 0,1

# No monitoring and faster-than-realtime sending (throughput benchmark)
python eval_wav_grpc.py --wav recording.wav --truth ground_truth.txt \
  --fast --vram-mode off
```

### Important for REMOTE gRPC target

NVML in the evaluation client measures the GPU of **the client's OS host**;
`--target 10.x.x.x:50051` does NOT magically expose server GPU counters. To
measure model GPU VRAM, run this evaluation code **on the GPU server host** and
connect via `127.0.0.1:50051` (or a local network namespace mapping).

If the client MUST be on another host, run a GPU monitor on the GPU server
and correlate its time series with request time; a gRPC protocol extension
would be required to return remote NVML measurements automatically.
`--vram-mode local` allows client-local sampling with a remote endpoint,
but that is usually NOT an inference VRAM measurement.

## Output files

```text
results/
  asr.txt                # cleaned, committed transcript
  metrics.json           # WER + GPU VRAM summary
  stream_events.jsonl    # original and cleaned partial/commit texts
  vram_samples.csv       # GPU memory time series, when NVML is available
```

Example VRAM section in metrics.json (illustrative values only):

```json
{
  "vram": {
    "status": "ok",
    "sampling_interval_ms": 100,
    "per_gpu": [{
      "gpu_index": 0,
      "baseline_used_mib": 14300.0,
      "peak_used_mib": 15040.0,
      "peak_delta_mib": 740.0,
      "end_used_mib": 14720.0,
      "end_delta_mib": 420.0,
      "post_used_mib": 14320.0
    }]
  }
}
```

Note: absolute VRAM usage **includes checkpoint weights already loaded**,
CUDA graph/KV cache reservations and other GPU processes. `peak_delta_mib`
is NOT equivalent to model weights or exact per-request allocation. The
sampler can miss memory spikes shorter than its sampling interval; change
`--vram-interval-ms` to 20ms for a denser trace (increased CPU/NVML load).
For multi-GPU, report the per-GPU rows first; sums of per-device peaks need
not occur simultaneously.

## Patch server.py to remove wrappers BEFORE punctuation/COMMIT

With `server.py` from the earlier `qwen3_asr_grpc_vad_commit.zip` project:

```bash
# Copy these two files into the folder with your server.py:
#   apply_server_text_fix.py
#   asr_text_clean.py
python apply_server_text_fix.py ./server.py
# The script creates server.py.<timestamp>.bak before modifying.
# Restart the server process to activate the changes.
```

The patch replaces two lines inside server.py:

```python
from asr_text_clean import strip_asr_markers

# In feed() when a model chunk completes:
new_text = strip_asr_markers(asr_state.text or "")

# In finish() BEFORE punctuation and COMMIT:
raw = strip_asr_markers(asr_state.text or "")
```

Do NOT edit `asr_state._raw_decoded`, `state.prompt_raw`, or tokenizer prefix
rollback internals. Qwen uses them to continue generating transcript text.

If your `server.py` is a different version, the patch stops with a clear
error; compare the markers and apply these two replacements manually.
The evaluation client still removes markers even before the server is patched.

## Test without a GPU/model

```bash
python -m pip install pytest
python -m pytest -q tests
```

These tests verify WAV chunking/streaming over a mock gRPC server,
WER normalization, prefix removal and NVML aggregation via fake GPU data.
**No Qwen checkpoint inference or real NVIDIA GPU was exercised here.**

## UPDATE: log raw Qwen transcript + after-punctuation transcript, and BOTH WERs

This version preserves the existing **80 ms gRPC chunk streaming, prefix cleanup,
partial/commit processing and NVML VRAM monitoring**, and now emits the following
separate artifacts from **COMMIT** messages:

```text
results/
  asr_before_punctuation.txt   # cleaned Qwen raw_text (before HF punctuation)
  asr_after_punctuation.txt    # cleaned formatted text (after punctuation)
  asr.txt                      # backward compatible alias of after punctuation
  metrics.json                 # two WER objects + unchanged VRAM summary
  stream_events.jsonl         # two transcript fields for EVERY partial/commit
  vram_samples.csv             # when NVML monitoring is enabled
```

### Start the evaluation (server.py must already be running)

```bash
python eval_wav_grpc.py \
  --wav /data/test.wav \
  --truth /data/ground_truth.txt \
  --target 127.0.0.1:50051 \
  --chunk-ms 80 \
  --vram-mode required \
  --gpu-indices 0 \
  --output-dir ./results
```

The script prints both `ASR BEFORE PUNCTUATION` and `ASR AFTER PUNCTUATION`,
plus both `WER`, `S/D/I` and reference/hypothesis word counts.

### metrics.json fields

```json
{
  "asr_before_punctuation": "Xin chào em",
  "asr_after_punctuation": "Xin, chào em.",
  "wer_before_punctuation": {
    "wer": 0.3333333333,
    "wer_percent": 33.333,
    "substitutions": 1,
    "deletions": 0,
    "insertions": 0
  },
  "wer_after_punctuation": {
    "wer": 0.3333333333,
    "wer_percent": 33.333,
    "substitutions": 1,
    "deletions": 0,
    "insertions": 0
  },
  "raw_text_available_for_all_commits": true,
  "missing_raw_utterance_indices": [],
  "lexical_content_changed_by_punctuation": false,
  "vram": {"status": "unavailable", "reason": "Example: VRAM off"}
}
```

Example is illustrative, and the actual JSON has additional metadata and
full error-count details. The legacy keys `asr_raw_text`, `asr_text` and
`metrics` are retained; `metrics` remains an alias of `wer_after_punctuation`
for compatibility with earlier scripts.

The `stream_events.jsonl` rows now also contain `asr_before_punctuation`,
`asr_after_punctuation` and `raw_text_available`, while retaining `raw_text`,
`text`, `raw_text_original` and `text_original`.

### Critical: why both WER values may be equal

Both WER calculations normalize Unicode, casefold and **remove punctuation**
while preserving Vietnamese diacritics. Thus if the punctuation model only
inserts commas, periods and question marks (and preserves recognized words),
`WER_before == WER_after` is the CORRECT result, not a bug.

The script also emits `lexical_content_changed_by_punctuation` to warn if
formatted text actually changed the lexical content, in which case the two
WERs can differ. To **evaluate punctuation accuracy**, use a separate
punctuation precision/recall/F1 against punctuated ground truth; WER alone
cannot measure punctuation quality.

### Server raw-text contract

The gRPC server must supply both fields on each COMMIT:

```python
raw = strip_asr_markers(asr_state.text or "")
formatted = await self.punctuate(raw)
# COMMIT: raw_text=raw, text=formatted.text
```

If `raw_text` is absent/empty while `text` has content, the client marks
`wer_before_punctuation=null` and names the problematic utterances under
`missing_raw_utterance_indices`. It never fabricates a raw transcript from
post-punctuation text. Update server.py to return `raw_text` if needed.

### Validation

```bash
python -m pytest -q tests
```

19 tests passed locally (including mock gRPC, prefix removal, two independent
WERs, missing-raw handling and mock VRAM). Actual vLLM GPU inference was not
run in this environment.
