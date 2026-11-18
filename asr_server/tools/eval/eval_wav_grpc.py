#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send a WAV to the ASR engine as 80-ms AudioChunk messages (speech.proto,
SpeechStream.Recognize); evaluate both raw and punctuated WER.

This client does NOT load a model. It sends what the RTP gateway sends: PCM16_LE
mono 16 kHz by default, with chunk_seq from 0 and RTP timestamps on an 8 kHz
clock, and end_of_stream on the last chunk. The input WAV may be multichannel,
floating-point or use another sampling rate. The last chunk may be <80 ms.

The WER before punctuation needs RecognitionResult.raw_text (tag 22).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any

import grpc
import numpy as np
import soundfile as sf
import soxr

import speech_pb2 as pb
import speech_pb2_grpc as pb_grpc
from asr_text_clean import strip_asr_markers
from vram_monitor import GpuVramMonitor, NvidiaNvmlBackend

LOGGER = logging.getLogger("asr_wav_eval")


def normalize_wer(text: str) -> str:
    """WER normalization: NFC, casefold, remove punctuation/symbols, whitespace.

    Preserve Vietnamese accents, digits and spoken words. Because Vietnamese
    normally uses spaces between syllables, this is a *syllable-token WER*.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    cleaned = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(cleaned.split())


def wer_details(reference: str, hypothesis: str) -> dict[str, Any]:
    """Exact word-level Levenshtein with insertion/deletion/substitution counts.

    Complexity O(N*M) time and O(M) memory, where M=len(hypothesis tokens).
    WER is not clipped to 100%; insertions can make it >100%.
    """
    ref = normalize_wer(reference).split()
    hyp = normalize_wer(strip_asr_markers(hypothesis)).split()
    n, m = len(ref), len(hyp)
    if n == 0:
        raise ValueError("Ground truth has no words after WER normalization")

    # A cell is (distance, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(m + 1)]
    for i, source in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, target in enumerate(hyp, 1):
            if source == target:
                # Matching is always preferred when free.
                item = previous[j - 1]
            else:
                diagonal = previous[j - 1]
                delete = previous[j]
                insert = current[j - 1]
                candidates = [
                    (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
                ]
                # Prefer substitution, then deletion, then insertion on ties.
                item = min(enumerate(candidates), key=lambda x: (x[1][0], x[0]))[1]
            current.append(item)
        previous = current
    edits, substitutions, deletions, insertions = previous[m]
    return {
        "wer": edits / n,
        "wer_percent": round(100.0 * edits / n, 3),
        "reference_token_count": n,
        "hypothesis_token_count": m,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": edits,
        "correct_words": n - substitutions - deletions,
        "reference_normalized": " ".join(ref),
        "asr_normalized": " ".join(hyp),
    }


def load_wav_as_pcm16(path: Path, target_sample_rate: int | None, channel: int | None = None) -> tuple[bytes, dict[str, Any]]:
    """Decode, mono downmix, convert to 8/16k, then encode S16LE."""
    if not path.is_file():
        raise FileNotFoundError(f"WAV not found: {path}")
    info = sf.info(str(path))
    if info.format != "WAV":
        raise ValueError(f"Only WAV input is supported; got {info.format!r}")
    data, original_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError("Input WAV contains zero audio samples")
    if channel is not None:
        if channel < 0 or channel >= data.shape[1]:
            raise ValueError(f"--channel {channel} out of range for {data.shape[1]}-channel WAV")
        LOGGER.info("Selecting WAV channel %d/%d", channel, data.shape[1])
        mono = data[:, channel]
    else:
        if data.shape[1] > 1:
            # In telecom recordings, two channels may be separate call legs.
            LOGGER.warning("WAV has %d channels; downmixing to mono. Use --channel 0 or 1 for a specific call leg.", data.shape[1])
        mono = np.mean(data, axis=1, dtype=np.float32)
    # The gateway always delivers 16 kHz; 8 kHz only on request.
    rate = 16000 if target_sample_rate is None else target_sample_rate
    if rate not in (8000, 16000):
        raise ValueError("Target sample rate must be 8000 or 16000")
    if int(original_sr) != rate:
        LOGGER.info("Resampling WAV %d Hz -> %d Hz", original_sr, rate)
        mono = soxr.resample(mono, int(original_sr), rate).astype(np.float32)
    mono = np.nan_to_num(mono, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    mono = np.clip(mono, -1.0, 1.0 - 1.0 / 32768)
    pcm_bytes = np.round(mono * 32768.0).astype("<i2").tobytes()
    metadata = {
        "file": str(path.resolve()),
        "original_sample_rate": int(original_sr),
        "original_channels": int(data.shape[1]),
        "selected_channel": channel,
        "source_subtype": info.subtype,
        "sample_rate": rate,
        "sample_count": len(mono),
        "duration_sec": len(mono) / rate,
        "pcm16_bytes": len(pcm_bytes),
    }
    return pcm_bytes, metadata


RTP_MODULO = 1 << 32


def label_event(message: Any) -> str:
    """PARTIAL replaces the open utterance text; COMMIT is the FINAL result (is_final)."""
    return "COMMIT" if message.is_final else "PARTIAL"


def reason_name(message: Any) -> str:
    try:
        return pb.CommitReason.Name(message.reason)
    except ValueError:
        return str(message.reason)


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    reference = args.truth.read_text(encoding="utf-8-sig").strip()
    if not reference:
        raise ValueError("Ground truth TXT is empty")
    pcm, audio_info = load_wav_as_pcm16(args.wav, args.rate, args.channel)
    rate = int(audio_info["sample_rate"])
    samples_per_chunk = int(rate * args.chunk_ms / 1000)
    if samples_per_chunk <= 0 or (rate * args.chunk_ms) % 1000:
        raise ValueError("chunk_ms must correspond to an integer sample count")
    bytes_per_chunk = samples_per_chunk * 2
    chunk_count = math.ceil(len(pcm) / bytes_per_chunk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "stream_events.jsonl"
    result_path = args.output_dir / "metrics.json"
    transcript_path = args.output_dir / "asr.txt"   # backward-compatible: punctuation output
    before_path = args.output_dir / "asr_before_punctuation.txt"
    after_path = args.output_dir / "asr_after_punctuation.txt"

    LOGGER.info("Input: %s | channels=%s | original_sr=%d | sent_sr=%d | %.3fs",
                args.wav, audio_info["original_channels"],
                audio_info["original_sample_rate"], rate, audio_info["duration_sec"])
    LOGGER.info("Streaming: %d chunks x %d ms; last chunk may be shorter; realtime=%s",
                chunk_count, args.chunk_ms, not args.fast)
    LOGGER.info("Server: %s | model already loaded by the engine", args.target)

    # VRAM comes from the local NVML driver, not the gRPC endpoint.
    # Auto skips remote endpoints instead of reporting the wrong machine's GPU.
    target_host = args.target.rsplit(":", 1)[0].strip("[]").lower()
    local_target = target_host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
    vram = None
    vram_unavailable = None
    if args.vram_mode == "off":
        vram_unavailable = "Disabled by --vram-mode off"
    elif args.vram_mode == "auto" and not local_target:
        vram_unavailable = (
            "Remote gRPC target: local NVML would read the CLIENT GPU, not the "
            "ASR server. Run this evaluation on the GPU server or use "
            "--vram-mode local explicitly if appropriate."
        )
    else:
        try:
            indices = None if args.gpu_indices is None else [int(x.strip()) for x in args.gpu_indices.split(",") if x.strip()]
            if indices is not None and not indices:
                raise ValueError("--gpu-indices must contain at least one index")
            vram = GpuVramMonitor(
                NvidiaNvmlBackend(indices), interval_ms=args.vram_interval_ms,
            )
            if not local_target:
                LOGGER.warning("Sampling GPU on client machine for a remote target; this may not be the ASR GPU")
        except Exception as exc:
            # NVML can throw a driver-specific exception, e.g. NVMLError_LibraryNotFound.
            if args.vram_mode == "required":
                raise RuntimeError(f"VRAM monitor required but unavailable: {exc}") from exc
            vram_unavailable = f"NVML unavailable on client host: {exc}"
    if vram_unavailable:
        LOGGER.warning("VRAM: %s", vram_unavailable)

    loop = asyncio.get_running_loop()
    sent_chunks = 0
    first_partial_ms = None
    first_commit_ms = None
    sent_start = None
    started_at = time.perf_counter()

    clock = int(args.rtp_clock_rate)

    def media_ms(rtp_timestamp: int) -> float:
        """RTP timestamp of a result -> milliseconds since the first sample sent."""
        return ((int(rtp_timestamp) - args.rtp_start) % RTP_MODULO) * 1000.0 / clock

    async def requests():
        nonlocal sent_chunks, sent_start
        sent_start = loop.time()
        sent_samples = 0
        for offset in range(0, len(pcm), bytes_per_chunk):
            audio = pcm[offset:offset + bytes_per_chunk]
            samples = len(audio) // 2
            last = offset + bytes_per_chunk >= len(pcm)
            if args.verbose_chunks:
                LOGGER.info("TX CHUNK seq=%d bytes=%d samples=%d media_t=%.3fs%s",
                            sent_chunks, len(audio), samples, (sent_samples + samples) / rate,
                            " end_of_stream" if last else "")
            yield pb.AudioChunk(
                session_id=args.call_id,
                stream_id=args.leg_id,
                pcm=audio,
                sample_rate=rate,
                channels=1,
                # RTP timestamp of the first sample of this chunk, source clock, modulo 2^32
                rtp_timestamp=(args.rtp_start + sent_samples * clock // rate) % RTP_MODULO,
                duration_ms=samples * 1000 // rate,
                end_of_stream=last,
                language=args.language,
                task="speech_to_text",
                rtp_clock_rate=clock,
                chunk_seq=sent_chunks,            # 0-based
            )
            sent_chunks += 1
            sent_samples += samples
            if not args.fast:
                sleep_for = sent_samples / rate - (loop.time() - sent_start)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
        LOGGER.info("TX end_of_stream sent with the last of %d chunks", sent_chunks)

    committed_texts: list[str] = []
    raw_commits: list[str] = []
    missing_raw_commit_ids: list[int] = []
    seen_commits: set[Any] = set()
    partials: dict[int, str] = {}
    received_messages = 0
    if vram is not None:
        await vram.start()
    try:
        with events_path.open("w", encoding="utf-8") as events_file:
            async with grpc.aio.insecure_channel(args.target, options=[
                ("grpc.max_receive_message_length", 8 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
            ]) as channel:
                stub = pb_grpc.SpeechStreamStub(channel)
                # Timeout covers the full call, not each chunk. For realtime mode
                # deadline must exceed WAV duration (plus inference overhead).
                rpc_timeout = args.timeout if args.timeout is not None else max(
                    120.0, audio_info["duration_sec"] * (3.0 if not args.fast else 1.5) + 60.0
                )
                async for response in stub.Recognize(requests(), timeout=rpc_timeout):
                    received_messages += 1
                    now_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    event = label_event(response)
                    cleaned_text = strip_asr_markers(response.text)
                    # raw_text is tag 22; an engine built from the gateway's speech.proto leaves it empty.
                    raw_original = response.raw_text
                    utterance = int(response.utterance_id)
                    reason = reason_name(response)
                    cleaned_raw = strip_asr_markers(raw_original)
                    # The protobuf has separate raw_text (Qwen output)
                    # and text (punctuation-restored output). Clean transport
                    # markers in BOTH, but never substitute post-punctuation
                    # text for missing raw_text: it would falsify raw WER.
                    record = {
                        "event": event,
                        "receive_elapsed_ms": now_ms,
                        "session_id": response.session_id,
                        "stream_id": response.stream_id,
                        "seq": int(response.seq),
                        "utterance_id": utterance,
                        "utterance_index": utterance,      # name used by earlier versions of this tool
                        "revision": response.revision,
                        "language": response.language,
                        "text": cleaned_text,
                        "raw_text": cleaned_raw,
                        "asr_before_punctuation": cleaned_raw,
                        "asr_after_punctuation": cleaned_text,
                        "raw_text_available": bool(cleaned_raw.strip()) or not bool(cleaned_text.strip()),
                        "text_original": response.text,
                        "raw_text_original": raw_original,
                        "reason": reason,
                        "ts_start": int(response.ts_start),
                        "ts_end": int(response.ts_end),
                        "start_ms": round(media_ms(response.ts_start), 1),
                        "end_ms": round(media_ms(response.ts_end), 1),
                    }
                    events_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    events_file.flush()
                    if event == "PARTIAL":
                        if first_partial_ms is None:
                            first_partial_ms = now_ms
                        partials[utterance] = cleaned_text
                        if not args.quiet_partials:
                            LOGGER.info("[PARTIAL] utterance=%s revision=%d | ASR(raw): %s | PUNCT: %s",
                                        utterance, response.revision,
                                        cleaned_raw or "[missing]", cleaned_text)
                    else:
                        # One FINAL per utterance_id; without utterance_id (tag 20) fall back to the unique seq.
                        key = utterance if utterance else ("seq", int(response.seq))
                        if key in seen_commits:
                            LOGGER.warning("Duplicate COMMIT ignored: %r", key)
                            continue
                        seen_commits.add(key)
                        if first_commit_ms is None:
                            first_commit_ms = now_ms
                        committed_texts.append(cleaned_text.strip())
                        raw_commits.append(cleaned_raw.strip())
                        if cleaned_text.strip() and not cleaned_raw.strip():
                            missing_raw_commit_ids.append(utterance)
                            LOGGER.warning("COMMIT utterance=%d has no raw_text; raw WER will be unavailable. "
                                           "The engine must fill RecognitionResult.raw_text (tag 22).",
                                           utterance)
                        partials.pop(utterance, None)
                        LOGGER.info("[COMMIT] seq=%s utt=%d reason=%s lang=%s %.0f-%.0fms (rtp %d-%d)",
                                    response.seq, utterance, reason, response.language or "-",
                                    record["start_ms"], record["end_ms"],
                                    response.ts_start, response.ts_end)
                        LOGGER.info("           ASR BEFORE PUNCT: %s", cleaned_raw or "[missing]")
                        LOGGER.info("           ASR AFTER  PUNCT: %s", cleaned_text or "[empty]")
    finally:
        if vram is not None:
            await vram.stop(post_ms=args.vram_post_ms)

    if vram is not None:
        vram.save_csv(args.output_dir / "vram_samples.csv")
        vram_metrics = vram.summary()
    else:
        vram_metrics = {"status": "unavailable", "reason": vram_unavailable}

    asr_text = " ".join(text for text in committed_texts if text)
    asr_raw = " ".join(text for text in raw_commits if text)
    elapsed_sec = time.perf_counter() - started_at
    scores_after = wer_details(reference, asr_text)
    raw_available = not missing_raw_commit_ids
    scores_before = wer_details(reference, asr_raw) if raw_available else None
    normalized_content_changed = (normalize_wer(asr_raw) != normalize_wer(asr_text)) if raw_available else None
    if normalized_content_changed:
        LOGGER.warning("Punctuation stage changed lexical token content, not just punctuation. "
                       "Compare the before/after transcripts for substitutions/deletions/insertions.")
    result = {
        "wav": audio_info,
        "ground_truth_file": str(args.truth.resolve()),
        "grpc_target": args.target,
        "call_id": args.call_id,
        "leg_id": args.leg_id,
        "chunk_ms": args.chunk_ms,
        "chunk_count_sent": sent_chunks,
        "realtime_pacing": not args.fast,
        "response_count": received_messages,
        "committed_utterances": len(committed_texts),
        "elapsed_sec": round(elapsed_sec, 3),
        "processing_rtf": round(elapsed_sec / audio_info["duration_sec"], 4),
        "first_partial_elapsed_ms": first_partial_ms,
        "first_commit_elapsed_ms": first_commit_ms,
        "ground_truth": reference,
        "asr_text": asr_text,  # backward-compatible alias of after punctuation
        "asr_raw_text": asr_raw,  # backward-compatible alias of before punctuation
        "asr_before_punctuation": asr_raw,
        "asr_after_punctuation": asr_text,
        "wer_before_punctuation": scores_before,
        "wer_after_punctuation": scores_after,
        "raw_text_available_for_all_commits": raw_available,
        "missing_raw_utterance_indices": missing_raw_commit_ids,
        "wer_normalization": "NFC, casefold, remove punctuation/symbols; Vietnamese diacritics retained; whitespace-separated syllable tokens",
        "lexical_content_changed_by_punctuation": normalized_content_changed,
        "metrics": scores_after,  # backward-compatible: formatted transcript score
        "vram": vram_metrics,
    }
    transcript_path.write_text(asr_text + "\n", encoding="utf-8")
    before_path.write_text(asr_raw + "\n", encoding="utf-8")
    after_path.write_text(asr_text + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + "=" * 65)
    print("GROUND TRUTH:\n" + reference)
    print("\nASR BEFORE PUNCTUATION:\n" + (asr_raw or "[empty / unavailable]"))
    print("\nASR AFTER PUNCTUATION:\n" + (asr_text or "[empty transcript]"))
    print("\nWER COMPARISON (ignores punctuation/case; Vietnamese syllable tokens):")
    for label, score in (("BEFORE punctuation", scores_before),
                         ("AFTER  punctuation", scores_after)):
        if score is None:
            print(f"  {label}: UNAVAILABLE (missing raw_text in commit utterance(s) "
                  f"{missing_raw_commit_ids})")
            continue
        print(f"  {label}: WER={score['wer_percent']:.2f}% | "
              f"S/D/I={score['substitutions']}/{score['deletions']}/{score['insertions']} | "
              f"Ref/Hyp={score['reference_token_count']}/{score['hypothesis_token_count']}")
    if normalized_content_changed is not None:
        print("  Lexical content changed by punctuation: " + str(normalized_content_changed))
        if not normalized_content_changed:
            print("  Note: equal WER is EXPECTED when punctuation only adds/removes marks.")
    print(f"  Commits      : {len(committed_texts)}")
    print(f"  Audio / Time : {audio_info['duration_sec']:.2f}s / {elapsed_sec:.2f}s")
    print(f"  RTF          : {result['processing_rtf']:.3f} (includes client pacing when realtime)")
    print("\nGPU VRAM (NVML, GPU-wide; model already loaded at baseline):")
    if vram_metrics["status"] in ("ok", "partial"):
        for gpu in vram_metrics["per_gpu"]:
            print(f"  GPU {gpu['gpu_index']} ({gpu['gpu_name']}):")
            print(f"    Before inference : {gpu['baseline_used_mib']:.1f} MiB")
            print(f"    Peak used        : {gpu['peak_used_mib']:.1f} MiB")
            print(f"    Peak - baseline  : {gpu['peak_delta_mib']:+.1f} MiB")
            print(f"    After inference  : {gpu['end_used_mib']:.1f} MiB")
            print(f"    End - baseline   : {gpu['end_delta_mib']:+.1f} MiB")
            if gpu['post_used_mib'] is not None:
                print(f"    After cool-down  : {gpu['post_used_mib']:.1f} MiB")
        print("  Samples: "+str(sum(x['samples'] for x in vram_metrics['per_gpu'])))
    else:
        print("  UNAVAILABLE: " + str(vram_metrics.get('reason')))
    print(f"\nSaved to: {args.output_dir.resolve()}")
    print("=" * 65)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Send a WAV to the ASR engine (speech.proto) as 80 ms chunks, measure WER.")
    p.add_argument("--wav", type=Path, required=True, help="Input .wav file")
    p.add_argument("--truth", type=Path, required=True, help="Ground truth .txt (UTF-8)")
    p.add_argument("--target", default="127.0.0.1:50051", help="gRPC host:port")
    p.add_argument("--chunk-ms", type=int, default=80, help="Audio chunk duration, default 80 ms")
    p.add_argument("--rate", type=int, choices=[8000, 16000], help="PCM rate sent to the engine; default 16000, as the gateway sends")
    p.add_argument("--channel", type=int, help="0-based channel index when WAV is stereo/multichannel; default downmix")
    p.add_argument("--language", default="vi", help="AudioChunk.language: vi | en | empty string for auto")
    p.add_argument("--call-id", default="wav-eval", help="AudioChunk.session_id")
    p.add_argument("--leg-id", default="wav-eval-a2b", help="AudioChunk.stream_id")
    p.add_argument("--rtp-clock-rate", type=int, default=8000, help="RTP clock of the source, Hz")
    p.add_argument("--rtp-start", type=int, default=0, help="RTP timestamp of the first sample (0 .. 2^32-1)")
    p.add_argument("--output-dir", type=Path, default=Path("asr_eval_output"))
    p.add_argument("--fast", action="store_true", help="Do not pace by wall clock; useful for throughput testing")
    p.add_argument("--quiet-partials", action="store_true", help="Log COMMIT only")
    p.add_argument("--verbose-chunks", action="store_true", help="Log every outgoing 80ms chunk")
    p.add_argument("--timeout", type=float, help="Entire gRPC RPC deadline, in seconds")
    p.add_argument("--vram-mode", choices=["auto", "local", "required", "off"], default="auto",
                   help="auto: monitor local GPU only when target is local; required: fail if NVML is absent")
    p.add_argument("--gpu-indices", help="NVML physical GPU index list, e.g. 0,1; default all NVIDIA GPUs")
    p.add_argument("--vram-interval-ms", type=int, default=100, help="GPU VRAM polling interval in ms")
    p.add_argument("--vram-post-ms", type=int, default=300, help="Time to wait after inference to sample retained VRAM")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if args.vram_interval_ms < 20 or args.vram_post_ms < 0:
        LOGGER.error("vram-interval-ms must be >=20 and vram-post-ms >=0")
        return 2
    if args.chunk_ms <= 0:
        LOGGER.error("chunk-ms must be positive")
        return 2
    if args.rtp_clock_rate <= 0 or not 0 <= args.rtp_start < RTP_MODULO:
        LOGGER.error("rtp-clock-rate must be positive and rtp-start within 0 .. 2^32-1")
        return 2
    try:
        asyncio.run(run_eval(args))
        return 0
    except grpc.aio.AioRpcError as exc:
        LOGGER.error("gRPC call failed: %s: %s", exc.code(), exc.details())
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
