"""Sample NVIDIA GPU memory during WAV->gRPC inference.

NVML samples local GPU(s), not remote gRPC target. GPU-level memory includes
model weights, runtime allocations, other processes, and reserved memory.
"""
from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

MIB = 1024.0 * 1024.0


class NvidiaNvmlBackend:
    """Simple NVML interface with testable dependency injection."""

    def __init__(self, device_indices: list[int] | None = None):
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError(
                "Install nvidia-ml-py (offline wheel supported) to track GPU VRAM"
            ) from exc
        self.lib = pynvml
        self.lib.nvmlInit()
        try:
            count = int(self.lib.nvmlDeviceGetCount())
            selected = device_indices if device_indices is not None else list(range(count))
            if not selected:
                raise RuntimeError("No NVIDIA GPU detected by NVML")
            if any(i < 0 or i >= count for i in selected):
                raise ValueError(f"GPU indices {selected} not in physical range 0..{count - 1}")
            self.devices = []
            for i in selected:
                handle = self.lib.nvmlDeviceGetHandleByIndex(i)
                raw_name = self.lib.nvmlDeviceGetName(handle)
                name = raw_name.decode(errors='replace') if isinstance(raw_name, bytes) else str(raw_name)
                self.devices.append((i, name, handle))
        except Exception:
            self.lib.nvmlShutdown()
            raise

    def read(self) -> list[dict[str, Any]]:
        out = []
        for index, name, handle in self.devices:
            mem = self.lib.nvmlDeviceGetMemoryInfo(handle)
            out.append({"gpu_index": index, "name": name,
                        "used_mib": float(mem.used) / MIB,
                        "total_mib": float(mem.total) / MIB})
        return out

    def close(self) -> None:
        self.lib.nvmlShutdown()


@dataclass
class GpuVramMonitor:
    backend: Any
    interval_ms: int = 100
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self):
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._start = None
        self._failure = None
        self._closed = False

    def capture(self, phase: str) -> None:
        if self._closed:
            return
        millis = round((self.clock() - self._start) * 1000, 3)
        for device in self.backend.read():
            self.samples.append({"elapsed_ms": millis, "phase": phase,
                                 **device})

    async def start(self) -> None:
        self._start = self.clock()
        self.capture('baseline')
        self._task = asyncio.create_task(self._poll())

    async def _poll(self):
        try:
            while True:
                await asyncio.sleep(self.interval_ms / 1000)
                self.capture('during')
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._failure = str(exc)

    async def stop(self, post_ms: int = 0):
        if self._closed:
            return
        if self._task is not None:
            self._task.cancel()
            await self._task
        try:
            self.capture('end')
            if post_ms:
                await asyncio.sleep(post_ms / 1000)
                self.capture('post')
        except Exception as exc:
            self._failure = str(exc)
        finally:
            self._closed = True
            self.backend.close()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"status": "unavailable", "reason": self._failure or "no NVML samples"}
        grouped = {}
        for sample in self.samples:
            grouped.setdefault(sample['gpu_index'], []).append(sample)
        rows = []
        for index, arr in sorted(grouped.items()):
            baseline = next(x for x in arr if x['phase'] == 'baseline')
            after = next((x for x in reversed(arr) if x['phase'] == 'end'), arr[-1])
            post = next((x for x in reversed(arr) if x['phase'] == 'post'), None)
            during = [x for x in arr if x['phase'] in ('baseline','during','end')]
            peak = max(during, key=lambda x: x['used_mib'])
            used = [x['used_mib'] for x in during]
            rows.append({
                "gpu_index": index, "gpu_name": baseline['name'],
                "total_mib": round(baseline['total_mib'], 2),
                "baseline_used_mib": round(baseline['used_mib'], 2),
                "min_used_mib": round(min(used), 2),
                "avg_used_mib": round(sum(used) / len(used), 2),
                "peak_used_mib": round(peak['used_mib'], 2),
                "peak_at_elapsed_ms": peak['elapsed_ms'],
                "peak_delta_mib": round(peak['used_mib'] - baseline['used_mib'], 2),
                "end_used_mib": round(after['used_mib'], 2),
                "end_delta_mib": round(after['used_mib'] - baseline['used_mib'], 2),
                "post_used_mib": round(post['used_mib'], 2) if post else None,
                "post_delta_mib": round(post['used_mib'] - baseline['used_mib'], 2) if post else None,
                "samples": len(arr),
            })
        return {
            "status": "ok" if self._failure is None else "partial",
            "warning": self._failure,
            "source": "NVML on client host (must be GPU server for meaningful inference VRAM)",
            "attribution": "GPU-wide usage; includes preloaded model and other GPU processes",
            "sampling_interval_ms": self.interval_ms,
            "gpu_count": len(rows),
            "total_baseline_used_mib": round(sum(x['baseline_used_mib'] for x in rows), 2),
            "sum_of_gpu_peak_deltas_mib": round(sum(x['peak_delta_mib'] for x in rows), 2),
            "per_gpu": rows,
        }

    def save_csv(self, output_path: Path) -> None:
        with output_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'elapsed_ms', 'phase', 'gpu_index', 'name', 'used_mib', 'total_mib'
            ])
            writer.writeheader()
            writer.writerows(self.samples)
