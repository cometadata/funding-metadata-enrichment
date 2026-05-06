from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional


def _sync(device: Optional[str]) -> None:
    if not device or device == "cpu":
        return
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class PhaseAccumulator:
    def __init__(self, sync_device: Optional[str] = None) -> None:
        self.sync_device = sync_device
        self._per_doc: List[Dict[str, float]] = []
        self._current: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        _sync(self.sync_device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.sync_device)
            self._current[name] += (time.perf_counter() - t0) * 1000.0
            self._counts[name] += 1

    def add_ms(self, name: str, ms: float) -> None:
        self._current[name] += ms
        self._counts[name] += 1

    def commit_doc(self) -> None:
        self._per_doc.append(dict(self._current))
        self._current = defaultdict(float)

    def per_doc_records(self) -> List[Dict[str, float]]:
        return list(self._per_doc)

    def call_counts(self) -> Dict[str, int]:
        return dict(self._counts)


def wrap_callable(
    obj: Any, attr: str, phase: str, accumulator: PhaseAccumulator
) -> Callable[..., Any]:
    original = getattr(obj, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with accumulator.phase(phase):
            return original(*args, **kwargs)

    setattr(obj, attr, wrapped)
    return original


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(0.95 * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def summarize(
    per_doc_records: List[Dict[str, float]],
    skip_warmup: bool = True,
) -> Dict[str, Any]:
    records = per_doc_records[1:] if skip_warmup and len(per_doc_records) > 1 else per_doc_records
    if not records:
        return {"error": "no records"}

    phases: set[str] = set()
    for rec in records:
        phases.update(rec.keys())

    total_per_doc = [sum(rec.values()) for rec in records]
    mean_total_ms = statistics.mean(total_per_doc) if total_per_doc else 0.0

    per_phase: Dict[str, Dict[str, float]] = {}
    for p in sorted(phases):
        vals = [rec.get(p, 0.0) for rec in records]
        per_phase[p] = {
            "mean_ms": statistics.mean(vals),
            "median_ms": statistics.median(vals),
            "p95_ms": _p95(vals),
            "total_ms": sum(vals),
            "pct_of_wallclock": (statistics.mean(vals) / mean_total_ms * 100.0)
            if mean_total_ms > 0
            else 0.0,
        }

    return {
        "phases": per_phase,
        "num_docs": len(records),
        "mean_total_ms": mean_total_ms,
        "warmup_skipped": skip_warmup,
    }
