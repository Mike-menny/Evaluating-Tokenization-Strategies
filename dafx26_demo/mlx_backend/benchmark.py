from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import platform
import random
from typing import Any, Iterable

from dafx26_demo.metrics import realtime_ok, rtf
from dafx26_demo.paths import MODEL_REVISION, UPSTREAM_REVISION

VARIANTS = (
    "pytorch_mps_eager",
    "mlx_eager_dynamic",
    "mlx_eager_block",
    "mlx_eager_bucket",
    "mlx_compiled_bucket",
)

DEFAULT_MODES = ("note", "note_velocity_pedal", "full")
DEFAULT_LENGTHS = (256, 512)
WARM_TRIALS = 7


@dataclass
class MlxBenchmarkRow:
    variant: str
    mode: str
    max_tokens: int
    trial: str
    decode_sec: float
    tokens_per_sec: float | None
    q1_sec: float
    q2_sec: float
    q3_sec: float
    q4_sec: float
    musical_duration_sec: float | None
    rtf: float | None
    peak_memory_bytes: int | None
    recompiled: bool
    ok: bool
    error: str | None
    load_sec: float
    warmup_sec: float
    cache_capacity: int | None = None
    cache_offset: int | None = None
    baseline_peak_memory_bytes: int | None = None
    peak_memory_delta_bytes: int | None = None
    decode_q1_segment_sec: float | None = None
    decode_q2_segment_sec: float | None = None
    decode_q3_segment_sec: float | None = None
    decode_q4_segment_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def median(values: Iterable[float]) -> float:
    items = sorted(float(v) for v in values)
    if not items:
        return 0.0
    mid = len(items) // 2
    if len(items) % 2:
        return items[mid]
    return (items[mid - 1] + items[mid]) / 2.0


def percentile(values: Iterable[float], p: float) -> float:
    items = sorted(float(v) for v in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    rank = (len(items) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return items[lo]
    weight = rank - lo
    return items[lo] * (1.0 - weight) + items[hi] * weight


def p95(values: Iterable[float]) -> float:
    return percentile(values, 95)


def quartile_times(step_times: list[float]) -> list[float]:
    if not step_times:
        return [0.0, 0.0, 0.0, 0.0]
    n = len(step_times)
    bounds = [0, n // 4, n // 2, (3 * n) // 4, n]
    return [float(sum(step_times[bounds[i] : bounds[i + 1]])) for i in range(4)]


def randomized_cells(
    variants: Iterable[str],
    modes: Iterable[str],
    lengths: Iterable[int],
    *,
    seed: int,
) -> list[tuple[str, str, int]]:
    cells = [(str(variant), str(mode), int(length)) for variant in variants for mode in modes for length in lengths]
    rng = random.Random(seed)
    rng.shuffle(cells)
    return cells


def dry_run_text(
    variants: Iterable[str],
    modes: Iterable[str],
    lengths: Iterable[int],
    *,
    seed: int = 0,
) -> str:
    lines = ["dry-run cells:"]
    for variant, mode, length in randomized_cells(variants, modes, lengths, seed=seed):
        lines.append(f"- {variant} {mode} {length}")
    return "\n".join(lines) + "\n"


def summarize_rows(rows: list[MlxBenchmarkRow]) -> str:
    lines = [
        "# Benchmark mlx-m3ultra",
        "",
        f"Host: {platform.node()} ({platform.platform()})",
        f"Model revision: `{MODEL_REVISION}`",
        f"Code revision: `{UPSTREAM_REVISION}`",
        "",
        "Fixed-token greedy replay. RTF is processing wall time / musical duration "
        "(ASR convention; realtime at <= 1).",
        "Warm statistics use seven trials after separate load and compile warmup.",
        "",
        "| variant | mode | tokens | n | decode median s | decode p95 s | tok/s median | RTF median | realtime | q1/q2/q3/q4 median s | mem B | recompiled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    groups: dict[tuple[str, str, int], list[MlxBenchmarkRow]] = {}
    for row in rows:
        if row.trial != "warm" or not row.ok:
            continue
        groups.setdefault((row.variant, row.mode, row.max_tokens), []).append(row)
    for key in sorted(groups):
        items = groups[key]
        decode = [row.decode_sec for row in items]
        tps = [row.tokens_per_sec or 0.0 for row in items]
        rtfs = [row.rtf for row in items if row.rtf is not None]
        q1 = median(row.q1_sec for row in items)
        q2 = median(row.q2_sec for row in items)
        q3 = median(row.q3_sec for row in items)
        q4 = median(row.q4_sec for row in items)
        rtf_med = median(rtfs) if rtfs else None
        mem = max((row.peak_memory_bytes or 0) for row in items)
        recompiled = any(row.recompiled for row in items)
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(items)} | "
            f"{median(decode):.3f} | {p95(decode):.3f} | {median(tps):.2f} | "
            f"{(rtf_med if rtf_med is not None else 0):.3f} | "
            f"{'yes' if realtime_ok(rtf_med) else 'no'} | "
            f"{q1:.3f}/{q2:.3f}/{q3:.3f}/{q4:.3f} | {mem} | {recompiled} |"
        )
    errors = [row for row in rows if not row.ok]
    if errors:
        lines.extend(["", "Errors:", ""])
        for row in errors:
            lines.append(f"- {row.variant} {row.mode} {row.max_tokens}: {row.error}")
    return "\n".join(lines) + "\n"


LONG_CONTEXT_VARIANTS = ("mlx_eager_dynamic", "mlx_eager_block")
LONG_CONTEXT_MODE = "note_velocity_pedal"
LONG_CONTEXT_LENGTHS = (2048, 4096)
LONG_CONTEXT_SMOKE_POSITIONS = 8192
LONG_CONTEXT_BLOCK_SIZE = 256


def long_context_cells() -> list[tuple[str, str, int]]:
    return [
        (variant, LONG_CONTEXT_MODE, length)
        for length in LONG_CONTEXT_LENGTHS
        for variant in LONG_CONTEXT_VARIANTS
    ]


def block_smoke_expectations(total_positions: int = LONG_CONTEXT_SMOKE_POSITIONS) -> dict[str, int]:
    return {
        "forwards": total_positions - 1,
        "final_offset": total_positions - 1,
        "capacity": total_positions,
        "max_position": total_positions - 2,
    }


def analytical_block_capacity(offset: int, *, block_size: int = LONG_CONTEXT_BLOCK_SIZE) -> int:
    if offset <= 0:
        return block_size
    blocks = (offset + block_size - 1) // block_size
    return max(block_size, blocks * block_size)


def long_context_dry_run_text() -> str:
    lines = ["dry-run cells:"]
    for variant, mode, length in long_context_cells():
        lines.append(f"- {variant} {mode} {length}")
    lines.append(
        f"- mlx_eager_block {LONG_CONTEXT_MODE} {LONG_CONTEXT_SMOKE_POSITIONS} smoke"
    )
    return "\n".join(lines) + "\n"


def _median_decode(rows: list[MlxBenchmarkRow], *, variant: str, length: int) -> float:
    values = [
        row.decode_sec
        for row in rows
        if row.ok and row.variant == variant and row.max_tokens == length and row.trial == "warm"
    ]
    if not values:
        raise ValueError(f"no warm {variant} rows at {length}")
    return median(values)


def _memory_win(rows: list[MlxBenchmarkRow]) -> bool:
    block_caps = [row.cache_capacity for row in rows if row.variant == "mlx_eager_block" and row.cache_capacity]
    dyn_caps = [row.cache_capacity for row in rows if row.variant == "mlx_eager_dynamic" and row.cache_capacity]
    if block_caps and dyn_caps and min(block_caps) < min(dyn_caps):
        return True
    block_peaks = [
        row.peak_memory_delta_bytes
        for row in rows
        if row.variant == "mlx_eager_block" and row.peak_memory_delta_bytes is not None
    ]
    dyn_peaks = [
        row.peak_memory_delta_bytes
        for row in rows
        if row.variant == "mlx_eager_dynamic" and row.peak_memory_delta_bytes is not None
    ]
    if not block_peaks or not dyn_peaks:
        return False
    block_med = median(block_peaks)
    dyn_med = median(dyn_peaks)
    return dyn_med > 0 and block_med <= dyn_med * 0.95


def long_context_verdict(rows: list[MlxBenchmarkRow]) -> str:
    errors = [row for row in rows if not row.ok]
    if errors:
        raise ValueError(f"long-context rows have errors: {errors[0].error}")
    slower_both = True
    for length in LONG_CONTEXT_LENGTHS:
        block = _median_decode(rows, variant="mlx_eager_block", length=length)
        dynamic = _median_decode(rows, variant="mlx_eager_dynamic", length=length)
        if not (block > dynamic * 1.10):
            slower_both = False
    if slower_both and not _memory_win(rows):
        return "dynamic"
    return "block"
