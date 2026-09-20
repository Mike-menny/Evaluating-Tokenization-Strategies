from __future__ import annotations

import json

import pytest

from dafx26_demo.mlx_backend.benchmark import (
    VARIANTS,
    MlxBenchmarkRow,
    long_context_cells,
    long_context_verdict,
    median,
    p95,
    percentile,
    quartile_times,
    randomized_cells,
    realtime_ok,
    rtf,
    summarize_rows,
)


def test_median_and_p95() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert median(values) == 3.0
    assert p95(values) == pytest.approx(80.8)
    assert median([]) == 0.0
    assert median([2.0, 4.0]) == 3.0
    assert p95([7.0]) == 7.0
    assert p95([]) == 0.0
    assert percentile([1.0, 2.0, 3.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 100) == 3.0


def test_rtf_and_realtime_threshold() -> None:
    assert rtf(wall_sec=1.0, music_sec=2.0) == 0.5
    assert realtime_ok(0.5) is True
    assert realtime_ok(1.0) is True
    assert realtime_ok(1.01) is False
    assert rtf(wall_sec=2.0, music_sec=0.0) is None


def test_quartile_times_split_even_steps() -> None:
    times = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    assert quartile_times(times) == [2.0, 4.0, 6.0, 8.0]
    assert quartile_times([]) == [0.0, 0.0, 0.0, 0.0]


def test_randomized_cells_are_complete_and_shuffled() -> None:
    modes = ["note", "full"]
    lengths = [256, 512]
    a = randomized_cells(VARIANTS, modes, lengths, seed=1)
    b = randomized_cells(VARIANTS, modes, lengths, seed=2)
    expected = {(variant, mode, length) for variant in VARIANTS for mode in modes for length in lengths}
    assert {cell[:3] for cell in a} == expected
    assert len(a) == 20
    assert a != b


def test_row_schema_and_summary_markdown() -> None:
    row = MlxBenchmarkRow(
        variant="mlx_eager_dynamic",
        mode="note",
        max_tokens=256,
        trial="warm",
        decode_sec=1.5,
        tokens_per_sec=170.0,
        q1_sec=0.2,
        q2_sec=0.3,
        q3_sec=0.4,
        q4_sec=0.6,
        musical_duration_sec=3.0,
        rtf=0.5,
        peak_memory_bytes=1024,
        recompiled=False,
        ok=True,
        error=None,
        load_sec=0.4,
        warmup_sec=0.1,
    )
    payload = row.to_dict()
    assert payload["variant"] == "mlx_eager_dynamic"
    assert "tokens_per_sec" in payload
    markdown = summarize_rows(
        [
            row,
            MlxBenchmarkRow(
                variant="mlx_eager_dynamic",
                mode="note",
                max_tokens=256,
                trial="warm",
                decode_sec=1.7,
                tokens_per_sec=150.0,
                q1_sec=0.2,
                q2_sec=0.3,
                q3_sec=0.4,
                q4_sec=0.8,
                musical_duration_sec=3.0,
                rtf=0.57,
                peak_memory_bytes=2048,
                recompiled=False,
                ok=True,
                error=None,
                load_sec=0.0,
                warmup_sec=0.1,
            ),
        ]
    )
    assert "mlx_eager_dynamic" in markdown
    assert "median" in markdown.lower() or "decode median" in markdown.lower()
    assert "ASR convention" in markdown
    assert "music / wall" not in markdown
    json.dumps([row.to_dict()])
    failed = summarize_rows(
        [
            MlxBenchmarkRow(
                variant="mlx_eager_dynamic",
                mode="note",
                max_tokens=256,
                trial="warm",
                decode_sec=0.0,
                tokens_per_sec=None,
                q1_sec=0.0,
                q2_sec=0.0,
                q3_sec=0.0,
                q4_sec=0.0,
                musical_duration_sec=None,
                rtf=None,
                peak_memory_bytes=None,
                recompiled=False,
                ok=False,
                error="boom",
                load_sec=0.0,
                warmup_sec=0.0,
            )
        ]
    )
    assert "boom" in failed


def test_run_mlx_dry_run_lists_cells(monkeypatch) -> None:
    from dafx26_demo.mlx_backend.benchmark import dry_run_text

    text = dry_run_text(["pytorch_mps_eager"], ["note"], [256], seed=0)
    assert "pytorch_mps_eager" in text
    assert "note" in text
    assert "256" in text


def test_long_context_cells_exclude_fixed_buckets() -> None:
    cells = long_context_cells()
    assert set(cells) == {
        ("mlx_eager_dynamic", "note_velocity_pedal", 2048),
        ("mlx_eager_dynamic", "note_velocity_pedal", 4096),
        ("mlx_eager_block", "note_velocity_pedal", 2048),
        ("mlx_eager_block", "note_velocity_pedal", 4096),
    }


def _lc_rows(*, block_latency: list[float], dynamic_latency: list[float], block_memory_win: bool) -> list[MlxBenchmarkRow]:
    rows = []
    for length, block_t, dyn_t in zip((2048, 4096), block_latency, dynamic_latency, strict=True):
        for variant, decode, capacity, peak in (
            ("mlx_eager_block", block_t, 2048 if block_memory_win else 4096, 1000 if block_memory_win else 2000),
            ("mlx_eager_dynamic", dyn_t, 4096, 2000),
        ):
            rows.append(
                MlxBenchmarkRow(
                    variant=variant,
                    mode="note_velocity_pedal",
                    max_tokens=length,
                    trial="warm",
                    decode_sec=decode,
                    tokens_per_sec=100.0,
                    q1_sec=decode / 4,
                    q2_sec=decode / 4,
                    q3_sec=decode / 4,
                    q4_sec=decode / 4,
                    musical_duration_sec=1.0,
                    rtf=decode,
                    peak_memory_bytes=peak,
                    recompiled=False,
                    ok=True,
                    error=None,
                    load_sec=0.0,
                    warmup_sec=0.0,
                    cache_capacity=capacity,
                    peak_memory_delta_bytes=peak,
                )
            )
    return rows


def test_block_is_rejected_only_if_over_ten_percent_slower_at_both_lengths() -> None:
    assert (
        long_context_verdict(
            _lc_rows(block_latency=[1.101, 1.101], dynamic_latency=[1, 1], block_memory_win=False)
        )
        == "dynamic"
    )
    assert (
        long_context_verdict(
            _lc_rows(block_latency=[1.10, 1.101], dynamic_latency=[1, 1], block_memory_win=False)
        )
        == "block"
    )


def test_long_context_dry_run_lists_four_cells_and_smoke() -> None:
    from dafx26_demo.mlx_backend.benchmark import long_context_dry_run_text

    text = long_context_dry_run_text()
    assert "mlx_eager_bucket" not in text
    assert "mlx_compiled_bucket" not in text
    assert text.count("note_velocity_pedal") == 5
    assert "8192 smoke" in text
    assert "2048" in text
    assert "4096" in text
    assert (
        long_context_verdict(
            _lc_rows(block_latency=[1.2, 1.2], dynamic_latency=[1, 1], block_memory_win=True)
        )
        == "block"
    )
