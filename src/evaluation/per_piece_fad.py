#!/usr/bin/env python3
"""Per-file FAD analysis: each wav vs the reference distribution for its composer/genre."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from src.evaluation.fad import (
    _collect_matching_categories,
    _file_level_embeddings,
    _load_frechet_audio_distance,
    _maybe_debug_torch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _embd_statistics(fad, embds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    embds = np.asarray(embds, dtype=np.float64)
    if embds.ndim == 1:
        embds = embds[np.newaxis, :]
    if embds.shape[0] == 0:
        raise ValueError("empty embeddings")
    mu = np.mean(embds, axis=0)
    if embds.shape[0] == 1:
        sigma = np.eye(embds.shape[1], dtype=np.float64) * 1e-6
    else:
        sigma = np.cov(embds, rowvar=False)
    return mu, sigma


def _fad_from_embeddings(fad, ref_embs: np.ndarray, eval_embs: np.ndarray) -> float:
    mu_ref, sigma_ref = _embd_statistics(fad, ref_embs)
    mu_eval, sigma_eval = _embd_statistics(fad, eval_embs)
    return float(fad.calculate_frechet_distance(mu_ref, sigma_ref, mu_eval, sigma_eval))


def _fad_one_vs_pool(fad, pool_embs: np.ndarray, one_emb: np.ndarray) -> float:
    one_emb = np.asarray(one_emb, dtype=np.float32)
    if one_emb.ndim == 1:
        one_emb = one_emb[np.newaxis, :]
    return _fad_from_embeddings(fad, pool_embs, one_emb)


def _summarize_scores(scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "n": float(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _cache_key(composer: str, genre: str) -> str:
    return f"{composer}__{genre}".replace("/", "_")


def build_ref_embedding_cache(
    fad,
    ref_root: Path,
    groups: list[Path],
    cache_dir: Path,
) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    for rel in groups:
        composer, genre = rel.parent.name, rel.name
        key = _cache_key(composer, genre)
        cache_path = cache_dir / f"{key}.npz"
        ref_files = sorted((ref_root / rel).glob("*.wav"))
        if cache_path.is_file():
            data = np.load(cache_path, allow_pickle=True)
            ref_embs = data["embeddings"]
            names = data["names"].tolist()
        else:
            print(f"Cache miss: embedding {composer}/{genre} ({len(ref_files)} files)", flush=True)
            ref_embs = _file_level_embeddings(fad, ref_files)
            names = [p.name for p in ref_files]
            np.savez(cache_path, embeddings=ref_embs, names=np.array(names, dtype=object))
        if ref_embs.shape[0] < 2:
            print(f"Skip {composer}/{genre}: only {ref_embs.shape[0]} reference file(s)", flush=True)
            continue
        ref_scores = [_fad_one_vs_pool(fad, ref_embs, ref_embs[i]) for i in range(ref_embs.shape[0])]
        cache[key] = {
            "composer": composer,
            "genre": genre,
            "ref_files": ref_files,
            "ref_embs": ref_embs,
            "ref_names": names,
            "ref_scores": ref_scores,
            "ref_summary": _summarize_scores(ref_scores),
        }
    return cache


def analyze_group_cached(
    fad,
    composer: str,
    genre: str,
    run_name: str,
    ref_embs: np.ndarray,
    ref_files: list[Path],
    ref_scores: list[float],
    out_files: list[Path],
) -> tuple[list[dict], dict]:
    gen_rows: list[dict] = []
    gen_scores: list[float] = []
    for wav in sorted(out_files):
        one = _file_level_embeddings(fad, [wav])
        if one.shape[0] != 1:
            continue
        score = _fad_one_vs_pool(fad, ref_embs, one[0])
        gen_scores.append(score)
        gen_rows.append({
            "run": run_name,
            "composer": composer,
            "genre": genre,
            "side": "generated",
            "wav_name": wav.name,
            "per_piece_fad": score,
        })

    gen_summary = _summarize_scores(gen_scores)
    if gen_scores:
        out_embs = _file_level_embeddings(fad, out_files)
        gen_summary["group_pooled_fad"] = _fad_from_embeddings(fad, ref_embs, out_embs)

    for i, wav in enumerate(sorted(ref_files)):
        gen_rows.append({
            "run": run_name,
            "composer": composer,
            "genre": genre,
            "side": "reference",
            "wav_name": wav.name,
            "per_piece_fad": ref_scores[i],
        })
    return gen_rows, gen_summary


def _per_piece_worker_loop(task_queue, result_queue, gpu_id: int, cache_dir: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    fad = _load_frechet_audio_distance()(model_name="vggish", use_pca=False)
    cache_path = Path(cache_dir)
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            key = _cache_key(task["composer"], task["genre"])
            npz = np.load(cache_path / f"{key}.npz", allow_pickle=True)
            ref_embs = npz["embeddings"]
            ref_files = [Path(p) for p in task["ref_files"]]
            ref_scores = task["ref_scores"]
            rows, gen_summary = analyze_group_cached(
                fad=fad,
                composer=task["composer"],
                genre=task["genre"],
                run_name=task["run_name"],
                ref_embs=ref_embs,
                ref_files=ref_files,
                ref_scores=ref_scores,
                out_files=[Path(p) for p in task["out_files"]],
            )
            result_queue.put({
                "ok": True,
                "rows": rows,
                "summary": {
                    "run": task["run_name"],
                    "composer": task["composer"],
                    "genre": task["genre"],
                    **{f"gen_{k}": v for k, v in gen_summary.items()},
                    **{f"ref_{k}": v for k, v in task["ref_summary"].items()},
                },
            })
        except Exception as e:
            result_queue.put({
                "ok": False,
                "run": task["run_name"],
                "composer": task["composer"],
                "genre": task["genre"],
                "error": str(e),
            })


def run_per_piece_analysis(
    reference_path: str,
    runs: list[tuple[str, str]],
    out_dir: str,
    num_gpus: int = 8,
) -> None:
    import multiprocessing as mp

    _maybe_debug_torch()
    ref_root = Path(reference_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path / "ref_emb_cache"

    all_groups: set[Path] = set()
    tasks: list[dict] = []
    for run_name, output_path in runs:
        out_root = Path(output_path)
        if not out_root.is_dir():
            print(f"Skip {output_path}: not found")
            continue
        for rel in _collect_matching_categories(ref_root, out_root):
            ref_files = sorted((ref_root / rel).glob("*.wav"))
            out_files = sorted((out_root / rel).glob("*.wav"))
            if not ref_files or not out_files:
                continue
            all_groups.add(rel)
            tasks.append({
                "run_name": run_name,
                "composer": rel.parent.name,
                "genre": rel.name,
                "ref_files": [str(p) for p in ref_files],
                "out_files": [str(p) for p in out_files],
            })

    if not tasks:
        print("No per-piece FAD tasks.")
        return

    print("Building reference embedding cache...", flush=True)
    fad = _load_frechet_audio_distance()(model_name="vggish", use_pca=False)
    ref_cache = build_ref_embedding_cache(fad, ref_root, sorted(all_groups), cache_dir)

    for task in tasks:
        key = _cache_key(task["composer"], task["genre"])
        if key not in ref_cache:
            print(f"Skip task {task['run_name']} {task['composer']}/{task['genre']}: no ref cache", flush=True)
            continue
        entry = ref_cache[key]
        task["ref_scores"] = entry["ref_scores"]
        task["ref_summary"] = entry["ref_summary"]

    tasks = [t for t in tasks if "ref_scores" in t]
    if not tasks:
        print("No runnable per-piece FAD tasks after cache filter.")
        return

    try:
        import torch
        num_gpus = min(num_gpus, max(1, torch.cuda.device_count()))
    except Exception:
        num_gpus = 1

    print(f"Per-piece FAD: {len(tasks)} tasks on {num_gpus} GPU(s)", flush=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=_per_piece_worker_loop,
            args=(task_queue, result_queue, gpu_id, str(cache_dir)),
        )
        p.start()
        workers.append(p)

    for task in tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    done = 0
    for _ in range(len(tasks)):
        result = result_queue.get()
        done += 1
        if not result.get("ok"):
            print(
                f"[{done}/{len(tasks)}] Error {result.get('run')} "
                f"{result.get('composer')}/{result.get('genre')}: {result.get('error')}",
                flush=True,
            )
            continue
        all_rows.extend(result["rows"])
        summaries.append(result["summary"])
        if done % 10 == 0 or done == len(tasks):
            print(f"[{done}/{len(tasks)}] tasks done", flush=True)

    piece_csv = out_path / "per_piece_fad.csv"
    with piece_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "composer", "genre", "side", "wav_name", "per_piece_fad"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_csv = out_path / "per_piece_fad_group_summary.csv"
    if summaries:
        fieldnames = sorted({k for row in summaries for k in row.keys()})
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    print(f"Wrote {piece_csv} ({len(all_rows)} rows)")
    print(f"Wrote {summary_csv} ({len(summaries)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-piece FAD vs reference pool by composer/genre.")
    parser.add_argument(
        "--reference_path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "asap_test_set_wav"),
    )
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/inference/per_piece_fad"))
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Repeatable run_dir:wav_dir pair, e.g. "
        "--run outputs/inference/asap_note_velocity_ft/SESSION:outputs/inference/asap_note_velocity_ft/SESSION_wav",
    )
    args = parser.parse_args()

    runs: list[tuple[str, str]] = []
    for item in args.run:
        if ":" not in item:
            raise SystemExit(f"Invalid --run {item!r}; expected run_dir:wav_dir")
        run_dir, wav_dir = item.split(":", 1)
        runs.append((run_dir.strip(), wav_dir.strip()))
    if not runs:
        raise SystemExit(
            "Pass at least one --run run_dir:wav_dir "
            "(example: --run outputs/inference/asap_note_velocity_ft/2026-01-01_120000:"
            "outputs/inference/asap_note_velocity_ft/2026-01-01_120000_wav)"
        )

    run_per_piece_analysis(
        reference_path=args.reference_path,
        runs=runs,
        out_dir=args.out_dir,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
