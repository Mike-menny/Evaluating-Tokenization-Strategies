import os
import csv
import tempfile
import sys
import types
import random
from pathlib import Path
import numpy as np

# VGGish loads two files; both must be local or it will try to download (SSL fails in containers).
_FAD_CKPT_NAMES = ("vggish-10086976.pth", "vggish_pca_params-970ea276.pth")

def _fad_checkpoints_ready(ckpt_dir: Path) -> bool:
    return ckpt_dir.is_dir() and all((ckpt_dir / n).is_file() for n in _FAD_CKPT_NAMES)

if "TORCH_HOME" not in os.environ:
    _torch_home = None
    _project_root = Path(__file__).resolve().parent.parent.parent
    _proj_ckpt_dir = _project_root / "torch_hub_checkpoints"
    _app_ckpt_dir = Path("/app/.cache/torch/hub/checkpoints")
    if _fad_checkpoints_ready(_proj_ckpt_dir):
        _cache = _project_root / ".cache" / "torch" / "hub" / "checkpoints"
        _cache.mkdir(parents=True, exist_ok=True)
        for _name in _FAD_CKPT_NAMES:
            _link = _cache / _name
            if not _link.exists():
                _link.symlink_to((_proj_ckpt_dir / _name).resolve())
        _torch_home = str(_project_root / ".cache" / "torch")
    elif _fad_checkpoints_ready(_app_ckpt_dir):
        _torch_home = "/app/.cache/torch"
    if _torch_home:
        os.environ["TORCH_HOME"] = _torch_home

def _ensure_fad_import_path() -> None:
    """
    Make frechet_audio_distance importable when running with plain python3 -m.
    """
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        # Workspace-local venv (common when dependencies are installed into .venv)
        project_root / ".venv" / "lib" / f"python{py_ver}" / "site-packages",
        # Container default for uv-managed virtual env
        Path("/root/.venv/lib") / f"python{py_ver}" / "site-packages",
    ]
    required = ("frechet_audio_distance", "frechet_audio_distance.py")
    for p in candidates:
        p_str = str(p)
        if not p.is_dir():
            continue
        # Only add a candidate if it likely contains the package we need,
        # to reduce the chance of accidentally shadowing other deps (e.g. torch).
        has_marker = any((p / r).exists() for r in required)
        if has_marker and p_str not in sys.path:
            # Append (not prepend) to avoid shadowing the current environment's packages,
            # especially `torch` which must match CUDA kernels for GPU architectures.
            sys.path.append(p_str)


def _maybe_debug_torch() -> None:
    if os.environ.get("FAD_DEBUG_TORCH", "") not in ("1", "true", "TRUE", "yes", "YES"):
        return
    try:
        import torch

        print(
            "[FAD_DEBUG_TORCH] torch_version=" + str(torch.__version__)
            + " torch_cuda=" + str(torch.version.cuda)
            + " cuda_available=" + str(torch.cuda.is_available())
            + " arch_list=" + str(torch.cuda.get_arch_list())
            + " torch_file=" + str(torch.__file__),
            flush=True,
        )
    except Exception as e:
        print("[FAD_DEBUG_TORCH] failed to import/inspect torch: " + str(e), flush=True)


_FrechetAudioDistance = None


def _load_frechet_audio_distance():
    """Import FrechetAudioDistance without pulling laion_clap (HF roberta download).

    frechet_audio_distance imports laion_clap at module load time, which triggers
    RobertaTokenizer.from_pretrained() even when model_name is vggish. Stub it out.
    """
    global _FrechetAudioDistance
    if _FrechetAudioDistance is not None:
        return _FrechetAudioDistance
    if "laion_clap" not in sys.modules:
        sys.modules["laion_clap"] = types.ModuleType("laion_clap")
    try:
        from frechet_audio_distance.fad import FrechetAudioDistance
    except ModuleNotFoundError:
        _ensure_fad_import_path()
        from frechet_audio_distance.fad import FrechetAudioDistance
    _FrechetAudioDistance = FrechetAudioDistance
    return _FrechetAudioDistance


def compute_fad_score(
    background_path: str,
    eval_path: str,
    model_name: str = "vggish",
    use_pca: bool = False,
    use_activation: bool = False,
) -> float:
    """
    Compute Frechet Audio Distance (FAD) score between background and eval audio sets.

    Args:
        background_path: Path to background/reference audio directory or file.
        eval_path: Path to evaluation audio directory or file.
        model_name: FAD model name (default: "vggish"). Weights are read from TORCH_HOME if present (e.g. pre-copied in Docker); otherwise the library may download on first use.
        use_pca: Whether to use PCA in the FAD model.
        use_activation: Whether to use activation in the FAD model.

    Returns:
        FAD score (float).
    """
    FrechetAudioDistance = _load_frechet_audio_distance()
    fad = FrechetAudioDistance(
        model_name=model_name,
        use_pca=use_pca,
        use_activation=use_activation,
    )
    return fad.score(background_path, eval_path)


def _symlink_wavs_into_dir(file_paths: list[Path], target_dir: Path) -> None:
    """Create symlinks for each wav file in target_dir with unique names."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(file_paths):
        # Unique name to avoid collisions when merging from multiple dirs
        link_name = target_dir / f"{i:06d}_{src.name}"
        os.symlink(src.resolve(), link_name, target_is_directory=False)


def _fad_score_with_model(fad, ref_files: list[Path], out_files: list[Path]) -> float:
    """Run FAD.score on symlink dirs built from file lists."""
    import shutil
    import tempfile

    tmp_dir_path = Path(tempfile.mkdtemp(prefix="fad_"))
    try:
        tmp_ref = tmp_dir_path / "ref"
        tmp_out = tmp_dir_path / "out"
        _symlink_wavs_into_dir(ref_files, tmp_ref)
        _symlink_wavs_into_dir(out_files, tmp_out)
        return fad.score(str(tmp_ref), str(tmp_out))
    finally:
        shutil.rmtree(tmp_dir_path, ignore_errors=True)


def _load_audio_task_fn():
    """Import frechet_audio_distance audio loader with local-path fallback."""
    try:
        from frechet_audio_distance.utils import load_audio_task
    except ModuleNotFoundError:
        _ensure_fad_import_path()
        from frechet_audio_distance.utils import load_audio_task
    return load_audio_task


def _file_level_embeddings(
    fad,
    wav_files: list[Path],
    dtype: str = "float32",
) -> np.ndarray:
    """
    Build one embedding vector per file (equal weight per file, not per frame).
    """
    load_audio_task = _load_audio_task_fn()
    file_embs: list[np.ndarray] = []
    for wav in wav_files:
        audio = load_audio_task(str(wav), fad.sample_rate, fad.channels, dtype)
        emb = fad.get_embeddings([audio], sr=fad.sample_rate)
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        if emb.size == 0:
            continue
        file_embs.append(np.mean(emb, axis=0))
    if not file_embs:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(file_embs, axis=0).astype(np.float32, copy=False)


def _fair_fad_from_file_embeddings(fad, ref_embs: np.ndarray, out_embs: np.ndarray) -> float:
    mu_ref, sigma_ref = fad.calculate_embd_statistics(ref_embs)
    mu_out, sigma_out = fad.calculate_embd_statistics(out_embs)
    return float(fad.calculate_frechet_distance(mu_ref, sigma_ref, mu_out, sigma_out))


def _bootstrap_fair_fad_with_model(
    fad,
    ref_files: list[Path],
    out_files: list[Path],
    sample_size: int,
    num_bootstrap: int,
    seed: int,
    dtype: str = "float32",
) -> dict[str, float | int]:
    """
    Estimate a fair FAD by equal-size, per-file bootstrap sampling.
    """
    if sample_size < 2:
        raise ValueError("sample_size must be >= 2 for covariance estimation")
    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be >= 1")

    ref_file_embs = _file_level_embeddings(fad, ref_files, dtype=dtype)
    out_file_embs = _file_level_embeddings(fad, out_files, dtype=dtype)
    n_ref = int(ref_file_embs.shape[0])
    n_out = int(out_file_embs.shape[0])
    if n_ref < sample_size or n_out < sample_size:
        raise ValueError(
            f"Not enough files for sample_size={sample_size}: ref={n_ref}, out={n_out}"
        )

    rng = random.Random(seed)
    scores: list[float] = []
    ref_idx_all = list(range(n_ref))
    out_idx_all = list(range(n_out))
    for _ in range(num_bootstrap):
        ref_idx = rng.sample(ref_idx_all, sample_size)
        out_idx = rng.sample(out_idx_all, sample_size)
        score = _fair_fad_from_file_embeddings(
            fad,
            ref_file_embs[ref_idx, :],
            out_file_embs[out_idx, :],
        )
        scores.append(score)

    scores_arr = np.asarray(scores, dtype=np.float64)
    return {
        "fad_score_mean": float(scores_arr.mean()),
        "fad_score_std": float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else 0.0,
        "fad_score_min": float(scores_arr.min()),
        "fad_score_max": float(scores_arr.max()),
        "sample_size": sample_size,
        "num_bootstrap": num_bootstrap,
        "ref_n_files": n_ref,
        "out_n_files": n_out,
    }


def _resolve_fair_sample_size(
    ref_n: int,
    out_n: int,
    sample_size: int | None,
    max_sample_size: int | None,
) -> int:
    fair_n = min(ref_n, out_n)
    if sample_size is not None:
        fair_n = min(fair_n, sample_size)
    if max_sample_size is not None:
        fair_n = min(fair_n, max_sample_size)
    return fair_n


def _build_fair_fad_tasks(
    reference_path: str,
    output_path: str,
    run_key: str,
    sample_size: int | None = None,
    max_sample_size: int | None = 20,
    include_overall: bool = True,
) -> list[dict]:
    tasks, _, _ = _build_fad_tasks(reference_path, output_path, run_key)
    if not include_overall:
        tasks = [
            t for t in tasks
            if not (t["composer"] == "OVERALL" and t["genre"] == "OVERALL")
        ]
    fair_tasks: list[dict] = []
    for task in tasks:
        fair_n = _resolve_fair_sample_size(
            len(task["ref_files"]),
            len(task["out_files"]),
            sample_size,
            max_sample_size,
        )
        fair_tasks.append({**task, "fair_sample_size": fair_n})
    return fair_tasks


def _fair_fad_worker_loop(task_queue, result_queue, gpu_id: int) -> None:
    """One persistent worker per GPU: load VGGish once, process fair-FAD tasks."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    FrechetAudioDistance = _load_frechet_audio_distance()
    fad = FrechetAudioDistance(model_name="vggish", use_pca=False)
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            stats = _bootstrap_fair_fad_with_model(
                fad=fad,
                ref_files=[Path(p) for p in task["ref_files"]],
                out_files=[Path(p) for p in task["out_files"]],
                sample_size=task["fair_sample_size"],
                num_bootstrap=task["num_bootstrap"],
                seed=task["seed"],
                dtype=task.get("dtype", "float32"),
            )
            row = {
                "run_key": task["run_key"],
                "composer": task["composer"],
                "genre": task["genre"],
                "fad_score": stats["fad_score_mean"],
                "fad_score_std": stats["fad_score_std"],
                "fad_score_min": stats["fad_score_min"],
                "fad_score_max": stats["fad_score_max"],
                "sample_size": stats["sample_size"],
                "num_bootstrap": stats["num_bootstrap"],
                "ref_n_files": stats["ref_n_files"],
                "out_n_files": stats["out_n_files"],
            }
        except Exception as e:
            row = {
                "run_key": task["run_key"],
                "composer": task["composer"],
                "genre": task["genre"],
                "fad_score": f"Error: {e}",
            }
        result_queue.put(row)


def _fad_worker_loop(task_queue, result_queue, gpu_id: int) -> None:
    """One persistent worker per GPU: load VGGish once, process tasks from queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    FrechetAudioDistance = _load_frechet_audio_distance()
    fad = FrechetAudioDistance(model_name="vggish", use_pca=False)
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            ref_files = [Path(p) for p in task["ref_files"]]
            out_files = [Path(p) for p in task["out_files"]]
            score = _fad_score_with_model(fad, ref_files, out_files)
            fad_score: float | str = score
        except Exception as e:
            fad_score = f"Error: {e}"
        result_queue.put({
            "run_key": task["run_key"],
            "composer": task["composer"],
            "genre": task["genre"],
            "fad_score": fad_score,
        })


def _collect_matching_categories(ref_root: Path, out_root: Path) -> list[Path]:
    matching: list[Path] = []
    for comp_dir in ref_root.iterdir():
        if not comp_dir.is_dir():
            continue
        for genre_dir in comp_dir.iterdir():
            if not genre_dir.is_dir():
                continue
            rel_path = Path(comp_dir.name) / genre_dir.name
            if (out_root / rel_path).is_dir():
                matching.append(rel_path)
            else:
                print(f"Skipping {rel_path}: not found in output path.")
    return matching


def _build_fad_tasks(
    reference_path: str,
    output_path: str,
    run_key: str,
) -> tuple[list[dict], list[Path], list[Path]]:
    """Build per-category and overall FAD tasks for one inference run."""
    ref_root = Path(reference_path)
    out_root = Path(output_path)
    if not ref_root.exists():
        raise FileNotFoundError(f"Reference path does not exist: {reference_path}")

    tasks: list[dict] = []
    all_ref_files: list[Path] = []
    all_out_files: list[Path] = []

    for rel in _collect_matching_categories(ref_root, out_root):
        ref_files = sorted((ref_root / rel).glob("*.wav"))
        out_files = sorted((out_root / rel).glob("*.wav"))
        if not ref_files or not out_files:
            print(f"Skipping {rel}: no wav files found in one or both directories.")
            continue
        all_ref_files.extend(ref_files)
        all_out_files.extend(out_files)
        tasks.append({
            "run_key": run_key,
            "composer": rel.parent.name,
            "genre": rel.name,
            "ref_files": [str(p) for p in ref_files],
            "out_files": [str(p) for p in out_files],
        })

    if all_ref_files and all_out_files:
        tasks.append({
            "run_key": run_key,
            "composer": "OVERALL",
            "genre": "OVERALL",
            "ref_files": [str(p) for p in all_ref_files],
            "out_files": [str(p) for p in all_out_files],
        })
    return tasks, all_ref_files, all_out_files


def _write_fad_csv(csv_path: str, results: list[dict]) -> None:
    def _sort_key(row: dict) -> tuple:
        if row["composer"] == "OVERALL":
            return (1, "", "")
        return (0, row["composer"], row["genre"])

    results = sorted(results, key=_sort_key)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["composer", "genre", "fad_score"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "composer": row["composer"],
                "genre": row["genre"],
                "fad_score": row["fad_score"],
            })


def _write_fair_fad_csv(csv_path: str, results: list[dict]) -> None:
    def _sort_key(row: dict) -> tuple:
        if row["composer"] == "OVERALL":
            return (1, "", "")
        return (0, row["composer"], row["genre"])

    fieldnames = [
        "composer",
        "genre",
        "fad_score",
        "fad_score_std",
        "fad_score_min",
        "fad_score_max",
        "sample_size",
        "num_bootstrap",
        "ref_n_files",
        "out_n_files",
    ]
    results = sorted(results, key=_sort_key)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "composer": row["composer"],
                "genre": row["genre"],
                "fad_score": row["fad_score"],
                "fad_score_std": row.get("fad_score_std"),
                "fad_score_min": row.get("fad_score_min"),
                "fad_score_max": row.get("fad_score_max"),
                "sample_size": row.get("sample_size"),
                "num_bootstrap": row.get("num_bootstrap"),
                "ref_n_files": row.get("ref_n_files"),
                "out_n_files": row.get("out_n_files"),
            })


def compute_fad_by_category_parallel(
    reference_path: str,
    output_path: str,
    csv_path: str,
    model_name: str = "vggish",
    num_gpus: int = 8,
) -> None:
    """
    Like compute_fad_by_category, but distributes categories across GPUs.

    Each GPU runs one persistent worker that loads VGGish once and processes
    tasks from a shared queue (one composer/genre pair per task).
    """
    import multiprocessing as mp

    _maybe_debug_torch()
    if model_name != "vggish":
        raise ValueError("Parallel FAD only supports model_name='vggish'")

    run_key = str(csv_path)
    tasks, _, _ = _build_fad_tasks(reference_path, output_path, run_key)
    if not tasks:
        print(f"No FAD tasks for {output_path}")
        return

    try:
        import torch
        num_gpus = min(num_gpus, max(1, torch.cuda.device_count()))
    except Exception:
        num_gpus = 1

    print(f"FAD parallel: {len(tasks)} tasks on {num_gpus} GPU(s) -> {csv_path}", flush=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=_fad_worker_loop,
            args=(task_queue, result_queue, gpu_id),
        )
        p.start()
        workers.append(p)

    for task in tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    results = [result_queue.get() for _ in range(len(tasks))]
    for p in workers:
        p.join()

    _write_fad_csv(csv_path, results)
    print(f"Successfully wrote FAD scores to {csv_path}", flush=True)


def compute_fair_fad_runs_parallel(
    runs: list[tuple[str, str, str]],
    reference_path: str,
    num_gpus: int = 8,
    model_name: str = "vggish",
    sample_size: int | None = None,
    max_sample_size: int | None = 20,
    num_bootstrap: int = 50,
    seed: int = 3407,
    include_overall: bool = True,
    dtype: str = "float32",
) -> None:
    """
    Compute fair FAD for multiple inference runs, sharing one GPU worker pool.

    Args:
        runs: list of (run_key, output_wav_path, csv_path).
        reference_path: reference WAV root ({composer}/{genre}/*.wav).
        num_gpus: number of GPUs to use in parallel.
    """
    import multiprocessing as mp

    _maybe_debug_torch()
    if model_name != "vggish":
        raise ValueError("Parallel fair FAD only supports model_name='vggish'")

    all_tasks: list[dict] = []
    csv_by_run: dict[str, str] = {}
    for run_key, output_path, csv_path in runs:
        if not Path(output_path).is_dir():
            print(f"Skip {output_path}: directory not found")
            continue
        tasks = _build_fair_fad_tasks(
            reference_path,
            output_path,
            run_key,
            sample_size=sample_size,
            max_sample_size=max_sample_size,
            include_overall=include_overall,
        )
        if not tasks:
            print(f"No fair FAD tasks for {output_path}")
            continue
        for task in tasks:
            task["num_bootstrap"] = num_bootstrap
            task["seed"] = seed
            task["dtype"] = dtype
        all_tasks.extend(tasks)
        csv_by_run[run_key] = csv_path

    if not all_tasks:
        print("No fair FAD tasks to run.")
        return

    try:
        import torch
        num_gpus = min(num_gpus, max(1, torch.cuda.device_count()))
    except Exception:
        num_gpus = 1

    print(
        f"Fair FAD parallel: {len(all_tasks)} tasks across {len(csv_by_run)} run(s) "
        f"on {num_gpus} GPU(s)",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=_fair_fad_worker_loop,
            args=(task_queue, result_queue, gpu_id),
        )
        p.start()
        workers.append(p)

    for task in all_tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    results_by_run: dict[str, list[dict]] = {k: [] for k in csv_by_run}
    for _ in range(len(all_tasks)):
        row = result_queue.get()
        results_by_run[row["run_key"]].append(row)

    for p in workers:
        p.join()

    for run_key, csv_path in csv_by_run.items():
        _write_fair_fad_csv(csv_path, results_by_run[run_key])
        print(f"Successfully wrote fair FAD scores to {csv_path}", flush=True)


def compute_fair_fad_by_category_parallel(
    reference_path: str,
    output_path: str,
    csv_path: str,
    model_name: str = "vggish",
    num_gpus: int = 8,
    sample_size: int | None = None,
    max_sample_size: int | None = 20,
    num_bootstrap: int = 50,
    seed: int = 3407,
    include_overall: bool = True,
    dtype: str = "float32",
) -> None:
    """Like compute_fair_fad_by_category, but distributes categories across GPUs."""
    import multiprocessing as mp

    _maybe_debug_torch()
    if model_name != "vggish":
        raise ValueError("Parallel fair FAD only supports model_name='vggish'")

    run_key = str(csv_path)
    tasks = _build_fair_fad_tasks(
        reference_path,
        output_path,
        run_key,
        sample_size=sample_size,
        max_sample_size=max_sample_size,
        include_overall=include_overall,
    )
    if not tasks:
        print(f"No fair FAD tasks for {output_path}")
        return

    for task in tasks:
        task["num_bootstrap"] = num_bootstrap
        task["seed"] = seed
        task["dtype"] = dtype

    try:
        import torch
        num_gpus = min(num_gpus, max(1, torch.cuda.device_count()))
    except Exception:
        num_gpus = 1

    print(
        f"Fair FAD parallel: {len(tasks)} tasks on {num_gpus} GPU(s) -> {csv_path}",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=_fair_fad_worker_loop,
            args=(task_queue, result_queue, gpu_id),
        )
        p.start()
        workers.append(p)

    for task in tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    results = [result_queue.get() for _ in range(len(tasks))]
    for p in workers:
        p.join()

    _write_fair_fad_csv(csv_path, results)
    print(f"Successfully wrote fair FAD scores to {csv_path}", flush=True)


def compute_fad_runs_parallel(
    runs: list[tuple[str, str, str]],
    reference_path: str,
    num_gpus: int = 8,
    model_name: str = "vggish",
) -> None:
    """
    Compute FAD for multiple inference runs, sharing one GPU worker pool.

    Args:
        runs: list of (run_key, output_wav_path, csv_path).
        reference_path: reference WAV root ({composer}/{genre}/*.wav).
        num_gpus: number of GPUs to use in parallel.
    """
    import multiprocessing as mp

    _maybe_debug_torch()
    if model_name != "vggish":
        raise ValueError("Parallel FAD only supports model_name='vggish'")

    all_tasks: list[dict] = []
    csv_by_run: dict[str, str] = {}
    for run_key, output_path, csv_path in runs:
        if not Path(output_path).is_dir():
            print(f"Skip {output_path}: directory not found")
            continue
        tasks, _, _ = _build_fad_tasks(reference_path, output_path, run_key)
        if not tasks:
            print(f"No FAD tasks for {output_path}")
            continue
        all_tasks.extend(tasks)
        csv_by_run[run_key] = csv_path

    if not all_tasks:
        print("No FAD tasks to run.")
        return

    try:
        import torch
        num_gpus = min(num_gpus, max(1, torch.cuda.device_count()))
    except Exception:
        num_gpus = 1

    print(
        f"FAD parallel: {len(all_tasks)} tasks across {len(csv_by_run)} run(s) "
        f"on {num_gpus} GPU(s)",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=_fad_worker_loop,
            args=(task_queue, result_queue, gpu_id),
        )
        p.start()
        workers.append(p)

    for task in all_tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    results_by_run: dict[str, list[dict]] = {k: [] for k in csv_by_run}
    for _ in range(len(all_tasks)):
        row = result_queue.get()
        results_by_run[row["run_key"]].append(row)

    for p in workers:
        p.join()

    for run_key, csv_path in csv_by_run.items():
        _write_fad_csv(csv_path, results_by_run[run_key])
        print(f"Successfully wrote FAD scores to {csv_path}", flush=True)


def compute_fad_by_category(
    reference_path: str,
    output_path: str,
    csv_path: str,
    model_name: str = "vggish",
) -> None:
    """
    Compute FAD scores between reference and output directories by matching 
    {composer}/{genre} structures.
    
    This function:
    1. Identifies composer/genre combinations in reference_path.
    2. Ignores combinations in output_path that are not in reference_path.
    3. Creates temporary dirs and symlinks matching wav files into them for FAD (no file copy).
    4. Computes both per-category and overall FAD scores.
    5. Outputs results to a CSV file. Temporary dirs are removed on exit.
    
    Args:
        reference_path: Path to reference audio directory ({composer}/{genre}/*.wav).
        output_path: Path to output audio directory ({composer}/{genre}/*.wav).
        csv_path: Path to save the results CSV.
        model_name: FAD model name (default: "vggish").
    """
    _maybe_debug_torch()

    ref_root = Path(reference_path)
    out_root = Path(output_path)

    FrechetAudioDistance = _load_frechet_audio_distance()
    fad = FrechetAudioDistance(model_name=model_name, use_pca=False)

    tasks, _, _ = _build_fad_tasks(reference_path, output_path, str(csv_path))
    results = []
    for task in tasks:
        rel_label = f"{task['composer']}/{task['genre']}"
        print(f"Processing category: {rel_label}...")
        ref_files = [Path(p) for p in task["ref_files"]]
        out_files = [Path(p) for p in task["out_files"]]
        try:
            score = _fad_score_with_model(fad, ref_files, out_files)
            results.append({
                "composer": task["composer"],
                "genre": task["genre"],
                "fad_score": score,
            })
        except Exception as e:
            print(f"Error computing FAD for {rel_label}: {e}")
            results.append({
                "composer": task["composer"],
                "genre": task["genre"],
                "fad_score": f"Error: {e}",
            })

    _write_fad_csv(csv_path, results)
    print(f"Successfully wrote FAD scores to {csv_path}")


def compute_fair_fad_by_category(
    reference_path: str,
    output_path: str,
    csv_path: str,
    model_name: str = "vggish",
    sample_size: int | None = None,
    max_sample_size: int | None = 20,
    num_bootstrap: int = 30,
    seed: int = 3407,
    include_overall: bool = True,
    dtype: str = "float32",
) -> None:
    """
    Compute fair FAD with equal per-category sample count and per-file weighting.

    For each composer/genre:
    1) Build one embedding per WAV file by averaging frame embeddings inside each file.
    2) Draw equal-size subsets from reference/output files.
    3) Bootstrap multiple rounds and report mean/std.
    """
    _maybe_debug_torch()
    if model_name != "vggish":
        raise ValueError("Fair FAD currently supports model_name='vggish' only")

    FrechetAudioDistance = _load_frechet_audio_distance()
    fad = FrechetAudioDistance(model_name=model_name, use_pca=False)

    tasks = _build_fair_fad_tasks(
        reference_path,
        output_path,
        str(csv_path),
        sample_size=sample_size,
        max_sample_size=max_sample_size,
        include_overall=include_overall,
    )

    results = []
    for task in tasks:
        rel_label = f"{task['composer']}/{task['genre']}"
        ref_files = [Path(p) for p in task["ref_files"]]
        out_files = [Path(p) for p in task["out_files"]]
        fair_n = task["fair_sample_size"]

        print(
            f"Processing fair FAD: {rel_label} "
            f"(ref={len(ref_files)}, out={len(out_files)}, sample={fair_n}, bootstrap={num_bootstrap})..."
        )
        try:
            stats = _bootstrap_fair_fad_with_model(
                fad=fad,
                ref_files=ref_files,
                out_files=out_files,
                sample_size=fair_n,
                num_bootstrap=num_bootstrap,
                seed=seed,
                dtype=dtype,
            )
            results.append({
                "composer": task["composer"],
                "genre": task["genre"],
                "fad_score": stats["fad_score_mean"],
                "fad_score_std": stats["fad_score_std"],
                "fad_score_min": stats["fad_score_min"],
                "fad_score_max": stats["fad_score_max"],
                "sample_size": stats["sample_size"],
                "num_bootstrap": stats["num_bootstrap"],
                "ref_n_files": stats["ref_n_files"],
                "out_n_files": stats["out_n_files"],
            })
        except Exception as e:
            print(f"Error computing fair FAD for {rel_label}: {e}")
            results.append({
                "composer": task["composer"],
                "genre": task["genre"],
                "fad_score": f"Error: {e}",
            })

    _write_fair_fad_csv(csv_path, results)
    print(f"Successfully wrote fair FAD scores to {csv_path}")


if __name__ == "__main__":
    reference_path = "/root/outputs/reference_audio/test"
    output_path = "/root/outputs/inference/2026-03-24_221303"
    csv_path = "/root/outputs/inference/2026-03-24_221303/fad.csv"
    compute_fad_by_category(reference_path, output_path, csv_path, model_name="vggish")
