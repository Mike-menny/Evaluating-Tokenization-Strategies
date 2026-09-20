from __future__ import annotations

import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = REPO_ROOT
MODELS_ROOT = REPO_ROOT / "models" / "MIDI_tokenization_models"
MODEL_REVISION = "ee77c63dfefaf51f9ffa9299e6f9ce7956f9fbdb"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "demo_generations"
CACHE_ROOT = REPO_ROOT / "outputs" / "demo_cache"
STATIC_ROOT = PACKAGE_ROOT / "static"
DATA_ROOT = PACKAGE_ROOT / "data"


def _repository_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


UPSTREAM_REVISION = _repository_revision()


def ensure_upstream_on_path() -> Path:
    path = str(UPSTREAM_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return UPSTREAM_ROOT
