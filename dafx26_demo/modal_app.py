"""On-demand Modal CUDA wrapper around the same local inference contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import modal
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install Modal extras: pip install -e '.[dafx26-demo,modal]'") from exc

LOCAL_ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "ee77c63dfefaf51f9ffa9299e6f9ce7956f9fbdb"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "mido",
        "numpy",
        "pandas",
        "huggingface_hub",
        "flask",
        "waitress",
    )
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(
        str(LOCAL_ROOT),
        remote_path="/app",
        ignore=[".venv", "models", "outputs", ".git", "dafx26_demo/static/vendor"],
    )
)

app = modal.App("dafx26-demo", image=image)
volume = modal.Volume.from_name("dafx-midi-models", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=15 * 60,
    volumes={"/models": volume},
)
def generate_remote(payload: dict[str, Any]) -> dict[str, Any]:
    import os
    import sys
    from pathlib import Path

    os.chdir("/app")
    sys.path.insert(0, "/app")
    from huggingface_hub import snapshot_download
    import torch

    from dafx26_demo.inference import ModelHub, generate
    from dafx26_demo.schema import request_from_mapping

    local_dir = "/models/MIDI_tokenization_models"
    snapshot_download(
        "cnmat/MIDI_tokenization_models",
        revision=payload.get("model_revision") or MODEL_REVISION,
        local_dir=local_dir,
    )
    hub = ModelHub(Path(local_dir), torch.device("cuda"), keep_on_device=True)
    request = request_from_mapping({**payload, "device": "cuda"})
    return generate(hub, request, use_cache=False).to_dict()


@app.local_entrypoint()
def main(mode: str = "note_velocity_pedal", max_tokens: int = 256) -> None:
    payload = {
        "mode": mode,
        "composer": "Chopin",
        "genre": "etude",
        "seed": 0,
        "max_tokens": max_tokens,
        "temperature": 0.95,
        "top_p": 0.98,
        "device": "cuda",
        "model_revision": MODEL_REVISION,
    }
    print("cold call")
    first = generate_remote.remote(payload)
    print(first)
    print("warm call")
    second = generate_remote.remote(payload)
    print(second)
