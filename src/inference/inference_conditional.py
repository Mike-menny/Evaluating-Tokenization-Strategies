#!/usr/bin/env python3
"""Inference for the compact composer/genre conditional MIDI Transformer.

Loads unique (composer, genre) combinations from the manifest split and generates
``--n_outputs`` samples per combination (default 20).

python3 -m src.inference.inference_conditional \
  --model outputs/asap_full/final \
  --prompts_csv asap-dataset-master/asap_v2_manifest.csv \
  --split test \
  --output_dir outputs/inference/conditional_midi_test \
  --batch_size 16 \
  --max_tokens 10000 \
  --temperature 0.95 \
  --top_p 0.98

8-GPU data parallel (``--batch_size 0`` = one mega-batch per rank, best GPU fill):
SESSION=$(date +%Y-%m-%d_%H%M%S) torchrun --nproc_per_node=8 -m src.inference.inference_conditional \
  --model outputs/asap_note_velocity_ft/checkpoint-8100 \
  --prompts_csv asap-dataset-master/asap_v2_manifest.csv \
  --split test \
  --output_dir outputs/inference/asap_note_velocity \
  --session "$SESSION" \
  --batch_size 0 \
  --max_tokens 8192 \
  --n_outputs 20 \
  --temperature 0.8
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.model.conditional_transformer import ConditionalMidiTransformer
from src.tokenization.conditional_convert import allowed_token_ids, tokens_to_midi
from src.tokenization.conditional_vocab import (
    BOS_ID,
    DUR_OFFSET,
    EOS_ID,
    NOTE_OFFSET,
    PEDAL_OFF_ID,
    PEDAL_ON_ID,
    TOKENIZATION_MODE_CHOICES,
    VELOCITY_OFFSET,
    BEAT_B_ID,
    BEAT_DB_ID,
    BEAT_BR_ID,
    composer_genre_from_manifest_path,
    composer_id,
    get_tokenization_spec,
    genre_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sanitize_path_component(s: str) -> str:
    s = re.sub(r'[/\\:*?"<>|]', "_", str(s).strip())
    return s or "unknown"


def _expand_entries(entries: list[dict], n_outputs: int) -> list[dict]:
    """Flatten (condition, sample) so generation can run in one large GPU batch."""
    if n_outputs <= 1:
        return [{**entry, "sample_idx": 0} for entry in entries]
    expanded: list[dict] = []
    for entry in entries:
        for sample_idx in range(n_outputs):
            expanded.append({**entry, "sample_idx": sample_idx})
    return expanded


def _checkpoint_candidates(model: str) -> list[Path]:
    raw = Path(model)
    bases: list[Path]
    if raw.is_absolute():
        bases = [raw]
    else:
        bases = [PROJECT_ROOT / raw, PROJECT_ROOT / "outputs" / raw, Path.cwd() / raw]

    candidates: list[Path] = []
    for base in bases:
        candidates.append(base)
        name = base.name
        parent = base.parent
        if name.isdigit():
            candidates.append(parent / f"checkpoint-{name}")
        elif name.startswith("checkpoint-"):
            suffix = name.removeprefix("checkpoint-")
            if suffix.isdigit():
                candidates.append(parent / suffix)
    return candidates


def _resolve_model_path(model: str) -> Path:
    seen: set[Path] = set()
    tried: list[Path] = []
    for candidate in _checkpoint_candidates(model):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        tried.append(resolved)
        if resolved.is_file() and resolved.name == "pytorch_model.bin":
            return resolved
        weights = resolved / "pytorch_model.bin"
        if weights.exists():
            return resolved

    tried_msg = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"Could not find model weights for {model!r}. Tried:\n  {tried_msg}\n"
        "Use a checkpoint directory like outputs/asap_note_velocity_ft/checkpoint-8100 "
        "or the pytorch_model.bin file itself."
    )


def _distributed_context(device: str) -> tuple[int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and torch.cuda.is_available():
        resolved = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    elif device == "cuda" and torch.cuda.is_available():
        resolved = "cuda:0"
    else:
        resolved = device
    return rank, world_size, resolved


def _load_entries(args: argparse.Namespace) -> list[dict]:
    if args.condition_composer and args.condition_genre:
        return [
            {
                "condition_idx": i,
                "composer": args.condition_composer,
                "genre": args.condition_genre,
            }
            for i in range(args.num_conditions)
        ]

    df = pd.read_csv(args.prompts_csv)
    if args.split:
        df = df[df["split"] == args.split]

    pairs: dict[tuple[str, str], None] = {}
    for _, row in df.iterrows():
        composer, genre = composer_genre_from_manifest_path(str(row["midi_path"]))
        if args.composer and composer != args.composer:
            continue
        if args.genre and genre != args.genre:
            continue
        pairs[(composer, genre)] = None

    unique_pairs = sorted(pairs.keys())
    print(f"Found {len(unique_pairs)} unique composer+genre combinations:")
    for composer, genre in unique_pairs:
        print(f"  - {composer} / {genre}")

    entries = [
        {
            "condition_idx": i,
            "composer": composer,
            "genre": genre,
        }
        for i, (composer, genre) in enumerate(unique_pairs)
    ]
    if args.max_rows > 0:
        entries = entries[: args.max_rows]
    return entries


def _mask_logits(
    logits: torch.Tensor,
    generated: torch.Tensor,
    mode: str,
    min_tokens: int,
) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    for i in range(logits.shape[0]):
        seq = generated[i].detach().cpu().tolist()
        seq = [t for t in seq if t != BOS_ID]
        allow_eos = len(seq) >= min_tokens
        allowed = allowed_token_ids(seq, mode=mode, allow_eos=allow_eos)
        masked[i, allowed] = logits[i, allowed]
    return masked


class FastMidiConstraint:
    """GPU-side phase mask for v2 token grammar; avoids rescanning generated tokens every step."""

    def __init__(self, batch_size: int, vocab_size: int, mode: str, min_tokens: int, device: torch.device):
        spec = get_tokenization_spec(mode)
        if spec.vocab_size > vocab_size:
            raise ValueError(
                f"Tokenization mode {spec.name!r} requires vocab_size >= {spec.vocab_size}, "
                f"but model.config.vocab_size is {vocab_size}. "
                f"This usually means the --tokenization_mode flag does not match the mode "
                f"the checkpoint was trained with. Check the checkpoint's config.json "
                f"(tokenization_mode field) and rerun with the matching mode."
            )
        self.phase = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.length = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.min_tokens = int(min_tokens)
        self.device = device
        self.eos_id = EOS_ID
        self.spec = spec
        self.vocab_size = vocab_size

        masks = []
        for _ in range(4):
            masks.append(torch.full((vocab_size,), float("-inf"), device=device))
        masks[0][3:DUR_OFFSET] = 0.0
        masks[1][DUR_OFFSET:NOTE_OFFSET] = 0.0
        if spec.use_beat:
            masks[1][[BEAT_B_ID, BEAT_DB_ID, BEAT_BR_ID]] = 0.0
        if spec.use_pedal:
            masks[1][[PEDAL_OFF_ID, PEDAL_ON_ID]] = 0.0
        masks[2][NOTE_OFFSET:VELOCITY_OFFSET] = 0.0
        masks[3][VELOCITY_OFFSET:VELOCITY_OFFSET + 128] = 0.0
        self.masks = torch.stack(masks, dim=0)

    def mask(self, logits: torch.Tensor) -> torch.Tensor:
        phase_mask = self.masks[self.phase].to(dtype=logits.dtype)
        masked = logits + phase_mask
        allow_eos = self.length >= self.min_tokens
        if bool(allow_eos.any()):
            masked[allow_eos, self.eos_id] = logits[allow_eos, self.eos_id]
        return masked

    def update(self, next_token: torch.Tensor) -> None:
        tok = next_token.to(self.device)
        phase = self.phase
        is_eos = tok.eq(self.eos_id)
        is_dur = (DUR_OFFSET <= tok) & (tok < NOTE_OFFSET)
        is_short_event = phase.eq(1) & ~is_dur

        new_phase = phase.clone()
        new_phase = torch.where(phase.eq(0), torch.ones_like(new_phase), new_phase)
        new_phase = torch.where(is_short_event, torch.zeros_like(new_phase), new_phase)
        new_phase = torch.where(phase.eq(1) & is_dur, torch.full_like(new_phase, 2), new_phase)
        if self.spec.use_velocity:
            new_phase = torch.where(phase.eq(2), torch.full_like(new_phase, 3), new_phase)
            new_phase = torch.where(phase.eq(3), torch.zeros_like(new_phase), new_phase)
        else:
            new_phase = torch.where(phase.eq(2), torch.zeros_like(new_phase), new_phase)
        new_phase = torch.where(is_eos, torch.zeros_like(new_phase), new_phase)
        self.phase = new_phase
        self.length += (~is_eos).long()


def _save_one(
    tokens: list[int],
    output_dir: Path,
    row_label: str,
    mode: str,
    save_tokens: bool,
) -> bool:
    clean: list[int] = []
    for token in tokens:
        if token == BOS_ID:
            continue
        if token == EOS_ID:
            break
        clean.append(token)
    if not clean:
        return False
    try:
        midi = tokens_to_midi(clean, mode=mode)
        midi.save(str(output_dir / f"{row_label}.mid"))
        if save_tokens:
            (output_dir / f"token_{row_label}.txt").write_text("\n".join(map(str, clean)) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        print(f"Failed to save {row_label}: {e}")
        return False


def run(args: argparse.Namespace) -> None:
    try:
        _run_impl(args)
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
        print(f"rank={rank} inference failed", flush=True)
        raise


def _run_impl(args: argparse.Namespace) -> None:
    rank, world_size, device_name = _distributed_context(args.device)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model_path = _resolve_model_path(args.model)
    print(f"Loading model from {model_path}", flush=True)
    model, config = ConditionalMidiTransformer.from_pretrained(model_path, map_location=device)
    model.to(device=device)
    if args.bf16 and device.type == "cuda":
        model.to(dtype=torch.bfloat16)
    model.eval()
    mode = args.tokenization_mode or config.get("tokenization_mode", "full")

    entries = _load_entries(args)
    entries = _expand_entries(entries, args.n_outputs)
    total_jobs = len(entries)
    if world_size > 1:
        entries = entries[rank::world_size]

    session = args.session or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    root = Path(args.output_dir) / session
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"rank={rank}/{world_size} jobs={len(entries)} "
        f"(total={total_jobs}, n_outputs={args.n_outputs}) output={root}"
    )

    ok = 0
    batch_size = len(entries) if args.batch_size <= 0 else args.batch_size
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        composer_ids = torch.tensor([composer_id(e["composer"]) for e in batch], dtype=torch.long, device=device)
        genre_ids = torch.tensor([genre_id(e["genre"]) for e in batch], dtype=torch.long, device=device)
        constraint = FastMidiConstraint(
            batch_size=len(batch),
            vocab_size=model.config.vocab_size,
            mode=mode,
            min_tokens=args.min_tokens,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
            generated = model.generate(
                composer_ids=composer_ids,
                genre_ids=genre_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                logits_processor=constraint,
            )
        for i, entry in enumerate(batch):
            composer_dir = root / _sanitize_path_component(entry["composer"])
            genre_dir = composer_dir / _sanitize_path_component(entry["genre"])
            genre_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{entry['sample_idx'] + 1}" if args.n_outputs > 1 else ""
            row_label = f"r{rank}_cond_{entry['condition_idx'] + 1:03d}{suffix}"
            if _save_one(generated[i].detach().cpu().tolist(), genre_dir, row_label, mode, args.save_tokens_txt):
                ok += 1
    print(f"rank={rank} saved={ok}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI from manifest composer/genre conditions")
    parser.add_argument("--model", type=str, required=True, help="Path to conditional checkpoint, e.g. outputs/conditional_midi/final")
    parser.add_argument("--prompts_csv", type=str, default=str(PROJECT_ROOT / "asap-dataset-master" / "asap_v2_manifest.csv"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--composer", type=str, default="")
    parser.add_argument("--genre", type=str, default="", help="Canonical genre, e.g. etude, sonata, fugue")
    parser.add_argument("--condition_composer", type=str, default="", help="Direct generation composer; bypasses manifest rows when paired with --condition_genre")
    parser.add_argument("--condition_genre", type=str, default="", help="Direct generation genre; can form unseen composer/genre pairs")
    parser.add_argument("--num_conditions", type=int, default=1, help="Number of direct-condition prompts to generate")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "inference" / "conditional_midi"))
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--tokenization_mode", type=str, default="", choices=("", *TOKENIZATION_MODE_CHOICES))
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Generation batch size after expanding condition x n_outputs. "
        "0 = one batch with all jobs assigned to this rank (recommended).",
    )
    parser.add_argument("--max_rows", type=int, default=0, help="Limit number of unique composer/genre conditions (0 = all)")
    parser.add_argument("--n_outputs", type=int, default=20, help="Number of generated samples per composer/genre condition")
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--save_tokens_txt", action="store_true", default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
