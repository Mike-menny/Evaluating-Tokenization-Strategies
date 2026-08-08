#!/usr/bin/env python3
"""Convert an existing pretokenized cache to a smaller tokenization mode.

The token ID layout is a prefix hierarchy (note ⊂ velocity ⊂ beat ⊂ pedal),
so a larger-mode cache can be projected to any smaller mode by:
  - Dropping beat events if the target doesn't use beats
  - Dropping pedal events if the target doesn't use pedals
  - Stripping the velocity token from each note event if the target doesn't use velocity

Usage:
  python -m src.train.convert_conditional_cache \
    --src cache/conditional_asap_full \
    --dst cache/conditional_asap_note \
    --target_mode note
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.tokenization.conditional_vocab import (
    BOS_ID,
    EOS_ID,
    TOKENIZATION_MODE_CHOICES,
    get_tokenization_spec,
    is_beat_token,
    is_pedal_token,
    is_velocity_token,
    iter_event_slices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _convert_tokens(tokens: list[int], src_mode: str, dst_mode: str) -> list[int]:
    """Project a token sequence from *src_mode* to *dst_mode*."""
    src_spec = get_tokenization_spec(src_mode)
    dst_spec = get_tokenization_spec(dst_mode)

    # Fast path: same mode
    if src_spec == dst_spec:
        return tokens

    # Validate that source is a superset of target
    if src_spec.vocab_size < dst_spec.vocab_size:
        raise ValueError(
            f"Source mode {src_mode!r} (vocab={src_spec.vocab_size}) cannot "
            f"produce target mode {dst_mode!r} (vocab={dst_spec.vocab_size}): "
            f"target requires tokens not present in source."
        )

    # Strip BOS / EOS, work on the interior
    interior = tokens
    head: list[int] = []
    tail: list[int] = []
    if interior and interior[0] == BOS_ID:
        head.append(interior[0])
        interior = interior[1:]
    if interior and interior[-1] == EOS_ID:
        tail.append(interior[-1])
        interior = interior[:-1]

    out: list[int] = []
    for s, e in iter_event_slices(interior, mode=src_mode):
        seg = interior[s:e]
        if len(seg) == src_spec.note_event_size:
            # Note event: optionally drop velocity
            if dst_spec.use_velocity:
                out.extend(seg)
            else:
                # Keep time, dur, note; drop velocity
                out.extend(seg[:3])
        elif len(seg) == 2 and is_beat_token(seg[1]):
            if dst_spec.use_beat:
                out.extend(seg)
            # else: drop beat event
        elif len(seg) == 2 and is_pedal_token(seg[1]):
            if dst_spec.use_pedal:
                out.extend(seg)
            # else: drop pedal event
        else:
            # Unknown event type – keep as-is (shouldn't happen)
            out.extend(seg)

    return head + out + tail


def convert_split(
    src_payload: dict,
    src_mode: str,
    dst_mode: str,
) -> dict:
    """Convert one split (train/validation/test) to the target mode."""
    src_spec = get_tokenization_spec(src_mode)
    dst_spec = get_tokenization_spec(dst_mode)

    samples = src_payload["samples"]
    converted: list[dict] = []
    for sample in samples:
        tok_tensor = sample["tokens"]
        tokens = tok_tensor.tolist() if isinstance(tok_tensor, torch.Tensor) else list(tok_tensor)
        new_tokens = _convert_tokens(tokens, src_mode, dst_mode)
        converted.append({
            **sample,
            "tokens": torch.tensor(new_tokens, dtype=torch.int16),
        })

    metadata = {**src_payload["metadata"]}
    metadata["tokenization_mode"] = dst_mode
    metadata["vocab_size"] = dst_spec.vocab_size
    metadata["source_mode"] = src_mode
    metadata["num_samples"] = len(converted)

    return {"metadata": metadata, "samples": converted}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a pretokenized cache to a smaller tokenization mode"
    )
    parser.add_argument(
        "--src", type=str, required=True,
        help="Source cache directory (e.g. cache/conditional_asap_full)",
    )
    parser.add_argument(
        "--dst", type=str, required=True,
        help="Destination cache directory (e.g. cache/conditional_asap_note)",
    )
    parser.add_argument(
        "--target_mode", type=str, required=True, choices=TOKENIZATION_MODE_CHOICES,
        help="Target tokenization mode (must be <= source mode)",
    )
    parser.add_argument(
        "--splits", type=str, default="train,validation,test",
        help="Comma-separated splits to convert (default: train,validation,test)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source cache not found: {src_dir}")

    # Detect source mode from config
    config_path = src_dir / "pretokenize_config.json"
    if config_path.exists():
        src_config = json.loads(config_path.read_text(encoding="utf-8"))
        src_mode = src_config.get("tokenization_mode", "full")
    else:
        # Fallback: infer from first .pt file
        src_mode = "full"
        print(f"Warning: no pretokenize_config.json found, assuming source mode={src_mode!r}")

    src_spec = get_tokenization_spec(src_mode)
    dst_spec = get_tokenization_spec(args.target_mode)

    # Validate compatibility
    if src_spec.vocab_size < dst_spec.vocab_size:
        raise ValueError(
            f"Cannot convert {src_mode!r} (vocab={src_spec.vocab_size}) -> "
            f"{args.target_mode!r} (vocab={dst_spec.vocab_size}): target is larger."
        )
    if src_mode == args.target_mode:
        print(f"Source and target are both {src_mode!r}, nothing to do.")
        return

    dropped = []
    if src_spec.use_velocity and not dst_spec.use_velocity:
        dropped.append("velocity")
    if src_spec.use_beat and not dst_spec.use_beat:
        dropped.append("beat")
    if src_spec.use_pedal and not dst_spec.use_pedal:
        dropped.append("pedal")
    print(f"Converting: {src_mode!r} -> {args.target_mode!r} (dropping {', '.join(dropped)})")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Write destination config
    dst_config = {
        "src_cache": str(src_dir),
        "source_tokenization_mode": src_mode,
        "tokenization_mode": args.target_mode,
        "window_duration_sec": src_config.get("window_duration_sec", 100.0) if config_path.exists() else 100.0,
    }
    (dst_dir / "pretokenize_config.json").write_text(
        json.dumps(dst_config, indent=2) + "\n", encoding="utf-8"
    )

    total_kept = 0
    total_src_tokens = 0
    total_dst_tokens = 0

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        pt_path = src_dir / f"{split}.pt"
        if not pt_path.exists():
            print(f"  {split}: skip (file not found: {pt_path})")
            continue

        payload = torch.load(pt_path, map_location="cpu")
        src_samples = payload["samples"]
        print(f"  {split}: {len(src_samples)} samples, converting...", end=" ", flush=True)

        converted_payload = convert_split(payload, src_mode, args.target_mode)
        dst_path = dst_dir / f"{split}.pt"
        torch.save(converted_payload, dst_path)

        # Stats
        for s in src_samples:
            t = s["tokens"]
            total_src_tokens += t.numel() if isinstance(t, torch.Tensor) else len(t)
        for s in converted_payload["samples"]:
            t = s["tokens"]
            total_dst_tokens += t.numel() if isinstance(t, torch.Tensor) else len(t)
        total_kept += len(converted_payload["samples"])

        ratio = total_dst_tokens / max(total_src_tokens, 1)
        print(f"done ({len(src_samples)} -> {len(converted_payload['samples'])} samples, token ratio={ratio:.3f})")

    print(f"\nFinished: {src_mode!r} -> {args.target_mode!r}")
    print(f"  Samples: {total_kept}")
    print(f"  Tokens:  {total_src_tokens} -> {total_dst_tokens} ({total_dst_tokens / max(total_src_tokens, 1):.1%})")
    print(f"  Output:  {dst_dir}")


if __name__ == "__main__":
    main()
