#!/usr/bin/env python3
"""Train the compact composer/genre conditional MIDI Transformer from scratch.

Native DDP (8 GPU):
torchrun --nproc_per_node=8 -m src.train.train_conditional \
  --cache_dir cache/conditional_asap_full \
  --output_dir outputs/asap_full_ft \
  --tokenization_mode full \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size 28 \
  --gradient_accumulation_steps 2 --gradient_checkpointing \
  --num_workers 8 \
  --num_train_epochs 50
torchrun --nproc_per_node=8 -m src.train.train_conditional \
  --cache_dir cache/conditional_asap_note_velocity \
  --output_dir outputs/asap_note_velocity_ft \
  --resume_from_checkpoint outputs/maestro_pretrain_nvp/final \
  --tokenization_mode note_velocity \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size 28 \
  --gradient_accumulation_steps 2 --gradient_checkpointing \
  --num_workers 8 \
  --num_train_epochs 62

MAESTRO pretraining (note_velocity_pedal, deduped vs ASAP):
torchrun --nproc_per_node=8 -m src.train.train_conditional \
  --cache_dir cache/conditional_maestro_nvp \
  --output_dir outputs/maestro_pretrain_nvp \
  --tokenization_mode note_velocity_pedal \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 --gradient_checkpointing \
  --num_workers 8 \
  --num_train_epochs 10

note_pedal (notes + pedal without velocity):
torchrun --nproc_per_node=8 -m src.train.train_conditional \
  --cache_dir cache/conditional_asap_note_pedal \
  --output_dir outputs/asap_note_pedal_ft \
  --tokenization_mode note_pedal \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size 28 \
  --gradient_accumulation_steps 2 --gradient_checkpointing \
  --num_workers 8 \
  --num_train_epochs 50

DeepSpeed ZeRO-2 (8 GPU):
deepspeed --num_gpus=8 --module src.train.train_conditional \
  --cache_dir cache/conditional_asap_full \
  --output_dir outputs/asap_full \
  --tokenization_mode full \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 --gradient_checkpointing \
  --deepspeed_config configs/ds_zero2.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.dataset.conditional_asap import (
    CachedConditionalAsapDataset,
    ConditionalAsapDataset,
    ConditionalMidiCollator,
    condition_metadata,
)
from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer
from src.tokenization.conditional_vocab import (
    COMPOSER_PAD_ID,
    GENRE_PAD_ID,
    PAD_ID,
    TOKENIZATION_MODE_CHOICES,
    WINDOW_SEC_DEFAULT,
    describe_dropped_vocab_regions,
    get_tokenization_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed() -> tuple[int, int, int]:
    if not _distributed():
        return 0, 1, 0
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if _distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _setup_rank0_logging(output_dir: str, rank: int):
    if not _is_main(rank):
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_file


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _load_or_create_model(config: ConditionalMidiConfig, args: argparse.Namespace, device: torch.device):
    if args.resume_from_checkpoint:
        model, checkpoint_config, (missing, unexpected) = ConditionalMidiTransformer.resume_from_checkpoint(
            args.resume_from_checkpoint,
            config,
            map_location=device,
        )
        model = model.to(device)
        if _is_main(int(os.environ.get("RANK", "0"))):
            ckpt_mode = checkpoint_config.get("tokenization_mode", "?")
            ckpt_vocab = checkpoint_config.get("vocab_size")
            if ckpt_mode != config.tokenization_mode:
                print(f"Resuming: tokenization_mode {ckpt_mode!r} -> {config.tokenization_mode!r}")
            if ckpt_vocab != config.vocab_size:
                dropped = describe_dropped_vocab_regions(ckpt_vocab, config.vocab_size)
                print(
                    f"Resuming: vocab {ckpt_vocab} -> {config.vocab_size} "
                    f"(dropped {ckpt_vocab - config.vocab_size} rows: {dropped})"
                )
            if checkpoint_config.get("max_seq_len") != config.max_seq_len:
                print(f"Resuming: max_seq_len {checkpoint_config.get('max_seq_len')} -> {config.max_seq_len}")
            if missing:
                print(f"  missing keys (randomly initialised): {missing}")
            if unexpected:
                print(f"  unexpected keys (ignored): {unexpected}")
        return model
    return ConditionalMidiTransformer(config).to(device)


def _save_checkpoint(model, output_dir: Path, name: str, extra_config: dict, deepspeed: bool = False) -> None:
    tag = str(output_dir / name)
    if deepspeed:
        model.save_checkpoint(str(output_dir), tag=name)
        if _is_main(0):
            unwrapped = model.module if hasattr(model, "module") else model
            payload = {**extra_config, "deeppeed_tag": name}
            (output_dir / name).mkdir(parents=True, exist_ok=True)
            ((output_dir / name) / "extra_config.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
    else:
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(output_dir / name, extra_config=extra_config)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, bf16: bool, max_batches: int, deepspeed: bool = False) -> float:
    if not deepspeed:
        model.eval()
    losses: list[float] = []
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda" and not deepspeed):
            loss = model(**batch)["loss"]
        if deepspeed:
            losses.append(float(loss.detach().cpu()))
        else:
            losses.append(float(loss.detach().cpu()))
    if not deepspeed:
        model.train()
    return sum(losses) / max(1, len(losses))


def train(args: argparse.Namespace) -> None:
    use_deepspeed = bool(args.deepspeed_config)
    rank, world_size, local_rank = _setup_distributed()
    log_file = _setup_rank0_logging(args.output_dir, rank)
    _seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        train_dataset = CachedConditionalAsapDataset(cache_dir / "train.pt", max_items=args.max_train_rows)
        eval_dataset = CachedConditionalAsapDataset(cache_dir / "validation.pt", max_items=args.max_eval_rows)
    else:
        train_dataset = ConditionalAsapDataset(
            manifest_csv=args.manifest,
            data_root=args.data_root,
            split="train",
            tokenization_mode=args.tokenization_mode,
            window_duration_sec=args.window_duration_sec,
            max_rows=args.max_train_rows,
        )
        eval_dataset = ConditionalAsapDataset(
            manifest_csv=args.manifest,
            data_root=args.data_root,
            split="validation",
            tokenization_mode=args.tokenization_mode,
            window_duration_sec=args.window_duration_sec,
            max_rows=args.max_eval_rows,
        )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if world_size > 1 else None
    collator = ConditionalMidiCollator(pad_to_multiple_of=args.pad_to_multiple_of, pad_token_id=PAD_ID, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    spec = get_tokenization_spec(args.tokenization_mode)
    metadata = condition_metadata()
    config = ConditionalMidiConfig(
        vocab_size=spec.vocab_size,
        num_composers=metadata["num_composers"],
        num_genres=metadata["num_genres"],
        num_composer_ids=metadata["num_composer_ids"],
        num_genre_ids=metadata["num_genre_ids"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        cond_dim=args.cond_dim,
        tokenization_mode=args.tokenization_mode,
    )
    model = _load_or_create_model(config, args, device)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    extra_config = {
        **metadata,
        "tokenization_mode": args.tokenization_mode,
        "window_duration_sec": args.window_duration_sec,
        "small_vocab": True,
    }
    output_dir = Path(args.output_dir)
    if _is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    if use_deepspeed:
        import deepspeed
        ds_config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["gradient_clipping"] = args.max_grad_norm
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_config,
        )
        model = model_engine
        device = model.device
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
        total_steps = max(1, steps_per_epoch * args.num_train_epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if _is_main(rank):
        (output_dir / "run_config.json").write_text(
            json.dumps({"args": vars(args), "model": config.__dict__, **extra_config}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"train windows={len(train_dataset)}, eval windows={len(eval_dataset)}, "
            f"vocab={spec.vocab_size}, composers={metadata['num_composers']}, genres={metadata['num_genres']}, "
            f"deeppeed={use_deepspeed}, grad_ckpt={args.gradient_checkpointing}"
        )

    if not use_deepspeed:
        steps_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
        total_steps = max(1, steps_per_epoch * args.num_train_epochs)
    else:
        total_steps = -1

    global_step = 0
    if not use_deepspeed:
        optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(args.num_train_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            if use_deepspeed:
                loss = model(**batch)["loss"]
                model.backward(loss)
                model.step()
                global_step = model.global_steps
            else:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                    loss = model(**batch)["loss"] / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps != 0 and (step + 1) != len(train_loader):
                    continue
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if _is_main(rank) and global_step % args.logging_steps == 0:
                loss_val = float(loss.detach().cpu())
                print(f"epoch={epoch + 1} step={global_step} loss={loss_val:.4f}")
            if _is_main(rank) and args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_checkpoint(model, output_dir, f"checkpoint-{global_step}", extra_config, deepspeed=use_deepspeed)
            if args.eval_steps > 0 and global_step % args.eval_steps == 0 and len(eval_dataset) > 0:
                eval_loss = evaluate(model, eval_loader, device, args.bf16, args.max_eval_batches, deepspeed=use_deepspeed)
                if _is_main(rank):
                    print(f"eval step={global_step} loss={eval_loss:.4f}")

    if _is_main(rank):
        _save_checkpoint(model, output_dir, "final", extra_config, deepspeed=use_deepspeed)
        # Also save a standalone pytorch_model.bin for easy inference
        if use_deepspeed:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.save_pretrained(output_dir / "final_standalone", extra_config=extra_config)
    _ = log_file
    _cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact composer/genre conditional MIDI Transformer")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "asap-dataset-master" / "asap_v2_manifest.csv"))
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "asap-dataset-master"))
    parser.add_argument("--cache_dir", type=str, default="", help="Offline cache directory containing train.pt and validation.pt")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "conditional_midi"))
    parser.add_argument("--tokenization_mode", type=str, default="full", choices=TOKENIZATION_MODE_CHOICES)
    parser.add_argument("--window_duration_sec", type=float, default=WINDOW_SEC_DEFAULT)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--ffn_dim", type=int, default=3072)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_eval_batches", type=int, default=50)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_eval_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Load model weights from a saved checkpoint/final directory")
    parser.add_argument("--deepspeed_config", type=str, default="", help="DeepSpeed JSON config; enables ZeRO when provided")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Accepted for CLI compatibility; ConditionalMidiTransformer always uses PyTorch SDPA",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="DeepSpeed/torchrun injected local rank")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
