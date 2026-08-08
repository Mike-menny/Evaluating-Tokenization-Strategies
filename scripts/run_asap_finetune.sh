#!/usr/bin/env bash
# Finetune one ASAP tokenization mode from a Lakh+MAESTRO pretrained checkpoint.
# Usage:
#   MODE=note_velocity bash scripts/run_asap_finetune.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:?set MODE to one of: note note_pedal note_velocity note_velocity_beat note_velocity_pedal full}"
NPROC="${NPROC:-8}"
PRETRAIN="${PRETRAIN:-outputs/lakh_maestro_pretrain_nvp/final}"
CACHE_DIR="${CACHE_DIR:-cache/conditional_asap_${MODE}}"
OUT_DIR="${OUT_DIR:-outputs/asap_${MODE}_ft}"
EPOCHS="${EPOCHS:-50}"
BS="${BS:-28}"

mkdir -p outputs cache

if [[ ! -f "$CACHE_DIR/train.pt" ]]; then
  echo "Missing $CACHE_DIR/train.pt"
  echo "See scripts/data/README.md to build ASAP caches."
  exit 1
fi

torchrun --nproc_per_node="$NPROC" -m src.train.train_conditional \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUT_DIR" \
  --resume_from_checkpoint "$PRETRAIN" \
  --tokenization_mode "$MODE" \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps 2 --gradient_checkpointing \
  --num_workers 8 \
  --num_train_epochs "$EPOCHS"

echo "Done. Model -> $OUT_DIR/final"
