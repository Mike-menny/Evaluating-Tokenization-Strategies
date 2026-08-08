#!/usr/bin/env bash
# Pretrain: pretokenize Lakh + MAESTRO -> merge -> train
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NPROC="${NPROC:-8}"
LAKH_DIR="${LAKH_DIR:-lakh_midi}"
MAESTRO_CACHE="${MAESTRO_CACHE:-cache/conditional_maestro_nvp}"
LAKH_CACHE="${LAKH_CACHE:-cache/conditional_lakh_nvp}"
COMBINED_CACHE="${COMBINED_CACHE:-cache/conditional_pretrain_combined}"
OUT_DIR="${OUT_DIR:-outputs/lakh_maestro_pretrain_nvp}"
MODE="${MODE:-note_velocity_pedal}"

mkdir -p outputs cache

echo "[1/3] Pretokenize Lakh (expects $LAKH_DIR/manifest.csv from clean_lakh)"
python3 -m src.train.pretokenize_lakh \
  --lakh_dir "$LAKH_DIR" \
  --cache_dir "$LAKH_CACHE" \
  --tokenization_mode "$MODE" \
  --num_workers "${NUM_WORKERS:-32}"

echo "[2/3] Merge MAESTRO + Lakh caches"
python3 -m src.train.merge_pretrain_cache \
  --inputs "$MAESTRO_CACHE" "$LAKH_CACHE" \
  --output_dir "$COMBINED_CACHE"

echo "[3/3] Pretrain"
torchrun --nproc_per_node="$NPROC" -m src.train.train_conditional \
  --cache_dir "$COMBINED_CACHE" \
  --output_dir "$OUT_DIR" \
  --tokenization_mode "$MODE" \
  --num_layers 24 --d_model 768 --n_heads 12 --ffn_dim 3072 \
  --max_seq_len 8192 --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 --gradient_checkpointing \
  --num_workers 8 \
  --num_train_epochs 10 \
  --learning_rate 3e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 20

echo "Done. Model -> $OUT_DIR/final"
