#!/usr/bin/env bash
# Conditional inference for one finetuned checkpoint.
# Usage:
#   MODEL=outputs/asap_note_velocity_ft/final MODE=note_velocity bash scripts/run_inference.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:?set MODEL to a checkpoint/final directory}"
MODE="${MODE:?set MODE to the checkpoint tokenization_mode}"
PROMPTS_CSV="${PROMPTS_CSV:-asap-dataset-master/asap_v2_manifest.csv}"
SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-outputs/inference/asap_${MODE}_ft}"
NPROC="${NPROC:-8}"
N_OUT="${N_OUT:-20}"
BS="${BS:-32}"
MAX_TOK="${MAX_TOK:-8192}"
TEMP="${TEMP:-0.8}"
SESSION="${SESSION:-$(date +%Y-%m-%d_%H%M%S)}"

torchrun --nproc_per_node="$NPROC" -m src.inference.inference_conditional \
  --model "$MODEL" \
  --prompts_csv "$PROMPTS_CSV" \
  --split "$SPLIT" \
  --output_dir "$OUT_DIR" \
  --session "$SESSION" \
  --tokenization_mode "$MODE" \
  --batch_size "$BS" \
  --max_tokens "$MAX_TOK" \
  --n_outputs "$N_OUT" \
  --temperature "$TEMP"

echo "Done. MIDI -> $OUT_DIR/$SESSION"
