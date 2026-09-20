# Evaluating-Tokenization-Strategies-for-Expressive-Classical-Piano-Performance-Generation
for DAFx 2026 demonstration track submission

Composer/genre-conditional MIDI Transformers trained with **multiple tokenization schemes**.

This release contains the code for:

1. **Pretraining** on **Lakh MIDI** (piano-cleaned) + **MAESTRO** (with augmentation)
2. **Finetuning** on **ASAP** (with augmentation)
3. **Inference** and **FAD** evaluation

Released checkpoints (six ASAP tokenization modes):  
[cnmat/MIDI_tokenization_models](https://huggingface.co/cnmat/MIDI_tokenization_models)

---

## Repository layout

```
Evaluating-Tokenization-Strategies/
├── README.md
├── MODEL_CARD.md                  # HF model card (upload with weights)
├── LICENSE
├── pyproject.toml
├── configs/ds_zero2.json
├── demo_static/                   # web demo UI
├── dafx26_demo/                   # offline comparison + continuous MLX demo
├── artifacts/                     # example generated MIDI + WAV (demo track)
├── scripts/
│   ├── run_lakh_maestro_pretrain.sh
│   ├── run_asap_finetune.sh
│   ├── run_inference.sh
│   ├── run_fad.py
│   ├── serve_midi_tokenization_demo.py
│   └── data/                      # dataset reproduction
│       ├── README.md
│       ├── generate_prompt.py     # MAESTRO prompts
│       ├── augment_dataset.py     # MAESTRO augment
│       ├── group_dataset.py       # MAESTRO -> grouped/ + final.csv
│       ├── augment_asap.py        # ASAP augment
│       └── build_asap_v2_manifest.py
├── src/
│   ├── model/conditional_transformer.py
│   ├── tokenization/
│   ├── dataset/
│   ├── train/                     # clean_lakh, pretokenize_*, train_conditional, ...
│   ├── inference/
│   └── evaluation/
└── tests/smoke_test.py
```

---

## Tokenization modes

Token IDs follow a **prefix hierarchy** (time → duration → pitch → velocity → beat → pedal). Smaller modes are vocabulary prefixes of larger ones, so a pretrained checkpoint can be sliced when finetuning.

| Mode | Velocity | Beat | Pedal | Notes |
|------|----------|------|-------|-------|
| `note` | | | | minimal |
| `note_pedal` | | | ✓ | |
| `note_velocity` | ✓ | | | |
| `note_velocity_beat` | ✓ | ✓ | | ASAP beats |
| `note_velocity_pedal` | ✓ | | ✓ | **default pretrain** |
| `full` | ✓ | ✓ | ✓ | all ASAP annotations |

Pretrain uses `note_velocity_pedal` (Lakh/MAESTRO have no reliable beat labels). Beat tokens appear at ASAP finetune for `note_velocity_beat` / `full`.

---

## Setup

```bash
cd Evaluating-Tokenization-Strategies
pip install -e .
# optional: flash-attn matching your CUDA/torch
# system: fluidsynth, ffmpeg (for FAD audio)

PYTHONPATH=. python3 tests/smoke_test.py
```

---

## Reproduce datasets

Full step-by-step (ranges, paths, order): **[`scripts/data/README.md`](scripts/data/README.md)**.

**MAESTRO (short):**

```bash
# 1) prompts  2) augment  3) group  4) pretokenize
python3 scripts/data/generate_prompt.py maestro_dataset/maestro-v3.0.0.csv maestro_dataset -o with_prompt.csv
python3 scripts/data/augment_dataset.py maestro_dataset/with_prompt.csv maestro_dataset/augmented.csv --dataset-root maestro_dataset
python3 scripts/data/group_dataset.py maestro_dataset/augmented.csv --dataset-root maestro_dataset --output-root maestro_dataset/grouped
python3 -m src.train.pretokenize_maestro \
  --maestro_csv maestro_dataset/final.csv --maestro_root maestro_dataset \
  --asap_manifest asap-dataset-master/asap_v2_manifest.csv --asap_root asap-dataset-master \
  --cache_dir cache/conditional_maestro_nvp --tokenization_mode note_velocity_pedal
```

**ASAP (short):**

```bash
python3 scripts/data/augment_asap.py \
  --data-dir asap-dataset-master/asap-dataset-master \
  --output-dir asap-dataset-master/augmented
python3 scripts/data/build_asap_v2_manifest.py \
  --data-dir asap-dataset-master/asap-dataset-master \
  --augmented-dir asap-dataset-master/augmented \
  --output asap-dataset-master/asap_v2_manifest.csv
python3 -m src.train.pretokenize_conditional \
  --manifest asap-dataset-master/asap_v2_manifest.csv \
  --data_root asap-dataset-master \
  --cache_dir cache/conditional_asap_full --tokenization_mode full
python3 -m src.train.convert_conditional_cache \
  --src cache/conditional_asap_full --dst cache/conditional_asap_note_velocity \
  --target_mode note_velocity
```

**Lakh:**

```bash
python3 -m src.train.clean_lakh --lakh_root /path/to/Lakh/raw --output_dir lakh_midi
```

Augmentation uses random factors; the **procedure and ranges** match training (exact file hashes need a fixed seed + identical sources).

---

## Training

### Pretrain (Lakh + MAESTRO)

```bash
# after MAESTRO cache exists and Lakh is cleaned:
bash scripts/run_lakh_maestro_pretrain.sh
```

Default: 24-layer / 768-d, `max_seq_len=8192`, mode `note_velocity_pedal`, 10 epochs.

### Finetune (ASAP, one mode)

```bash
MODE=note_velocity \
PRETRAIN=outputs/lakh_maestro_pretrain_nvp/final \
bash scripts/run_asap_finetune.sh
```

`--resume_from_checkpoint` loads weights only. Vocab is sliced automatically when the target mode is smaller than pretrain.

Modes: `note`, `note_pedal`, `note_velocity`, `note_velocity_beat`, `note_velocity_pedal`, `full`.

---

## Inference

```bash
MODEL=outputs/asap_note_velocity_ft/final \
MODE=note_velocity \
PROMPTS_CSV=asap-dataset-master/asap_v2_manifest.csv \
N_OUT=20 \
bash scripts/run_inference.sh
```

Generates **20 samples per unique composer+genre** on the chosen split (default `test`).  
`--tokenization_mode` must match the checkpoint `config.json`.

```bash
hf download cnmat/MIDI_tokenization_models --local-dir ./models/MIDI_tokenization_models
```

### Web demo

```bash
# after models are under models/MIDI_tokenization_models/
pip install -e .
PYTHONPATH=. python3 scripts/serve_midi_tokenization_demo.py --host 0.0.0.0 --port 9310
```

Open `http://127.0.0.1:9310/`. UI lives in `demo_static/`. All six modes load on CPU; the active mode is moved to GPU for generation.

### DAFx26 comparison and live demo

The `dafx26_demo/` package is the conference-ready demo. It compares multiple
tokenization modes with a shared seed, supports quality-checked pregenerated
fallbacks, and vendors its browser player and sampled piano for offline use.
On Apple Silicon, the optional MLX backend adds continuous, cancellable live
generation with a scrolling piano roll, playback-speed control, sustain bars,
and beat markers.

Install and download the pinned six-model release:

```bash
pip install -e '.[dafx26-demo]'
python scripts/download_dafx26_models.py
python scripts/check_dafx26_demo.py
python scripts/launch_dafx26_demo.py
```

Open `http://127.0.0.1:9310/`.

For continuous live generation on Apple Silicon:

```bash
pip install -e '.[dafx26-demo,mlx]'
python scripts/convert_dafx26_models_mlx.py
python scripts/launch_dafx26_demo.py --device mlx --live-buffer-sec 2
```

The live endpoint is intentionally available only with `--device mlx`;
PyTorch CPU, CUDA, and MPS remain available for finite comparison generation.
Offline browser assets are committed under `dafx26_demo/static/vendor/`; run
`python scripts/vendor_dafx26_static.py` only to restore missing assets.

Run the focused test suites with:

```bash
pytest -q
node --test tests/js/*.test.mjs
```

---

## Evaluation (FAD)

1. Synthesize reference ASAP test MIDI → `composer/genre/*.wav`.
2. Synthesize generated MIDI to a parallel WAV tree (`save_mp3=False`).
3. Score:

```bash
python3 scripts/run_fad.py \
  --reference outputs/asap_test_set_wav \
  --run name=asap_note_velocity_ft,wav=PATH_TO_GEN_WAV,csv=PATH_TO_fad.csv \
  --num_gpus 8
```

---

## Model

| Hyperparameter | Default |
|----------------|---------|
| Layers | 24 |
| `d_model` | 768 |
| Heads | 12 |
| FFN | 3072 |
| Condition dim | 128 |
| Max sequence length | 8192 |
| Conditioning | Cross-attention (composer + genre) |

Checkpoint: `config.json` + `pytorch_model.bin`.

---

## License

MIT — see [`LICENSE`](LICENSE). Respect upstream licenses for Lakh, MAESTRO, and ASAP, and Hugging Face terms for released weights.
