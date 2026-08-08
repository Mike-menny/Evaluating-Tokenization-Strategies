# Dataset reproduction (ASAP + MAESTRO)

These scripts rebuild the **same augmentation pipelines** used for training.
Run them from the **project root** (`MIDI_tokenization/`) so default paths resolve correctly.

```bash
cd MIDI_tokenization
export PYTHONPATH=.
```

---

## MAESTRO (pretrain half)

Official MAESTRO v3 MIDI + `maestro-v3.0.0.csv`. Place under `maestro_dataset/` (or pass explicit paths).

### 1) Prompts (deterministic; recommended for reproduction)

```bash
python3 scripts/data/generate_prompt.py \
  maestro_dataset/maestro-v3.0.0.csv \
  maestro_dataset \
  -o with_prompt.csv
```

Adds `system_prompt` / `user_prompt` from title→genre keywords.  
(`generate_description.py` is an optional LLM+search variant; not required for reproduction.)

### 2) Augment (pitch / time / velocity)

```bash
python3 scripts/data/augment_dataset.py \
  maestro_dataset/with_prompt.csv \
  maestro_dataset/augmented.csv \
  --dataset-root maestro_dataset
```

- Pitch: all of ±1,±2,±3 (6 variants per file)  
- Time: 5× uniform in **0.8–1.2**  
- Velocity: 5× uniform in **0.9–1.1**  
Writes MIDIs under `maestro_dataset/augmented/` and `augmented.csv`.

### 3) Group by composer / genre → `final.csv`

```bash
python3 scripts/data/group_dataset.py \
  maestro_dataset/augmented.csv \
  --dataset-root maestro_dataset \
  --output-root maestro_dataset/grouped
```

Writes `maestro_dataset/final.csv` with `midi_filename` relative to `grouped/`  
(e.g. `Bach/sonata/....mid` or `augmented/Bach/sonata/....mid`).

### 4) Pretokenize (dedup vs ASAP)

```bash
python3 -m src.train.pretokenize_maestro \
  --maestro_csv maestro_dataset/final.csv \
  --maestro_root maestro_dataset \
  --asap_manifest asap-dataset-master/asap_v2_manifest.csv \
  --asap_root asap-dataset-master \
  --cache_dir cache/conditional_maestro_nvp \
  --tokenization_mode note_velocity_pedal
```

---

## ASAP (finetune)

Place the ASAP tree at `asap-dataset-master/asap-dataset-master/` (composer/genre/…).

### 1) Augment

```bash
python3 scripts/data/augment_asap.py \
  --data-dir asap-dataset-master/asap-dataset-master \
  --output-dir asap-dataset-master/augmented
```

- Pitch: ±1–3 (same family as MAESTRO)  
- Time: performances **0.9–1.1**; `midi_score*` **0.25–4.0**  
- Velocity: 0.9–1.1  
Also scales matching `*_annotations.txt` when time-stretching.

### 2) Manifest

```bash
python3 scripts/data/build_asap_v2_manifest.py \
  --data-dir asap-dataset-master/asap-dataset-master \
  --augmented-dir asap-dataset-master/augmented \
  --output asap-dataset-master/asap_v2_manifest.csv
```

### 3) Pretokenize + project modes

```bash
python3 -m src.train.pretokenize_conditional \
  --manifest asap-dataset-master/asap_v2_manifest.csv \
  --data_root asap-dataset-master \
  --cache_dir cache/conditional_asap_full \
  --tokenization_mode full

python3 -m src.train.convert_conditional_cache \
  --src cache/conditional_asap_full \
  --dst cache/conditional_asap_note_velocity \
  --target_mode note_velocity
# repeat for note / note_pedal / note_velocity_beat / note_velocity_pedal as needed
```

---

## Notes

- Augmentation uses **random** factors; exact byte-identical MIDIs are not guaranteed unless you fix seeds and keep the same source files. The **procedure and ranges** match training.
- MAESTRO pretokenize **drops** pieces that overlap ASAP base recordings (dedup).
- Do not use the separate expressive-bundle builders (`augmented_v2` / `v3` under other repos); they are a different pipeline.
