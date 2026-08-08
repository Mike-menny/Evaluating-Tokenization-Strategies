---
license: mit
tags:
  - midi
  - music-generation
  - tokenization
---
# MIDI tokenization models (Lakh+MAESTRO pretrain → ASAP finetune)

Six composer/genre-conditional MIDI Transformers, one per tokenization mode.

| subfolder | tokenization_mode |
|---|---|
| `note` | note |
| `note_pedal` | note_pedal |
| `note_velocity` | note_velocity |
| `note_velocity_beat` | note_velocity_beat |
| `note_velocity_pedal` | note_velocity_pedal |
| `full` | full |

Each folder: `config.json` + `pytorch_model.bin`.

- Pretrain: Lakh MIDI + MAESTRO (`note_velocity_pedal`)
- Finetune: ASAP v2, mode as above

Upload this file as the Hugging Face model-card `README.md` when releasing weights.
