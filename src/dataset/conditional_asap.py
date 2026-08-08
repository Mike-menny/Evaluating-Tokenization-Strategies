"""ASAP dataset for the two-token composer/genre conditional MIDI model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.tokenization.conditional_convert import (
    estimate_performance_duration_sec,
    midi_and_annotations_to_tokens,
)
from src.tokenization.conditional_vocab import (
    COMPOSER_PAD_ID,
    COMPOSER_UNKNOWN_ID,
    COMPOSERS,
    GENRE_PAD_ID,
    GENRE_UNKNOWN_ID,
    GENRES,
    NUM_COMPOSER_IDS,
    NUM_GENRE_IDS,
    PAD_ID,
    WINDOW_SEC_DEFAULT,
    composer_genre_from_manifest_path,
    composer_id,
    genre_id,
)


@dataclass(frozen=True)
class WindowIndex:
    row_idx: int
    window_start_sec: float


class ConditionalAsapDataset(Dataset):
    """Each item is one deterministic 100 s window from one manifest row."""

    def __init__(
        self,
        manifest_csv: str | Path,
        data_root: str | Path,
        split: Optional[str] = None,
        tokenization_mode: str = "full",
        window_duration_sec: float = WINDOW_SEC_DEFAULT,
        max_rows: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.tokenization_mode = tokenization_mode
        self.window_duration_sec = float(window_duration_sec)

        df = pd.read_csv(manifest_csv)
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        if max_rows is not None and max_rows > 0:
            df = df.iloc[:max_rows].reset_index(drop=True)
        self.labels = df
        self.windows = self._build_window_index()

    def _resolve(self, rel_path: str) -> Path:
        path = self.data_root / rel_path
        if path.exists():
            return path
        parts = Path(rel_path).parts
        if parts and parts[0] == self.data_root.name:
            alt = self.data_root / Path(*parts[1:])
            if alt.exists():
                return alt
        return path

    def _build_window_index(self) -> list[WindowIndex]:
        windows: list[WindowIndex] = []
        for row_idx, row in self.labels.iterrows():
            midi_path = self._resolve(str(row["midi_path"]))
            ann_path = self._resolve(str(row["annotation_path"]))
            duration = estimate_performance_duration_sec(midi_path, ann_path, mode=self.tokenization_mode)
            n_windows = max(1, int(math.ceil(max(duration, 0.001) / self.window_duration_sec)))
            for k in range(n_windows):
                windows.append(WindowIndex(row_idx=int(row_idx), window_start_sec=k * self.window_duration_sec))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        win = self.windows[idx]
        row = self.labels.iloc[win.row_idx]
        composer, genre = composer_genre_from_manifest_path(str(row["midi_path"]))
        midi_path = self._resolve(str(row["midi_path"]))
        ann_path = self._resolve(str(row["annotation_path"]))
        try:
            tokens = midi_and_annotations_to_tokens(
                midi_path,
                ann_path,
                window_start_sec=win.window_start_sec,
                window_duration_sec=self.window_duration_sec,
                mode=self.tokenization_mode,
                add_special_tokens=True,
            )
        except Exception as e:
            print(f"Error tokenizing {midi_path} @ {win.window_start_sec:.1f}s: {e}")
            tokens = []

        return {
            "composer_id": torch.tensor(composer_id(composer), dtype=torch.long),
            "genre_id": torch.tensor(genre_id(genre), dtype=torch.long),
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long),
            "composer": composer,
            "genre": genre,
            "window_start_sec": win.window_start_sec,
        }


class CachedConditionalAsapDataset(Dataset):
    """Dataset backed by offline-tokenized samples saved by pretokenize_conditional.py."""

    def __init__(self, cache_path: str | Path, max_items: int | None = None):
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        self.metadata = payload.get("metadata", {})
        self.samples = payload["samples"]
        if max_items is not None and max_items > 0:
            self.samples = self.samples[:max_items]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        sample = self.samples[idx]
        tokens = sample["tokens"].to(dtype=torch.long)
        return {
            "composer_id": torch.tensor(int(sample["composer_id"]), dtype=torch.long),
            "genre_id": torch.tensor(int(sample["genre_id"]), dtype=torch.long),
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
            "composer": sample.get("composer", ""),
            "genre": sample.get("genre", ""),
            "window_start_sec": float(sample.get("window_start_sec", 0.0)),
        }


class ConditionalMidiCollator:
    def __init__(self, pad_to_multiple_of: int = 8, pad_token_id: int = PAD_ID, max_length: int = 0):
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length) if max_length and max_length > 0 else 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch if len(item["input_ids"]) > 0]
        labels = [item["labels"] for item in batch if len(item["labels"]) > 0]
        keep = [item for item in batch if len(item["input_ids"]) > 0]
        if not keep:
            input_ids = [torch.tensor([PAD_ID], dtype=torch.long)]
            labels = [torch.tensor([-100], dtype=torch.long)]
            keep = batch[:1]

        if self.max_length > 0:
            input_ids = [t[:self.max_length] for t in input_ids]
            labels = [t[:self.max_length] for t in labels]

        padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        if self.pad_to_multiple_of > 1:
            target_len = (
                (padded_inputs.shape[1] + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
            pad_len = target_len - padded_inputs.shape[1]
            if pad_len:
                padded_inputs = torch.nn.functional.pad(padded_inputs, (0, pad_len), value=self.pad_token_id)
                padded_labels = torch.nn.functional.pad(padded_labels, (0, pad_len), value=-100)
        attention_mask = padded_inputs.ne(self.pad_token_id)
        return {
            "composer_ids": torch.stack([item["composer_id"] for item in keep]),
            "genre_ids": torch.stack([item["genre_id"] for item in keep]),
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }


def condition_metadata() -> dict:
    return {
        "composers": list(COMPOSERS),
        "genres": list(GENRES),
        "num_composers": len(COMPOSERS),
        "num_genres": len(GENRES),
        "num_composer_ids": NUM_COMPOSER_IDS,
        "num_genre_ids": NUM_GENRE_IDS,
        "composer_pad_id": COMPOSER_PAD_ID,
        "composer_unknown_id": COMPOSER_UNKNOWN_ID,
        "genre_pad_id": GENRE_PAD_ID,
        "genre_unknown_id": GENRE_UNKNOWN_ID,
    }
