from __future__ import annotations

from pathlib import Path

from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.tokenization.conditional_convert import tokens_to_midi  # noqa: E402
from src.tokenization.conditional_vocab import BOS_ID, EOS_ID  # noqa: E402


def tokens_to_midi_bytes(tokens: list[int], mode: str) -> bytes:
    mid = tokens_to_midi(tokens, mode=mode)
    import io

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def midi_is_valid(data: bytes) -> bool:
    if len(data) < 20 or not data.startswith(b"MThd"):
        return False
    import io
    import mido

    try:
        mid = mido.MidiFile(file=io.BytesIO(data))
    except Exception:
        return False
    return bool(mid.tracks)


def clean_tokens(tokens: list[int]) -> list[int]:
    out: list[int] = []
    for token in tokens:
        if token == BOS_ID:
            continue
        if token == EOS_ID:
            break
        out.append(token)
    return out
