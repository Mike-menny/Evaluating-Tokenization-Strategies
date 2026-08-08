"""
MIDI to audio synthesis using FluidSynth with optional loudness normalization.

This module provides synthesize_midi_to_audio() and does not shadow the
external 'midi2audio' package (used here for FluidSynth).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Optional dependencies for audio synthesis
SYNTHESIS_AVAILABLE = False
LOUDNESS_NORM_AVAILABLE = False
_midi2audio_pkg = None

def _import_midi2audio_pkg():
    """Import the installed midi2audio package, avoiding shadowing by this module when run from src/evaluation."""
    global _midi2audio_pkg
    if _midi2audio_pkg is not None:
        return _midi2audio_pkg
    existing = sys.modules.get("midi2audio")
    if existing is not None and getattr(existing, "FluidSynth", None) is not None:
        _midi2audio_pkg = existing
        return _midi2audio_pkg
    # Clear shadowed module so "import midi2audio" can load the real package
    if existing is not None:
        del sys.modules["midi2audio"]
    our_dir = os.path.abspath(os.path.dirname(__file__))
    path_save = list(sys.path)
    try:
        sys.path = [p for p in sys.path if os.path.abspath(p) != our_dir]
        import midi2audio as pkg
        if getattr(pkg, "FluidSynth", None) is not None:
            _midi2audio_pkg = pkg
            return _midi2audio_pkg
    except ImportError:
        pass
    finally:
        sys.path = path_save
    return None

try:
    import librosa
    import soundfile as sf
    if _import_midi2audio_pkg() is not None:
        SYNTHESIS_AVAILABLE = True
    try:
        import pyloudnorm as pyln
        LOUDNESS_NORM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


DEFAULT_SAMPLERATE = 16000  # 16 kHz for faster synthesis and smaller MP3

_EVAL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SOUNDFONT_CANDIDATES = (
    _EVAL_ROOT / "soundfonts" / "FluidR3_GM.sf2",
    _EVAL_ROOT / "soundfonts" / "FluidR3_GM" / "FluidR3_GM.sf2",
    Path("soundfonts/FluidR3_GM/FluidR3_GM.sf2"),
)


def default_soundfont_path() -> str:
    env = os.environ.get("MIDI_SOUNDFONT", "").strip()
    if env:
        return env
    for candidate in _DEFAULT_SOUNDFONT_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return str(_DEFAULT_SOUNDFONT_CANDIDATES[0])


def synthesize_midi_to_audio(
    midi_path: str,
    soundfont_path: Optional[str] = None,
    save_mp3: bool = True,
    samplerate: Optional[int] = DEFAULT_SAMPLERATE,
    target_loudness: float = -18.0,
    output_dir: Optional[str] = None,
    debug_ffmpeg: bool = False,
) -> bool:
    """
    Synthesize MIDI file to audio (WAV/MP3) using FluidSynth with loudness normalization.

    Args:
        midi_path: Path to MIDI file
        soundfont_path: Path to SoundFont (.sf2) file
        save_mp3: If True, convert to MP3 and delete WAV
        samplerate: Sample rate for audio (default: 16000 Hz for speed/size)
        target_loudness: Target loudness in LUFS (default: -18.0)
        output_dir: If set, write output file(s) into this directory (same basename).
        debug_ffmpeg: If True, print ffmpeg stdout/stderr instead of hiding (for debugging).

    Returns:
        True if successful, False otherwise
    """
    if not SYNTHESIS_AVAILABLE:
        print("Warning: Audio synthesis libraries not available. Skipping synthesis.")
        print("Install with: conda install conda-forge::fluidsynth conda-forge::ffmpeg")
        print("              pip install midi2audio librosa soundfile pyloudnorm")
        return False

    if not soundfont_path:
        soundfont_path = default_soundfont_path()
    if not os.path.isfile(soundfont_path):
        print(f"Warning: SoundFont not found: {soundfont_path}")
        print("Set MIDI_SOUNDFONT or pass soundfont_path=...")
        return False

    try:
        name = os.path.basename(midi_path)
        low = name.lower()
        if low.endswith(".midi"):
            base = name[:-5]
        elif low.endswith(".mid"):
            base = name[:-4]
        else:
            base = name
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            wav_path = os.path.join(output_dir, base + ".wav")
        else:
            wav_path = os.path.join(os.path.dirname(midi_path), base + ".wav")

        # Initialize FluidSynth (from the installed midi2audio package)
        pkg = _import_midi2audio_pkg()
        if pkg is None:
            return False
        fs = pkg.FluidSynth(soundfont_path)
        if samplerate is not None:
            fs.sample_rate = samplerate

        fs.midi_to_audio(midi_path, wav_path)

        # Load and trim silence from audio (resample to samplerate when set, e.g. 16 kHz)
        wav, sr = librosa.load(wav_path, sr=samplerate)
        wav, _ = librosa.effects.trim(wav, top_db=30)

        # Apply loudness normalization
        if LOUDNESS_NORM_AVAILABLE:
            try:
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(wav)
                wav = pyln.normalize.loudness(wav, loudness, target_loudness)
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / max(abs(wav.max()), abs(wav.min()))
            except Exception as e:
                print(f"Warning: Loudness normalization failed: {e}")

        # Write normalized audio
        sf.write(wav_path, wav, sr)

        if save_mp3:
            mp3_path = (
                os.path.join(output_dir, base + ".mp3")
                if output_dir is not None
                else os.path.join(os.path.dirname(midi_path), base + ".mp3")
            )
            ar = f" -ar {sr}" if sr else ""
            cmd = f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2{ar} {mp3_path} -y"
            if debug_ffmpeg:
                import subprocess
                print(f"[debug] Running: {cmd}")
                r = subprocess.run(cmd, shell=True)
                ret = r.returncode
            else:
                ret = os.system(cmd + " >/dev/null 2>&1")
            # Only remove WAV if MP3 was created successfully; otherwise keep WAV
            if ret == 0 and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            elif ret != 0:
                print(f"Warning: ffmpeg failed (exit {ret}) for {midi_path}, keeping WAV")

        return True

    except Exception as e:
        print(f"Error synthesizing MIDI to audio: {e}")
        return False


def synthesize_midi_dir_to_audio(
    path: str,
    audio: str,
    soundfont_path: Optional[str] = None,
    save_mp3: bool = True,
    samplerate: Optional[int] = DEFAULT_SAMPLERATE,
    target_loudness: float = -18.0,
) -> int:
    """
    Recursively find all MIDI files under `path`, synthesize each to audio,
    and write outputs under `audio` preserving the same relative structure.

    Args:
        path: Root directory to search for .mid files (recursive).
        audio: Output directory for generated audio files (created as needed).
        soundfont_path: Path to SoundFont (.sf2) file.
        save_mp3: If True, output MP3 and remove WAV.
        samplerate: Sample rate for audio (default: 16000 Hz).
        target_loudness: Target loudness in LUFS.

    Returns:
        Number of MIDI files successfully synthesized.
    """
    path = os.path.abspath(path)
    audio = os.path.abspath(audio)
    midi_files = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            low = f.lower()
            if low.endswith(".mid") or low.endswith(".midi"):
                midi_files.append(os.path.join(root, f))

    success_count = 0
    for midi_path in midi_files:
        rel = os.path.relpath(midi_path, path)
        rel_dir = os.path.dirname(rel)
        out_dir = os.path.join(audio, rel_dir) if rel_dir else audio
        ok = synthesize_midi_to_audio(
            midi_path,
            soundfont_path,
            save_mp3=save_mp3,
            samplerate=samplerate,
            target_loudness=target_loudness,
            output_dir=out_dir,
        )
        if ok:
            success_count += 1
    return success_count


def main() -> None:
    """Convert every .mid under root/composer/genre/ to .mp3 in the same directory (in-place)."""
    parser = argparse.ArgumentParser(
        description="Convert MIDI files under root/composer/genre/ to MP3 in the same dirs (same format/sample rate)."
    )
    parser.add_argument("dir", type=str, help="Root directory containing composer/genre/ subdirs with .mid files")
    parser.add_argument(
        "--soundfont",
        type=str,
        default=default_soundfont_path(),
        help="Path to SoundFont .sf2 (or set MIDI_SOUNDFONT)",
    )
    parser.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLERATE, help="Output sample rate (default: 16000)")
    parser.add_argument("--loudness", type=float, default=-18.0, help="Target loudness LUFS (default: -18.0)")
    parser.add_argument("--debug-ffmpeg", action="store_true", help="Print ffmpeg command")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)

    total = 0
    for composer in sorted(os.listdir(root)):
        composer_path = os.path.join(root, composer)
        if not os.path.isdir(composer_path):
            continue
        for genre in sorted(os.listdir(composer_path)):
            genre_path = os.path.join(composer_path, genre)
            if not os.path.isdir(genre_path):
                continue
            n_genre = 0
            for f in sorted(os.listdir(genre_path)):
                low = f.lower()
                if not (low.endswith(".mid") or low.endswith(".midi")):
                    continue
                midi_path = os.path.join(genre_path, f)
                ok = synthesize_midi_to_audio(
                    midi_path,
                    soundfont_path=args.soundfont,
                    save_mp3=True,
                    samplerate=args.samplerate,
                    target_loudness=args.loudness,
                    output_dir=None,
                    debug_ffmpeg=args.debug_ffmpeg,
                )
                if ok:
                    n_genre += 1
                    total += 1
            if n_genre > 0:
                print(f"{composer}/{genre}: {n_genre} MIDI -> MP3")
    print(f"Done. Synthesized {total} MIDI files to MP3 in place.")


if __name__ == "__main__":
    main()
