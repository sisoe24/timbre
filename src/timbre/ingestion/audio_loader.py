"""
audio_loader.py
---------------
Handles loading, validation, and normalization of audio files.

Supported formats: .wav, .mp3, .flac, .ogg, .aiff, .m4a
Outputs a normalized, mono AudioFile dataclass ready for downstream analysis.
"""

from __future__ import annotations

import logging
from typing import List
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

# Formats that soundfile can read natively (for accurate metadata extraction)
SF_NATIVE_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}

# All formats we support (librosa handles the rest via ffmpeg / audioread)
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif', '.m4a'}


@dataclass
class AudioFile:
    """Normalized representation of a loaded audio clip."""

    path: Path
    file_name: str

    # Acoustic properties
    duration: float          # seconds
    sample_rate: int         # Hz (after resampling to target_sr)
    original_sample_rate: int
    num_channels: int        # original channel count
    format: str              # file extension without dot

    # Waveform — always mono float32, shape: (num_samples,)
    waveform: np.ndarray

    # Derived convenience
    num_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_samples = len(self.waveform)

    @property
    def is_short(self) -> bool:
        """True if clip is shorter than 0.5 s (may produce unreliable results)."""
        return self.duration < 0.5

    @property
    def is_long(self) -> bool:
        """True if clip is longer than 60 s (may be slow to process)."""
        return self.duration > 60.0

    def summary(self) -> str:
        return (
            f"{self.file_name} | {self.format.upper()} | "
            f"{self.duration:.2f}s | {self.sample_rate} Hz | "
            f"{self.num_channels}ch"
        )


def load_audio(
    path: str | Path,
    target_sr: int = 48000,
    mono: bool = True,
) -> AudioFile:
    """
    Load an audio file and return a normalized AudioFile.

    Parameters
    ----------
    path       : path to the audio file
    target_sr  : target sample rate for resampling (default 48 kHz, CLAP requirement)
    mono       : if True, downmix to mono before returning

    Returns
    -------
    AudioFile with waveform as float32 numpy array

    Raises
    ------
    FileNotFoundError   : if file does not exist
    ValueError          : if file format is unsupported or file is unreadable
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    logger.debug('Loading: %s', path)

    # -- Retrieve original metadata -------------------------------------------
    original_sr = _get_original_sr(path, ext)
    num_channels = _get_num_channels(path, ext)

    # -- Load + resample with librosa -----------------------------------------
    try:
        waveform, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            dtype=np.float32,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load audio file '{path.name}': {exc}") from exc

    # librosa.load always returns mono when mono=True
    duration = len(waveform) / sr

    audio = AudioFile(
        path=path,
        file_name=path.name,
        duration=duration,
        sample_rate=sr,
        original_sample_rate=original_sr,
        num_channels=num_channels,
        format=ext.lstrip('.'),
        waveform=waveform,
    )

    if audio.is_short:
        logger.warning(
            '%s is very short (%.2fs). Analysis may be unreliable.',
            path.name,
            audio.duration,
        )

    logger.debug('Loaded: %s', audio.summary())
    return audio


def load_batch(
    paths: List[str | Path],
    target_sr: int = 48000,
    skip_errors: bool = True,
) -> List[AudioFile]:
    """
    Load multiple audio files. Returns only successfully loaded files.

    Parameters
    ----------
    paths       : list of file paths
    target_sr   : target sample rate
    skip_errors : if True, log errors and continue; if False, raise on first error
    """
    results: List[AudioFile] = []
    failed: List[str] = []

    for p in paths:
        try:
            af = load_audio(p, target_sr=target_sr)
            results.append(af)
        except (FileNotFoundError, ValueError) as exc:
            if skip_errors:
                logger.error("Skipping '%s': %s", Path(p).name, exc)
                failed.append(str(p))
            else:
                raise

    if failed:
        logger.warning(
            'Failed to load %d/%d file(s): %s',
            len(failed),
            len(paths),
            failed,
        )

    return results


def discover_audio_files(directory: str | Path, recursive: bool = True) -> List[Path]:
    """
    Discover all supported audio files in a directory.

    Parameters
    ----------
    directory : root directory to scan
    recursive : if True, scan subdirectories as well
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = '**/*' if recursive else '*'
    found: List[Path] = []
    for ext in SUPPORTED_FORMATS:
        found.extend(directory.glob(f"{pattern}{ext}"))
        found.extend(directory.glob(f"{pattern}{ext.upper()}"))

    # Deduplicate and sort for deterministic ordering
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_original_sr(path: Path, ext: str) -> int:
    """Best-effort retrieval of the file's original sample rate."""
    if ext in SF_NATIVE_FORMATS:
        try:
            info = sf.info(str(path))
            return info.samplerate
        except Exception:
            pass
    # Fall back to librosa native read (no resampling)
    try:
        _, sr = librosa.load(str(path), sr=None, duration=0.1)
        return sr
    except Exception:
        return 0  # unknown


def _get_num_channels(path: Path, ext: str) -> int:
    """Best-effort retrieval of the file's channel count."""
    if ext in SF_NATIVE_FORMATS:
        try:
            info = sf.info(str(path))
            return info.channels
        except Exception:
            pass
    return 1  # assume mono if undetectable
