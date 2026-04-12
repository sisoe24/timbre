"""
feature_extractor.py
--------------------
Extracts low-level acoustic features from audio using librosa.

Produces only the features that flow into the pipeline:
  - RMS energy + dynamic range (for AcousticSummary)
  - Spectral centroid + flatness (for AcousticSummary + heuristics)
  - Silence ratio (for acoustic flag + AcousticSummary)
  - Onset strength + count (for is_percussive heuristic)
  - Frequency band energies (for dominant_band + low-freq/broadband heuristics)
  - Five boolean flags: is_percussive, is_tonal, is_noisy,
    is_low_frequency_heavy, is_broadband
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class AcousticFeatures:
    """Compact acoustic feature summary for one audio clip."""

    # ---- Energy -------------------------------------------------------
    rms_mean: float
    dynamic_range_db: float

    # ---- Spectral -------------------------------------------------------
    spectral_centroid_mean: float
    spectral_flatness_mean: float

    # ---- Silence --------------------------------------------------------
    silence_ratio: float

    # ---- Transient ------------------------------------------------------
    onset_strength_max: float
    num_onsets: int

    # ---- Frequency band (dominant only) --------------------------------
    dominant_frequency_band: str  # sub_bass / bass / low_mid / mid / high / air

    # ---- Derived boolean heuristics ------------------------------------
    is_percussive: bool
    is_tonal: bool
    is_noisy: bool
    is_low_frequency_heavy: bool
    is_broadband: bool

    # ---- Duration -------------------------------------------------------
    duration: float


def extract_features(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    silence_threshold_db: float = -50.0,
) -> AcousticFeatures:
    """
    Extract acoustic features from a mono float32 waveform.

    Parameters
    ----------
    waveform              : mono float32 waveform, shape (num_samples,)
    sr                    : sample rate in Hz
    n_fft                 : FFT size
    hop_length            : STFT hop length
    silence_threshold_db  : frames below this level are counted as silent
    """
    duration = len(waveform) / sr

    # -- RMS energy -------------------------------------------------------
    rms = librosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    dynamic_range_db = float(np.percentile(rms_db, 99) - np.percentile(rms_db, 10))

    # -- Silence ratio ----------------------------------------------------
    silence_threshold_amp = librosa.db_to_amplitude(silence_threshold_db)
    silence_ratio = float(np.mean(rms < silence_threshold_amp))

    # -- STFT / spectral features -----------------------------------------
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    spectral_centroid_mean = float(np.mean(
        librosa.feature.spectral_centroid(S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    ))
    spectral_flatness_mean = float(np.mean(
        librosa.feature.spectral_flatness(S=magnitude)[0]
    ))

    # -- Onset strength ---------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
    onset_strength_max = float(np.max(onset_env))
    num_onsets = int(len(librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )))

    # -- Frequency band energies -----------------------------------------
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_energies = _compute_band_energies(magnitude, freqs)
    dominant_frequency_band = max(band_energies, key=band_energies.get)

    # -- Derived boolean heuristics --------------------------------------
    is_percussive = onset_strength_max > 5.0 or num_onsets > max(2, duration * 0.5)
    is_tonal = spectral_flatness_mean < 0.05
    is_noisy = spectral_flatness_mean > 0.3
    is_low_frequency_heavy = (
        band_energies['sub_bass'] + band_energies['bass']
    ) > 0.5
    is_broadband = (
        band_energies['sub_bass'] > 0.05
        and band_energies['high'] > 0.05
        and band_energies['air'] > 0.02
    )

    return AcousticFeatures(
        rms_mean=rms_mean,
        dynamic_range_db=dynamic_range_db,
        spectral_centroid_mean=spectral_centroid_mean,
        spectral_flatness_mean=spectral_flatness_mean,
        silence_ratio=silence_ratio,
        onset_strength_max=onset_strength_max,
        num_onsets=num_onsets,
        dominant_frequency_band=dominant_frequency_band,
        is_percussive=is_percussive,
        is_tonal=is_tonal,
        is_noisy=is_noisy,
        is_low_frequency_heavy=is_low_frequency_heavy,
        is_broadband=is_broadband,
        duration=duration,
    )


def _compute_band_energies(magnitude: np.ndarray, freqs: np.ndarray) -> dict:
    """Compute normalized energy in each frequency band."""
    bands = {
        'sub_bass': (20, 80),
        'bass': (80, 250),
        'low_mid': (250, 1000),
        'mid': (1000, 4000),
        'high': (4000, 12000),
        'air': (12000, 20000),
    }
    total_energy = np.sum(magnitude ** 2) + 1e-12
    return {
        name: float(np.sum(magnitude[(freqs >= f_low) & (freqs < f_high)] ** 2)) / total_energy
        for name, (f_low, f_high) in bands.items()
    }
