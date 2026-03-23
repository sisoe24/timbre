"""
feature_extractor.py
--------------------
Extracts low-level acoustic features from audio using librosa.

These features serve two purposes:
  1. Enrich the CLAP-based analysis with physics-level signal properties
  2. Drive rule-based heuristics in the description synthesizer
     (e.g., detect silence, estimate percussiveness, roughness, etc.)

Features extracted
------------------
  - RMS energy + dynamics range
  - Spectral centroid (brightness)
  - Spectral rolloff
  - Spectral flatness (tonality vs noise-like)
  - Zero-crossing rate (noisiness indicator)
  - Onset strength envelope (transient / percussive activity)
  - Estimated tempo (if rhythmic content detected)
  - Harmonic-to-noise ratio (HNR approximation)
  - Silence / low-energy ratio
  - Frequency band energy distribution (sub, low, mid, high, air)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class AcousticFeatures:
    """
    Compact acoustic feature summary for one audio clip.
    All scalar values unless noted.
    """

    # ---- Energy -------------------------------------------------------
    rms_mean: float           # mean RMS energy (0–1 normalized waveform range)
    rms_max: float            # peak RMS energy
    rms_std: float            # RMS variability (high = dynamic content)
    dynamic_range_db: float   # peak-to-floor ratio in dB

    # ---- Spectral -------------------------------------------------------
    spectral_centroid_mean: float    # Hz — higher = brighter
    spectral_centroid_std: float
    spectral_rolloff_mean: float     # Hz — frequency below which 85% energy lives
    spectral_flatness_mean: float    # 0=tonal, 1=white-noise-like
    zero_crossing_rate_mean: float   # noisy/transient indicator

    # ---- Temporal / Transient ------------------------------------------
    onset_strength_mean: float   # average onset strength (higher = more percussive)
    onset_strength_max: float    # max onset spike (detect sharp impacts)
    num_onsets: int              # count of detected onset events
    tempo_bpm: float             # estimated tempo (0 if not rhythmic)

    # ---- Tonal ----------------------------------------------------------
    harmonic_ratio: float        # ratio of harmonic energy to total energy (0–1)

    # ---- Silence --------------------------------------------------------
    silence_ratio: float         # fraction of frames that are below energy threshold

    # ---- Frequency band energies (normalized 0–1) ----------------------
    sub_bass_energy: float       # 20–80 Hz
    bass_energy: float           # 80–250 Hz
    low_mid_energy: float        # 250–1000 Hz
    mid_energy: float            # 1000–4000 Hz
    high_energy: float           # 4000–12000 Hz
    air_energy: float            # 12000–20000 Hz

    # ---- Derived properties (human-readable heuristics) ----------------
    is_percussive: bool          # strong transients detected
    is_tonal: bool               # spectral flatness is low (tonal content)
    is_noisy: bool               # high flatness / high ZCR
    is_low_frequency_heavy: bool  # most energy in sub/bass bands
    is_broadband: bool           # energy spread across all bands

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
    rms_max = float(np.max(rms))
    rms_std = float(np.std(rms))

    # Dynamic range: difference between peak and noise floor in dB
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    dynamic_range_db = float(np.percentile(rms_db, 99) - np.percentile(rms_db, 10))

    # -- Silence ratio ----------------------------------------------------
    silence_threshold_amp = librosa.db_to_amplitude(silence_threshold_db)
    silence_ratio = float(np.mean(rms < silence_threshold_amp))

    # -- STFT / spectral features -----------------------------------------
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_centroid_mean = float(np.mean(centroid))
    spectral_centroid_std = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=sr, roll_percent=0.85
    )[0]
    spectral_rolloff_mean = float(np.mean(rolloff))

    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    spectral_flatness_mean = float(np.mean(flatness))

    zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=hop_length)[0]
    zero_crossing_rate_mean = float(np.mean(zcr))

    # -- Onset strength ---------------------------------------------------
    onset_env = librosa.onset.onset_strength(
        y=waveform, sr=sr, hop_length=hop_length
    )
    onset_strength_mean = float(np.mean(onset_env))
    onset_strength_max = float(np.max(onset_env))

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    num_onsets = int(len(onset_frames))

    # -- Tempo ------------------------------------------------------------
    try:
        tempo_result = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        tempo_bpm = float(tempo_result[0]) if len(tempo_result) > 0 else 0.0
    except Exception:
        tempo_bpm = 0.0

    # -- Harmonic content -------------------------------------------------
    harmonic, percussive = librosa.effects.hpss(waveform)
    harmonic_energy = float(np.mean(harmonic ** 2))
    total_energy = float(np.mean(waveform ** 2)) + 1e-12
    harmonic_ratio = min(1.0, harmonic_energy / total_energy)

    # -- Frequency band energies -----------------------------------------
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_energies = _compute_band_energies(magnitude, freqs)

    # -- Derived boolean heuristics --------------------------------------
    is_percussive = (onset_strength_max > 5.0) or (num_onsets > max(2, duration * 0.5))
    is_tonal = spectral_flatness_mean < 0.05 and harmonic_ratio > 0.4
    is_noisy = spectral_flatness_mean > 0.3 or zero_crossing_rate_mean > 0.15
    is_low_frequency_heavy = (
        band_energies['sub_bass'] + band_energies['bass']
    ) > 0.5
    is_broadband = (
        band_energies['sub_bass'] > 0.05
        and band_energies['high'] > 0.05
        and band_energies['air'] > 0.02
    )

    return AcousticFeatures(
        # Energy
        rms_mean=rms_mean,
        rms_max=rms_max,
        rms_std=rms_std,
        dynamic_range_db=dynamic_range_db,
        # Spectral
        spectral_centroid_mean=spectral_centroid_mean,
        spectral_centroid_std=spectral_centroid_std,
        spectral_rolloff_mean=spectral_rolloff_mean,
        spectral_flatness_mean=spectral_flatness_mean,
        zero_crossing_rate_mean=zero_crossing_rate_mean,
        # Temporal
        onset_strength_mean=onset_strength_mean,
        onset_strength_max=onset_strength_max,
        num_onsets=num_onsets,
        tempo_bpm=tempo_bpm,
        # Tonal
        harmonic_ratio=harmonic_ratio,
        # Silence
        silence_ratio=silence_ratio,
        # Bands
        sub_bass_energy=band_energies['sub_bass'],
        bass_energy=band_energies['bass'],
        low_mid_energy=band_energies['low_mid'],
        mid_energy=band_energies['mid'],
        high_energy=band_energies['high'],
        air_energy=band_energies['air'],
        # Heuristics
        is_percussive=is_percussive,
        is_tonal=is_tonal,
        is_noisy=is_noisy,
        is_low_frequency_heavy=is_low_frequency_heavy,
        is_broadband=is_broadband,
        # Duration
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_band_energies(
    magnitude: np.ndarray,
    freqs: np.ndarray,
) -> dict:
    """
    Compute normalized energy in each frequency band.

    Bands (Hz): sub_bass 20-80, bass 80-250, low_mid 250-1k,
                mid 1k-4k, high 4k-12k, air 12k-20k
    """
    bands = {
        'sub_bass': (20, 80),
        'bass': (80, 250),
        'low_mid': (250, 1000),
        'mid': (1000, 4000),
        'high': (4000, 12000),
        'air': (12000, 20000),
    }

    total_energy = np.sum(magnitude ** 2) + 1e-12
    result = {}

    for name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        band_energy = float(np.sum(magnitude[mask] ** 2)) / total_energy
        result[name] = band_energy

    return result
