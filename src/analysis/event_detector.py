"""
event_detector.py
-----------------
Detects temporal sound events using sliding-window CLAP classification.

How it works
------------
1. CLAP runs zero-shot classification on overlapping windows
   (e.g., 2 s windows, 1 s hop)
2. For each window, the top-scoring label above a confidence threshold
   is recorded as the active sound event
3. Consecutive windows with the same top label are merged into a single event
4. Output: an ordered list of SoundEvent objects with start/end times
   and a confidence score

This approach enables temporal descriptions like:
   "Metallic impact at 0.0s → echo decay at 0.5s → silence"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models.clap_tagger import CLAPTagger

logger = logging.getLogger(__name__)

# Minimum confidence to record a detection (below this → "background noise")
MIN_CONFIDENCE_THRESHOLD = 0.25

# Minimum event duration to keep (short spurious windows are filtered)
MIN_EVENT_DURATION_S = 0.3


@dataclass
class SoundEvent:
    """
    Represents a detected sound event within a temporal window.

    Attributes
    ----------
    label      : human-readable sound descriptor (from controlled vocabulary)
    category   : high-level category (e.g., "impact", "ambience", "machinery")
    start_time : start time in seconds
    end_time   : end time in seconds
    confidence : CLAP classification confidence (0–1)
    """

    label: str
    category: str
    start_time: float
    end_time: float
    confidence: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return (
            f"[{self.start_time:.1f}s–{self.end_time:.1f}s] "
            f"{self.label} ({self.category}, conf={self.confidence:.2f})"
        )


def detect_events(
    waveform: np.ndarray,
    sr: int,
    tagger: CLAPTagger,
    candidate_labels: List[str],
    label_to_category: Dict[str, str],
    window_seconds: float = 2.0,
    hop_seconds: float = 0.5,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
    cache=None,  # Optional[LabelEmbeddingCache]
    top_k_categories: int = 5,
) -> List[SoundEvent]:
    """
    Run sliding-window CLAP classification to detect temporal sound events.

    Parameters
    ----------
    waveform         : mono float32 waveform at 48 kHz
    sr               : sample rate (must be 48000)
    tagger           : pre-loaded CLAPTagger instance
    candidate_labels : list of text labels (used only when cache=None)
    label_to_category: mapping from label → category string
    window_seconds   : CLAP window size
    hop_seconds      : window hop size
    min_confidence   : discard detections below this confidence
    cache            : LabelEmbeddingCache — when provided, text embeddings
                       are loaded from cache instead of re-encoded each window
                       (significantly faster for large vocabularies)

    Returns
    -------
    List of SoundEvent objects in temporal order
    """
    if len(waveform) == 0:
        return []

    if cache is not None:
        # Fast path: embed audio once per window, score against cached text
        window_results = _classify_windowed_cached(
            waveform=waveform,
            sr=sr,
            tagger=tagger,
            cache=cache,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            top_k_categories=top_k_categories,
        )
        # Use the cache's label_to_category for lookups
        label_to_category = cache.label_to_category
    else:
        # Legacy path: re-encode all labels on every window
        window_results = tagger.classify_windowed(
            waveform=waveform,
            sr=sr,
            candidate_labels=candidate_labels,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )

    # Convert each window to a single best-match event
    raw_events: List[Tuple[float, float, str, str, float]] = []
    for (t_start, t_end, scores) in window_results:
        if not scores:
            continue
        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        if top_score < min_confidence:
            top_label = "background noise"
            category = "background"
        else:
            category = label_to_category.get(top_label, "unknown")

        raw_events.append((t_start, t_end, top_label, category, top_score))

    # Merge consecutive windows with the same top label
    merged = _merge_consecutive_events(raw_events)

    # Filter out very short events
    filtered = [e for e in merged if e.duration >= MIN_EVENT_DURATION_S]

    return filtered


def _classify_windowed_cached(
    waveform: np.ndarray,
    sr: int,
    tagger: CLAPTagger,
    cache,  # LabelEmbeddingCache
    window_seconds: float,
    hop_seconds: float,
    top_k_categories: int = 5,
) -> List[Tuple[float, float, Dict[str, float]]]:
    """
    Sliding-window classification using pre-computed text embeddings.

    For each window: embed audio → cosine similarity against cached text matrix.
    Text is never re-encoded, making this much faster than the legacy path.
    """
    import numpy as np
    from ..models.clap_tagger import CLAP_MAX_SECONDS

    window_samples = int(window_seconds * sr)
    hop_samples = int(hop_seconds * sr)
    total_samples = len(waveform)
    results: List[Tuple[float, float, Dict[str, float]]] = []

    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        chunk = waveform[start:end]

        # Pad short final window
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        audio_embed = tagger.embed_audio(chunk, sr)
        scores = cache.classify(audio_embed, top_k_categories=top_k_categories)

        t_start = start / sr
        t_end = end / sr
        results.append((t_start, t_end, scores))

        if end >= total_samples:
            break
        start += hop_samples

    return results


def detect_events_from_full_clip(
    full_scores: Dict[str, float],
    label_to_category: Dict[str, str],
    duration: float,
    top_n: int = 5,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
) -> List[SoundEvent]:
    """
    Create a simplified single-event description from full-clip CLAP scores.

    Used when the clip is too short for sliding-window analysis (< 2 s)
    or when windowed analysis is disabled.
    """
    sorted_labels = sorted(full_scores.items(), key=lambda x: x[1], reverse=True)
    events = []
    for label, score in sorted_labels[:top_n]:
        if score >= min_confidence:
            category = label_to_category.get(label, "unknown")
            events.append(
                SoundEvent(
                    label=label,
                    category=category,
                    start_time=0.0,
                    end_time=duration,
                    confidence=score,
                )
            )
    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_consecutive_events(
    raw: List[Tuple[float, float, str, str, float]],
) -> List[SoundEvent]:
    """
    Merge adjacent windows that share the same label into a single SoundEvent.
    Confidence is averaged across merged windows.
    """
    if not raw:
        return []

    merged: List[SoundEvent] = []
    cur_start, cur_end, cur_label, cur_cat, cur_conf = raw[0]
    conf_accum = [cur_conf]

    for t_start, t_end, label, cat, conf in raw[1:]:
        if label == cur_label:
            # Extend the current event
            cur_end = t_end
            conf_accum.append(conf)
        else:
            # Finalize current event
            merged.append(
                SoundEvent(
                    label=cur_label,
                    category=cur_cat,
                    start_time=cur_start,
                    end_time=cur_end,
                    confidence=float(np.mean(conf_accum)),
                )
            )
            cur_start, cur_end, cur_label, cur_cat, cur_conf = (
                t_start,
                t_end,
                label,
                cat,
                conf,
            )
            conf_accum = [cur_conf]

    # Finalize last event
    merged.append(
        SoundEvent(
            label=cur_label,
            category=cur_cat,
            start_time=cur_start,
            end_time=cur_end,
            confidence=float(np.mean(conf_accum)),
        )
    )

    return merged
