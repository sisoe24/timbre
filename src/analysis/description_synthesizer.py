"""
description_synthesizer.py
--------------------------
Converts structured analysis results (CLAP tags, acoustic features, events)
into clean, human-readable descriptions for audio cataloging.

Design principles
-----------------
- NO emotion, genre, or cinematic interpretation
- NO hallucination — only describe what CLAP and features confirm
- Conservative wording when confidence is low
- Consistent vocabulary (uses the controlled vocabulary from config)
- Outputs both a short_description (1 sentence) and a detailed_description
- Generates a final confidence score based on CLAP agreement + acoustic signals
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .feature_extractor import AcousticFeatures
from .event_detector import SoundEvent

logger = logging.getLogger(__name__)

# Confidence thresholds for description wording
HIGH_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.45
LOW_CONFIDENCE = 0.25

# Max tags to include in output
MAX_TAGS = 10


@dataclass
class DescriptionResult:
    """
    Structured output from the description synthesizer.
    Ready for serialization to JSON/CSV/Markdown.
    """

    short_description: str
    detailed_description: str
    tags: List[str]
    sound_events: List[str]   # human-readable event labels (ordered)
    confidence: float
    primary_category: str     # top-level category (e.g., "impact", "ambience")
    primary_label: str        # most confident CLAP label


def synthesize_description(
    file_name: str,
    full_scores: Dict[str, float],
    events: List[SoundEvent],
    features: AcousticFeatures,
    label_to_category: Dict[str, str],
) -> DescriptionResult:
    """
    Build a DescriptionResult from CLAP scores, detected events, and acoustic features.

    Parameters
    ----------
    file_name        : base file name (for logging)
    full_scores      : CLAP zero-shot scores for all candidate labels
    events           : temporal sound events from the event detector
    features         : acoustic features from the feature extractor
    label_to_category: controlled vocabulary mapping

    Returns
    -------
    DescriptionResult ready for output serialization
    """

    # --- 1. Identify top labels and primary sound -------------------------
    sorted_labels = sorted(full_scores.items(), key=lambda x: x[1], reverse=True)
    top_labels = [(l, s) for l, s in sorted_labels if s >= LOW_CONFIDENCE]

    if not top_labels:
        # Fallback: use highest score regardless of threshold
        top_labels = sorted_labels[:3] if sorted_labels else [("unidentified sound", 0.1)]

    primary_label, primary_score = top_labels[0]
    primary_category = label_to_category.get(primary_label, "unknown")

    # --- 2. Compute overall confidence -----------------------------------
    confidence = _compute_confidence(
        primary_score=primary_score,
        top_labels=top_labels,
        features=features,
        events=events,
    )

    # --- 3. Build tags ---------------------------------------------------
    tags = _build_tags(
        top_labels=top_labels,
        features=features,
        primary_category=primary_category,
        max_tags=MAX_TAGS,
    )

    # --- 4. Build sound events list (human-readable) --------------------
    if events:
        # Use temporal events if detected
        sound_event_labels = _deduplicate_preserve_order(
            [e.label for e in events if e.label != "background noise"]
        )
    else:
        # Fall back to top CLAP labels
        sound_event_labels = [l for l, _ in top_labels[:5]]

    # --- 5. Synthesize descriptions --------------------------------------
    short_description = _build_short_description(
        primary_label=primary_label,
        primary_category=primary_category,
        primary_score=primary_score,
        top_labels=top_labels,
        features=features,
        events=events,
    )

    detailed_description = _build_detailed_description(
        primary_label=primary_label,
        primary_category=primary_category,
        top_labels=top_labels,
        features=features,
        events=events,
        confidence=confidence,
    )

    return DescriptionResult(
        short_description=short_description,
        detailed_description=detailed_description,
        tags=tags,
        sound_events=sound_event_labels,
        confidence=round(confidence, 3),
        primary_category=primary_category,
        primary_label=primary_label,
    )


# ---------------------------------------------------------------------------
# Short description builder
# ---------------------------------------------------------------------------

def _build_short_description(
    primary_label: str,
    primary_category: str,
    primary_score: float,
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    events: List[SoundEvent],
) -> str:
    """
    Construct a single-sentence description of the audio.
    Uses temporal events when available; falls back to CLAP labels + acoustic cues.
    """

    # --- Case 1: temporal event sequence detected -----------------------
    if len(events) >= 2:
        notable = [e for e in events if e.label != "background noise"]
        if len(notable) >= 2:
            first = notable[0].label
            second = notable[1].label
            if len(notable) == 2:
                return _capitalize(f"{first}, followed by {second}.")
            else:
                remainder = ", then ".join(e.label for e in notable[2:])
                return _capitalize(f"{first}, followed by {second}, then {remainder}.")

    # --- Case 2: single dominant sound with acoustic detail --------------
    detail = _acoustic_detail(features, primary_category)
    hedge = _confidence_hedge(primary_score)

    secondary = _pick_secondary_label(top_labels, primary_label)

    if secondary:
        return _capitalize(f"{hedge}{primary_label} with {secondary}{detail}.")
    else:
        return _capitalize(f"{hedge}{primary_label}{detail}.")


# ---------------------------------------------------------------------------
# Detailed description builder
# ---------------------------------------------------------------------------

def _build_detailed_description(
    primary_label: str,
    primary_category: str,
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    events: List[SoundEvent],
    confidence: float,
) -> str:
    """
    Build a 2–4 sentence description covering:
    - What is heard (primary content)
    - Temporal structure (if events detected)
    - Acoustic character (derived from features)
    - A note about confidence if low
    """
    parts: List[str] = []

    # Sentence 1: primary sound identification
    hedge = _confidence_hedge(top_labels[0][1]) if top_labels else ""
    primary = top_labels[0][0] if top_labels else primary_label
    parts.append(_capitalize(f"The clip contains {hedge}{primary}."))

    # Sentence 2: secondary sounds (if any)
    secondary_sounds = [
        l for l, s in top_labels[1:4]
        if s >= LOW_CONFIDENCE and l != primary
    ]
    if secondary_sounds:
        if len(secondary_sounds) == 1:
            parts.append(
                _capitalize(f"Secondary sounds include {secondary_sounds[0]}.")
            )
        else:
            joined = ", ".join(secondary_sounds[:-1]) + f", and {secondary_sounds[-1]}"
            parts.append(_capitalize(f"Additional sounds include {joined}."))

    # Sentence 3: temporal structure from events
    if events:
        notable = [e for e in events if e.label != "background noise"]
        if len(notable) >= 2:
            sequence = " → ".join(e.label for e in notable[:4])
            parts.append(f"The temporal sequence is: {sequence}.")
        elif len(notable) == 1 and notable[0].duration < features.duration * 0.5:
            parts.append(
                f"The main sound event occurs around "
                f"{notable[0].start_time:.1f}–{notable[0].end_time:.1f}s "
                f"and fades out afterward."
            )

    # Sentence 4: acoustic character
    char = _describe_acoustic_character(features)
    if char:
        parts.append(_capitalize(char))

    # Sentence 5: low-confidence warning
    if confidence < MEDIUM_CONFIDENCE:
        parts.append(
            "Note: analysis confidence is low; description may be imprecise."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tag builder
# ---------------------------------------------------------------------------

def _build_tags(
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    primary_category: str,
    max_tags: int,
) -> List[str]:
    """
    Assemble a deduplicated tag list from CLAP labels + acoustic feature tags.
    CLAP-derived tags come first (ranked by score), then acoustic feature tags.
    """
    tags: List[str] = []

    # CLAP-derived tags
    for label, score in top_labels:
        if score >= LOW_CONFIDENCE and label not in tags:
            tags.append(label)

    # Primary category as a tag
    if primary_category and primary_category not in tags:
        tags.insert(0, primary_category)

    # Acoustic feature tags
    feature_tags = _acoustic_feature_tags(features)
    for tag in feature_tags:
        if tag not in tags:
            tags.append(tag)

    return tags[:max_tags]


def _acoustic_feature_tags(features: AcousticFeatures) -> List[str]:
    """Derive simple descriptor tags from acoustic features."""
    tags = []

    # Transient character
    if features.is_percussive:
        tags.append("percussive")
    if features.onset_strength_max > 10.0:
        tags.append("sharp transient")
    elif features.onset_strength_max > 5.0:
        tags.append("transient")

    # Spectral character
    if features.is_tonal:
        tags.append("tonal")
    if features.is_noisy:
        tags.append("broadband noise")
    if features.spectral_flatness_mean < 0.01:
        tags.append("pure tone")

    # Frequency character
    if features.is_low_frequency_heavy:
        tags.append("low frequency")
    if features.sub_bass_energy > 0.3:
        tags.append("deep bass")
    if features.air_energy > 0.1:
        tags.append("high frequency content")

    # Dynamics
    if features.dynamic_range_db > 30:
        tags.append("high dynamic range")
    elif features.dynamic_range_db < 5:
        tags.append("consistent level")

    # Duration
    if features.duration < 1.0:
        tags.append("short clip")
    elif features.duration > 30.0:
        tags.append("long clip")

    # Silence
    if features.silence_ratio > 0.5:
        tags.append("intermittent")
    elif features.silence_ratio < 0.05:
        tags.append("continuous")

    # Rhythm
    if 60 < features.tempo_bpm < 300 and features.is_percussive:
        tags.append("rhythmic")

    # Reverb proxy (high silence ratio + decaying onsets → likely reverberant space)
    if features.rms_std / (features.rms_mean + 1e-9) > 1.5 and features.num_onsets < 5:
        tags.append("reverb")

    return tags


# ---------------------------------------------------------------------------
# Acoustic character helpers
# ---------------------------------------------------------------------------

def _acoustic_detail(features: AcousticFeatures, category: str) -> str:
    """Return a brief acoustic qualifier phrase based on features."""
    parts = []

    if features.is_percussive and features.onset_strength_max > 8.0:
        parts.append(" with a sharp attack")
    elif features.is_percussive:
        parts.append(" with noticeable transients")

    if features.spectral_flatness_mean < 0.02 and features.is_tonal:
        parts.append(" and a tonal character")
    elif features.is_noisy:
        parts.append(" and broadband noise content")

    if features.sub_bass_energy > 0.4:
        parts.append(" and strong low-frequency energy")

    if features.silence_ratio > 0.6:
        parts.append(", occurring intermittently against silence")

    return "".join(parts) if parts else ""


def _describe_acoustic_character(features: AcousticFeatures) -> str:
    """Build an acoustic-character sentence for the detailed description."""
    observations = []

    if features.is_percussive and features.onset_strength_max > 8.0:
        observations.append("The sound has a sharp, transient attack")
    elif features.is_percussive:
        observations.append("The sound contains noticeable transient elements")

    if features.is_tonal and features.harmonic_ratio > 0.6:
        observations.append("strong harmonic content suggests a tonal source")
    elif features.is_noisy:
        observations.append("high spectral flatness indicates a broadband or noise-like source")

    if features.is_low_frequency_heavy:
        observations.append("energy is concentrated in the low-frequency range")

    if features.silence_ratio > 0.5:
        observations.append("significant portions of the clip are near-silent")
    elif features.silence_ratio < 0.05 and features.rms_std < 0.01:
        observations.append("the sound level is relatively uniform throughout")

    if not observations:
        return ""

    # Combine into one sentence
    if len(observations) == 1:
        return observations[0] + "."
    main = observations[0]
    rest = "; ".join(observations[1:])
    return f"{main}; {rest}."


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def _compute_confidence(
    primary_score: float,
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    events: List[SoundEvent],
) -> float:
    """
    Estimate overall description confidence from multiple signals.

    Components:
    - CLAP primary score (60% weight)
    - Label agreement (20%): gap between top and second-best label
    - Acoustic feature quality (20%): penalize very short/silent clips
    """
    # Component 1: CLAP primary score
    clap_conf = primary_score

    # Component 2: label separation (higher gap = more confident)
    if len(top_labels) >= 2:
        gap = top_labels[0][1] - top_labels[1][1]
        separation_conf = min(1.0, gap * 3.0)  # normalize: 0.33 gap → 1.0
    else:
        separation_conf = 1.0

    # Component 3: acoustic quality
    quality_conf = 1.0
    if features.duration < 0.5:
        quality_conf *= 0.5   # very short clips are unreliable
    if features.silence_ratio > 0.8:
        quality_conf *= 0.6   # mostly silent
    if features.rms_max < 1e-4:
        quality_conf *= 0.3   # effectively inaudible

    # Weighted combination
    confidence = (
        0.60 * clap_conf
        + 0.20 * separation_conf
        + 0.20 * quality_conf
    )
    return min(1.0, max(0.0, confidence))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pick_secondary_label(
    top_labels: List[Tuple[str, float]],
    primary_label: str,
) -> Optional[str]:
    """Return the top secondary label if it is above LOW_CONFIDENCE."""
    for label, score in top_labels[1:]:
        if label != primary_label and score >= LOW_CONFIDENCE:
            return label
    return None


def _confidence_hedge(score: float) -> str:
    """Return a hedging prefix phrase for low-confidence detections."""
    if score >= HIGH_CONFIDENCE:
        return ""
    elif score >= MEDIUM_CONFIDENCE:
        return "what appears to be "
    else:
        return "possibly "


def _capitalize(s: str) -> str:
    """Capitalize the first letter of a string."""
    if not s:
        return s
    return s[0].upper() + s[1:]


def _deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving insertion order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
