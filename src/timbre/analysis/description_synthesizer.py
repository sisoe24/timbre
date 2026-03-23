"""
description_synthesizer.py
--------------------------
Converts structured analysis results (CLAP tags, acoustic features, events)
into UCS-compliant metadata fields for audio cataloging.

UCS output fields
-----------------
- fx_name       : Brief title ~25 chars (UCS FXName)
- description   : Long-form description (UCS Description)
- keywords      : Search terms (UCS Keywords)
- sound_events  : Ordered temporal event labels
- category      : UCS Category (e.g. "IMPACTS")
- subcategory   : UCS SubCategory (e.g. "METAL")
- cat_id        : UCS CatID (e.g. "IMPMtl")
- category_full : "IMPACTS-METAL"
- confidence    : Overall confidence score

Design principles
-----------------
- NO emotion, genre, or cinematic interpretation
- NO hallucination — only describe what CLAP and features confirm
- Conservative wording when confidence is low
- FXName is title-style, ~25 chars, plain language
- Description is 2-4 sentences, factual and acoustic-detail-rich
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .event_detector import SoundEvent
from .feature_extractor import AcousticFeatures

logger = logging.getLogger(__name__)

# Confidence thresholds for description wording
HIGH_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.45
LOW_CONFIDENCE = 0.25

# Max keywords to include in output
MAX_KEYWORDS = 12


@dataclass
class DescriptionResult:
    """
    UCS-compliant output from the description synthesizer.
    Ready for serialization to JSON/CSV/Markdown.
    """

    fx_name: str               # UCS FXName (~25 chars, title style)
    description: str           # UCS Description (long-form)
    keywords: List[str]        # UCS Keywords (search terms)
    sound_events: List[str]    # ordered event labels
    confidence: float
    category: str              # UCS Category
    subcategory: str           # UCS SubCategory
    cat_id: str                # UCS CatID
    category_full: str         # "CATEGORY-SUBCATEGORY"


def synthesize_description(
    file_name: str,
    full_scores: Dict[str, float],
    events: List[SoundEvent],
    features: AcousticFeatures,
    label_to_category: Dict[str, str],
    label_to_subcategory: Dict[str, str] = None,
    label_to_cat_id: Dict[str, str] = None,
    label_to_category_full: Dict[str, str] = None,
) -> DescriptionResult:
    """
    Build a UCS DescriptionResult from CLAP scores, detected events,
    and acoustic features.

    Parameters
    ----------
    file_name              : base file name (for logging)
    full_scores            : CLAP zero-shot scores for all candidate labels
    events                 : temporal sound events from the event detector
    features               : acoustic features from the feature extractor
    label_to_category      : label → UCS Category
    label_to_subcategory   : label → UCS SubCategory
    label_to_cat_id        : label → UCS CatID
    label_to_category_full : label → "CATEGORY-SUBCATEGORY"

    Returns
    -------
    DescriptionResult ready for output serialization
    """
    # Provide empty dicts as defaults for optional maps
    label_to_subcategory = label_to_subcategory or {}
    label_to_cat_id = label_to_cat_id or {}
    label_to_category_full = label_to_category_full or {}

    # --- 1. Identify top labels and primary sound -------------------------
    sorted_labels = sorted(full_scores.items(), key=lambda x: x[1], reverse=True)
    top_labels = [(l, s) for l, s in sorted_labels if s >= LOW_CONFIDENCE]

    if not top_labels:
        # Fallback: use highest score regardless of threshold
        top_labels = sorted_labels[:3] if sorted_labels else [('unidentified sound', 0.1)]

    primary_label, primary_score = top_labels[0]

    # UCS classification for primary label
    category = label_to_category.get(primary_label, 'BACKGROUND')
    subcategory = label_to_subcategory.get(primary_label, 'NOISE')
    cat_id = label_to_cat_id.get(primary_label, 'BGNDNs')
    category_full = label_to_category_full.get(
        primary_label, f"{category}-{subcategory}"
    )

    # --- 2. Compute overall confidence -----------------------------------
    confidence = _compute_confidence(
        primary_score=primary_score,
        top_labels=top_labels,
        features=features,
        events=events,
    )

    # --- 3. Build keywords -----------------------------------------------
    keywords = _build_keywords(
        top_labels=top_labels,
        features=features,
        category=category,
        subcategory=subcategory,
        max_keywords=MAX_KEYWORDS,
    )

    # --- 4. Build sound events list (human-readable) --------------------
    if events:
        sound_event_labels = _deduplicate_preserve_order(
            [e.label for e in events if e.label != 'background noise']
        )
    else:
        sound_event_labels = [l for l, _ in top_labels[:5]]

    # --- 5. Synthesize UCS FXName (title, ~25 chars) --------------------
    fx_name = _build_fx_name(
        primary_label=primary_label,
        primary_score=primary_score,
        top_labels=top_labels,
        features=features,
        events=events,
    )

    # --- 6. Synthesize UCS Description (long-form) ----------------------
    description = _build_description(
        primary_label=primary_label,
        category=category,
        top_labels=top_labels,
        features=features,
        events=events,
        confidence=confidence,
    )

    return DescriptionResult(
        fx_name=fx_name,
        description=description,
        keywords=keywords,
        sound_events=sound_event_labels,
        confidence=round(confidence, 3),
        category=category,
        subcategory=subcategory,
        cat_id=cat_id,
        category_full=category_full,
    )


# ---------------------------------------------------------------------------
# FXName builder (UCS: brief title, ~25 chars)
# ---------------------------------------------------------------------------

def _build_fx_name(
    primary_label: str,
    primary_score: float,
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    events: List[SoundEvent],
) -> str:
    """
    Construct a UCS FXName: a brief title ~25 characters.

    Strategy
    --------
    - Use temporal event sequence if multiple distinct events exist
    - Otherwise: primary label + a key acoustic qualifier
    - Always title-case, no trailing punctuation
    - Aim for ≤25 chars; hard-trim at 50
    """

    # Case 1: temporal sequence
    if len(events) >= 2:
        notable = [e for e in events if e.label != 'background noise']
        if len(notable) >= 2:
            first = _title(notable[0].label)
            second = _title(notable[1].label)
            name = f"{first} to {second}"
            return name[:50]

    # Case 2: primary + optional acoustic qualifier
    base = _title(primary_label)
    qualifier = _fx_qualifier(features, primary_score)

    if qualifier:
        name = f"{base} {qualifier}"
    else:
        name = base

    return name[:50]


def _fx_qualifier(features: AcousticFeatures, score: float) -> str:
    """Return a short acoustic qualifier word for the FXName."""
    if features.is_percussive and features.onset_strength_max > 8.0:
        return 'Hard'
    if features.is_percussive:
        return 'Soft'
    if features.duration < 1.0:
        return 'Short'
    if features.duration > 30.0:
        return 'Long'
    if features.silence_ratio > 0.5:
        return 'Intermittent'
    if features.sub_bass_energy > 0.4:
        return 'Deep'
    return ''


# ---------------------------------------------------------------------------
# Description builder (UCS: long-form, 2–4 sentences)
# ---------------------------------------------------------------------------

def _build_description(
    primary_label: str,
    category: str,
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    events: List[SoundEvent],
    confidence: float,
) -> str:
    """
    Build a 2–4 sentence UCS Description covering:
    - What is heard (primary content)
    - Temporal structure (if events detected)
    - Acoustic character (derived from features)
    - A note about confidence if low
    """
    parts: List[str] = []

    # Sentence 1: primary sound identification
    hedge = _confidence_hedge(top_labels[0][1]) if top_labels else ''
    primary = top_labels[0][0] if top_labels else primary_label
    parts.append(_capitalize(f"The clip contains {hedge}{primary}."))

    # Sentence 2: secondary sounds (if any)
    secondary_sounds = [
        l for l, s in top_labels[1:4]
        if s >= LOW_CONFIDENCE and l != primary
    ]
    if secondary_sounds:
        if len(secondary_sounds) == 1:
            parts.append(_capitalize(f"Secondary sounds include {secondary_sounds[0]}."))
        else:
            joined = ', '.join(secondary_sounds[:-1]) + f", and {secondary_sounds[-1]}"
            parts.append(_capitalize(f"Additional sounds include {joined}."))

    # Sentence 3: temporal structure from events
    if events:
        notable = [e for e in events if e.label != 'background noise']
        if len(notable) >= 2:
            sequence = ' → '.join(e.label for e in notable[:4])
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
            'Note: analysis confidence is low; description may be imprecise.'
        )

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Keywords builder (UCS Keywords field)
# ---------------------------------------------------------------------------

def _build_keywords(
    top_labels: List[Tuple[str, float]],
    features: AcousticFeatures,
    category: str,
    subcategory: str,
    max_keywords: int,
) -> List[str]:
    """
    Assemble a deduplicated keyword list from:
    1. UCS Category and SubCategory (always first)
    2. CLAP-derived labels (ranked by score)
    3. Acoustic feature keywords
    """
    keywords: List[str] = []

    # UCS category terms first
    if category and category not in keywords:
        keywords.append(category.lower())
    if subcategory and subcategory.lower() not in keywords:
        keywords.append(subcategory.lower())

    # CLAP-derived keywords
    for label, score in top_labels:
        if score >= LOW_CONFIDENCE and label not in keywords:
            keywords.append(label)

    # Acoustic feature keywords
    for kw in _acoustic_feature_keywords(features):
        if kw not in keywords:
            keywords.append(kw)

    return keywords[:max_keywords]


def _acoustic_feature_keywords(features: AcousticFeatures) -> List[str]:
    """Derive descriptor keywords from acoustic features."""
    kws = []

    if features.is_percussive:
        kws.append('percussive')
    if features.onset_strength_max > 10.0:
        kws.append('sharp transient')
    elif features.onset_strength_max > 5.0:
        kws.append('transient')
    if features.is_tonal:
        kws.append('tonal')
    if features.is_noisy:
        kws.append('broadband noise')
    if features.spectral_flatness_mean < 0.01:
        kws.append('pure tone')
    if features.is_low_frequency_heavy:
        kws.append('low frequency')
    if features.sub_bass_energy > 0.3:
        kws.append('deep bass')
    if features.air_energy > 0.1:
        kws.append('high frequency content')
    if features.dynamic_range_db > 30:
        kws.append('high dynamic range')
    if features.duration < 1.0:
        kws.append('short')
    elif features.duration > 30.0:
        kws.append('long')
    if features.silence_ratio > 0.5:
        kws.append('intermittent')
    elif features.silence_ratio < 0.05:
        kws.append('continuous')
    if 60 < features.tempo_bpm < 300 and features.is_percussive:
        kws.append('rhythmic')
    if features.rms_std / (features.rms_mean + 1e-9) > 1.5 and features.num_onsets < 5:
        kws.append('reverberant')

    return kws


# ---------------------------------------------------------------------------
# Acoustic character helpers
# ---------------------------------------------------------------------------

def _describe_acoustic_character(features: AcousticFeatures) -> str:
    """Build an acoustic-character sentence for the description."""
    observations = []

    if features.is_percussive and features.onset_strength_max > 8.0:
        observations.append('The sound has a sharp, transient attack')
    elif features.is_percussive:
        observations.append('The sound contains noticeable transient elements')

    if features.is_tonal and features.harmonic_ratio > 0.6:
        observations.append('strong harmonic content suggests a tonal source')
    elif features.is_noisy:
        observations.append('high spectral flatness indicates a broadband or noise-like source')

    if features.is_low_frequency_heavy:
        observations.append('energy is concentrated in the low-frequency range')

    if features.silence_ratio > 0.5:
        observations.append('significant portions of the clip are near-silent')
    elif features.silence_ratio < 0.05 and features.rms_std < 0.01:
        observations.append('the sound level is relatively uniform throughout')

    if not observations:
        return ''

    if len(observations) == 1:
        return observations[0] + '.'
    main = observations[0]
    rest = '; '.join(observations[1:])
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
    clap_conf = primary_score

    if len(top_labels) >= 2:
        gap = top_labels[0][1] - top_labels[1][1]
        separation_conf = min(1.0, gap * 3.0)
    else:
        separation_conf = 1.0

    quality_conf = 1.0
    if features.duration < 0.5:
        quality_conf *= 0.5
    if features.silence_ratio > 0.8:
        quality_conf *= 0.6
    if features.rms_max < 1e-4:
        quality_conf *= 0.3

    confidence = (
        0.60 * clap_conf
        + 0.20 * separation_conf
        + 0.20 * quality_conf
    )
    return min(1.0, max(0.0, confidence))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _confidence_hedge(score: float) -> str:
    """Return a hedging prefix phrase for low-confidence detections."""
    if score >= HIGH_CONFIDENCE:
        return ''
    elif score >= MEDIUM_CONFIDENCE:
        return 'what appears to be '
    else:
        return 'possibly '


def _title(s: str) -> str:
    """Title-case a label string."""
    return s.title()


def _capitalize(s: str) -> str:
    """Capitalize only the first letter of a string."""
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
