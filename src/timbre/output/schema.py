"""
schema.py
---------
Pydantic data model for a single audio analysis record.

All output fields conform to the UCS (Universal Category System) v8.2
naming conventions. The primary output JSON matches the UCS metadata schema
used in professional sound effects libraries.

Reference: universalcategorysystem.com
"""

from __future__ import annotations

import re
import datetime
from typing import Dict, List

from pydantic import Field, BaseModel, field_validator


class AudioMetadata(BaseModel):
    """Low-level file and signal metadata."""

    file_name: str
    file_path: str
    format: str
    duration_seconds: float = Field(..., ge=0.0)
    sample_rate_hz: int = Field(..., gt=0)
    original_sample_rate_hz: int = Field(..., ge=0)
    num_channels: int = Field(..., ge=1)
    num_samples: int = Field(..., ge=0)


class AcousticSummary(BaseModel):
    """Key acoustic features (subset exposed in the record)."""

    rms_mean: float
    spectral_centroid_mean_hz: float
    spectral_flatness_mean: float
    is_percussive: bool
    is_tonal: bool
    is_noisy: bool
    silence_ratio: float
    dynamic_range_db: float
    dominant_frequency_band: str   # sub_bass / bass / low_mid / mid / high / air


class AnalysisProvenance(BaseModel):
    """Inputs and cache provenance for a rendered analysis record."""

    model_id: str
    config_path: str
    vocab_path: str
    vocab_sha256: str
    analysis_elapsed_seconds: float = Field(..., ge=0.0)
    profile_name: str = 'default'
    profile_fingerprint: str | None = None
    cache_path: str | None = None
    cache_fingerprint: str | None = None


class AudioAnalysisRecord(BaseModel):
    """
    Complete UCS-compliant analysis record for one audio file.

    Core UCS fields
    ---------------
    category        : UCS Category (e.g. "IMPACTS")
    subcategory     : UCS SubCategory (e.g. "METAL")
    cat_id          : UCS CatID abbreviation (e.g. "IMPMtl")
    category_full   : Full dash-separated form (e.g. "IMPACTS-METAL")
    fx_name         : Brief title ~25 chars (replaces short_description)
    description     : Long-form description (replaces detailed_description)
    keywords        : Search keywords (replaces tags)
    sound_events    : Ordered temporal event labels
    confidence      : Overall confidence score [0.0–1.0]

    UCS identity fields (from config.yaml)
    ---------------------------------------
    creator_id      : Who made/recorded the sound (e.g. "TN", "ACME")
    source_id       : Source library identifier (e.g. "NONE", "MyLib")
    user_data       : Free field: mic type, perspective, unique ID

    Suggested filename
    ------------------
    suggested_filename : UCS-compliant filename suggestion
                         Pattern: CatID_FXName_CreatorID_SourceID[_UserData]
    """

    # ---- Identity -------------------------------------------------------
    file_name: str
    analyzed_at: str = Field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace('+00:00', 'Z')
    )

    # ---- UCS core fields ------------------------------------------------
    category: str              # UCS Category (e.g. "IMPACTS")
    subcategory: str           # UCS SubCategory (e.g. "METAL")
    cat_id: str                # UCS CatID (e.g. "IMPMtl")
    category_full: str         # "IMPACTS-METAL"
    fx_name: str               # ~25 char title (was short_description)
    description: str           # Long-form description (was detailed_description)
    keywords: List[str] = Field(default_factory=list)   # search terms (was tags)
    sound_events: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    # ---- UCS identity ---------------------------------------------------
    creator_id: str = 'UNKNOWN'
    source_id: str = 'NONE'
    user_data: str = ''

    # ---- Suggested UCS filename -----------------------------------------
    suggested_filename: str = ''

    # ---- Classification detail (internal) --------------------------------
    top_labels: Dict[str, float] = Field(default_factory=dict)

    # ---- Metadata -------------------------------------------------------
    metadata: AudioMetadata
    acoustic_summary: AcousticSummary
    analysis_provenance: AnalysisProvenance

    @field_validator('confidence')
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)

    def to_brief_dict(self) -> dict:
        """
        Return the UCS-format output record.
        This is the primary JSON output format.
        """
        return {
            'file_name': self.file_name,
            'category': self.category,
            'subcategory': self.subcategory,
            'cat_id': self.cat_id,
            'category_full': self.category_full,
            'fx_name': self.fx_name,
            'description': self.description,
            'keywords': self.keywords,
            'sound_events': self.sound_events,
            'confidence': self.confidence,
            'creator_id': self.creator_id,
            'source_id': self.source_id,
            'user_data': self.user_data,
            'suggested_filename': self.suggested_filename,
            'analysis_provenance': self.analysis_provenance.model_dump(),
        }

    def to_full_dict(self) -> dict:
        """Return complete record as a dict (includes metadata + acoustics)."""
        return self.model_dump()


def build_suggested_filename(
    cat_id: str,
    fx_name: str,
    creator_id: str,
    source_id: str,
    user_data: str = '',
) -> str:
    """
    Construct a UCS-compliant suggested filename (without extension).

    Pattern: CatID_FXName_CreatorID_SourceID[_UserData]

    The FXName is sanitized: special characters replaced with spaces,
    multiple spaces collapsed, leading/trailing spaces stripped.
    Max FXName length in filename: 50 chars (generous limit for readability).
    """
    # Sanitize FXName — remove characters that are problematic in filenames
    safe_fx = re.sub(r'[<>:"/\\|?*]', ' ', fx_name)
    safe_fx = re.sub(r'\s+', ' ', safe_fx).strip()
    safe_fx = safe_fx[:50]  # hard cap

    parts = [cat_id, safe_fx, creator_id, source_id]
    if user_data:
        parts.append(user_data.strip())

    return '_'.join(parts)
