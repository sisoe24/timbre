"""
schema.py
---------
Pydantic data model for a single audio analysis record.

This is the canonical output format for Phase 1.
All fields map directly to the JSON spec defined in the project brief.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import datetime


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


class AudioAnalysisRecord(BaseModel):
    """
    Complete analysis record for one audio file.

    This is the primary output unit. A list of these records forms a catalog.
    """

    # ---- Identity -------------------------------------------------------
    file_name: str
    analyzed_at: str = Field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )

    # ---- Core output (matches project spec JSON) -----------------------
    short_description: str
    detailed_description: str
    tags: List[str] = Field(default_factory=list)
    sound_events: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    # ---- Classification -------------------------------------------------
    primary_label: str
    primary_category: str
    top_labels: Dict[str, float] = Field(default_factory=dict)

    # ---- Metadata -------------------------------------------------------
    metadata: AudioMetadata
    acoustic_summary: AcousticSummary

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)

    def to_brief_dict(self) -> dict:
        """
        Return only the core fields (project spec format).
        Used for JSON/CSV output.
        """
        return {
            "file_name": self.file_name,
            "short_description": self.short_description,
            "detailed_description": self.detailed_description,
            "tags": self.tags,
            "sound_events": self.sound_events,
            "confidence": self.confidence,
        }

    def to_full_dict(self) -> dict:
        """Return complete record as a dict (includes metadata + acoustics)."""
        return self.model_dump()
