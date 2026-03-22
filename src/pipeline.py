"""
pipeline.py
-----------
Main analysis pipeline. Orchestrates all Phase 1 components:

  AudioFile → Features → CLAP Tags → Events → Description → Record

Single entry point for both the single-file CLI and the batch processor.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .ingestion.audio_loader import AudioFile, load_audio
from .models.clap_tagger import CLAPTagger, CLAP_SAMPLE_RATE
from .models.label_cache import LabelEmbeddingCache
from .analysis.feature_extractor import AcousticFeatures, extract_features
from .analysis.event_detector import (
    SoundEvent,
    detect_events,
    detect_events_from_full_clip,
)
from .analysis.description_synthesizer import synthesize_description, DescriptionResult
from .output.schema import (
    AudioAnalysisRecord, AudioMetadata, AcousticSummary,
    build_suggested_filename,
)

logger = logging.getLogger(__name__)


class AudioAnalysisPipeline:
    """
    Orchestrates the full Phase 1 audio analysis pipeline.

    Typical usage
    -------------
        pipeline = AudioAnalysisPipeline(config)
        pipeline.load_model()
        record = pipeline.analyze_file("path/to/clip.wav")

    Parameters
    ----------
    config : dict with keys from config/config.yaml
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.tagger: Optional[CLAPTagger] = None
        self.cache: Optional[LabelEmbeddingCache] = None

        # Load vocabulary from config (UCS lookups)
        self.candidate_labels: List[str] = config.get("candidate_labels", [])
        self.label_to_category: Dict[str, str] = config.get("label_to_category", {})
        self.label_to_subcategory: Dict[str, str] = config.get("label_to_subcategory", {})
        self.label_to_cat_id: Dict[str, str] = config.get("label_to_cat_id", {})
        self.label_to_category_full: Dict[str, str] = config.get("label_to_category_full", {})

        # UCS identity fields
        self.ucs_creator_id: str = config.get("ucs_creator_id", "UNKNOWN")
        self.ucs_source_id: str = config.get("ucs_source_id", "NONE")
        self.ucs_user_data: str = config.get("ucs_user_data", "")

        # Pipeline settings
        self.target_sr: int = config.get("target_sr", CLAP_SAMPLE_RATE)
        self.window_seconds: float = config.get("window_seconds", 2.0)
        self.hop_seconds: float = config.get("hop_seconds", 0.5)
        self.min_confidence: float = config.get("min_confidence", 0.25)
        self.use_windowed_analysis: bool = config.get("use_windowed_analysis", True)
        self.windowed_min_duration: float = config.get("windowed_min_duration", 2.0)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the CLAP model and the label embedding cache.

        If the cache file does not exist (or is stale) it is built
        automatically on first run — this adds a one-time overhead of
        ~30-60 s but makes every subsequent run significantly faster.
        """
        model_id = self.config.get("model_id", "laion/larger_clap_general")
        device = self.config.get("device", None)
        fp16 = self.config.get("fp16", True)

        self.tagger = CLAPTagger(model_id=model_id, device=device, fp16=fp16)
        self.tagger.load()

        # --- Label embedding cache -------------------------------------
        cache_path = self.config.get("label_cache_path")
        if cache_path:
            self.cache = LabelEmbeddingCache(cache_path)
            n_labels = len(self.candidate_labels)

            if self.cache.is_valid(expected_label_count=n_labels):
                self.cache.load()
                logger.info("Label cache loaded (%d labels).", n_labels)
            else:
                logger.info(
                    "Label cache missing or stale — building now "
                    "(one-time cost, ~30-60 s)…"
                )
                self.cache.build(
                    tagger=self.tagger,
                    candidate_labels=self.candidate_labels,
                    label_to_category=self.label_to_category,
                    label_to_subcategory=self.label_to_subcategory,
                    label_to_cat_id=self.label_to_cat_id,
                    label_to_category_full=self.label_to_category_full,
                )
                logger.info("Label cache built and saved to %s.", cache_path)
        else:
            logger.warning(
                "label_cache_path not set in config — falling back to "
                "slow per-file label encoding. Add 'label_cache_path' to "
                "config.yaml to enable the embedding cache."
            )

    # ------------------------------------------------------------------
    # Single-file analysis
    # ------------------------------------------------------------------

    def analyze_file(
        self,
        path: str | Path,
        audio_file: Optional[AudioFile] = None,
    ) -> AudioAnalysisRecord:
        """
        Analyze a single audio file and return an AudioAnalysisRecord.

        Parameters
        ----------
        path       : path to the audio file
        audio_file : pre-loaded AudioFile (skip re-loading if already loaded)

        Returns
        -------
        AudioAnalysisRecord — complete analysis result
        """
        if self.tagger is None:
            raise RuntimeError(
                "Model not loaded. Call pipeline.load_model() first."
            )

        t0 = time.perf_counter()

        # --- 1. Load audio -----------------------------------------------
        if audio_file is None:
            audio_file = load_audio(str(path), target_sr=self.target_sr)
        af = audio_file

        logger.info("Analyzing: %s (%.2fs)", af.file_name, af.duration)

        # --- 2. Extract acoustic features --------------------------------
        features: AcousticFeatures = extract_features(af.waveform, af.sample_rate)

        # --- 3. Full-clip CLAP classification ----------------------------
        if self.cache is not None:
            # Fast path: embed audio once, score against cached text matrix
            audio_embed = self.tagger.embed_audio(af.waveform, af.sample_rate)
            full_scores: Dict[str, float] = self.cache.classify(audio_embed)
        else:
            # Legacy path: re-encode all labels on every file (slow)
            full_scores = self.tagger.classify(
                waveform=af.waveform,
                sr=af.sample_rate,
                candidate_labels=self.candidate_labels,
            )

        logger.debug(
            "Top label: %s (%.3f)",
            max(full_scores, key=full_scores.get),
            max(full_scores.values()),
        )

        # --- 4. Event detection (sliding window or single-clip fallback) -
        events: List[SoundEvent] = self._detect_events(
            af, full_scores, features, audio_embed if self.cache else None
        )

        # --- 5. Synthesize description -----------------------------------
        # Prefer lookup dicts from the cache (they match what was scored)
        l2cat   = self.cache.label_to_category      if self.cache else self.label_to_category
        l2sub   = self.cache.label_to_subcategory   if self.cache else self.label_to_subcategory
        l2catid = self.cache.label_to_cat_id        if self.cache else self.label_to_cat_id
        l2full  = self.cache.label_to_category_full if self.cache else self.label_to_category_full

        description: DescriptionResult = synthesize_description(
            file_name=af.file_name,
            full_scores=full_scores,
            events=events,
            features=features,
            label_to_category=l2cat,
            label_to_subcategory=l2sub,
            label_to_cat_id=l2catid,
            label_to_category_full=l2full,
        )

        # --- 6. Assemble the output record -------------------------------
        record = self._assemble_record(af, features, full_scores, events, description)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Done: %s | conf=%.2f | %.1fs elapsed",
            af.file_name,
            record.confidence,
            elapsed,
        )

        return record

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        paths: List[str | Path],
        skip_errors: bool = True,
        progress_callback=None,
    ) -> List[AudioAnalysisRecord]:
        """
        Analyze a list of audio files.

        Parameters
        ----------
        paths            : list of paths to audio files
        skip_errors      : if True, log errors and continue; else raise
        progress_callback: optional callable(current, total, file_name)

        Returns
        -------
        List of AudioAnalysisRecord (only successful results)
        """
        if self.tagger is None:
            raise RuntimeError("Call pipeline.load_model() first.")

        results: List[AudioAnalysisRecord] = []
        total = len(paths)

        for i, path in enumerate(paths, start=1):
            if progress_callback:
                progress_callback(i, total, Path(path).name)
            try:
                record = self.analyze_file(path)
                results.append(record)
            except Exception as exc:
                if skip_errors:
                    logger.error("Failed '%s': %s", Path(path).name, exc)
                else:
                    raise

        logger.info(
            "Batch complete: %d/%d files analyzed successfully.",
            len(results),
            total,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_events(
        self,
        af: AudioFile,
        full_scores: Dict[str, float],
        features: AcousticFeatures,
        audio_embed=None,  # pre-computed embedding when cache is active
    ) -> List[SoundEvent]:
        """Choose windowed or single-clip event detection based on duration."""
        if (
            self.use_windowed_analysis
            and af.duration >= self.windowed_min_duration
        ):
            try:
                events = detect_events(
                    waveform=af.waveform,
                    sr=af.sample_rate,
                    tagger=self.tagger,
                    candidate_labels=self.candidate_labels,
                    label_to_category=self.label_to_category,
                    window_seconds=self.window_seconds,
                    hop_seconds=self.hop_seconds,
                    min_confidence=self.min_confidence,
                    cache=self.cache,  # None → legacy path; set → fast path
                )
                return events
            except Exception as exc:
                logger.warning(
                    "Windowed event detection failed for '%s': %s. Using full-clip fallback.",
                    af.file_name,
                    exc,
                )

        # Fallback: single-clip event list from full-clip CLAP scores
        return detect_events_from_full_clip(
            full_scores=full_scores,
            label_to_category=self.label_to_category,
            duration=af.duration,
            min_confidence=self.min_confidence,
        )

    def _assemble_record(
        self,
        af: AudioFile,
        features: AcousticFeatures,
        full_scores: Dict[str, float],
        events: List[SoundEvent],
        description: DescriptionResult,
    ) -> AudioAnalysisRecord:
        """Build the final AudioAnalysisRecord from all pipeline outputs."""

        # Determine dominant frequency band
        band_scores = {
            "sub_bass": features.sub_bass_energy,
            "bass": features.bass_energy,
            "low_mid": features.low_mid_energy,
            "mid": features.mid_energy,
            "high": features.high_energy,
            "air": features.air_energy,
        }
        dominant_band = max(band_scores, key=band_scores.get)

        metadata = AudioMetadata(
            file_name=af.file_name,
            file_path=str(af.path),
            format=af.format,
            duration_seconds=round(af.duration, 4),
            sample_rate_hz=af.sample_rate,
            original_sample_rate_hz=af.original_sample_rate,
            num_channels=af.num_channels,
            num_samples=af.num_samples,
        )

        acoustic_summary = AcousticSummary(
            rms_mean=round(features.rms_mean, 6),
            spectral_centroid_mean_hz=round(features.spectral_centroid_mean, 1),
            spectral_flatness_mean=round(features.spectral_flatness_mean, 5),
            is_percussive=features.is_percussive,
            is_tonal=features.is_tonal,
            is_noisy=features.is_noisy,
            silence_ratio=round(features.silence_ratio, 3),
            dynamic_range_db=round(features.dynamic_range_db, 2),
            dominant_frequency_band=dominant_band,
        )

        # Top-10 CLAP scores for the record
        top_labels = dict(
            sorted(full_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Build UCS suggested filename
        suggested_filename = build_suggested_filename(
            cat_id=description.cat_id,
            fx_name=description.fx_name,
            creator_id=self.ucs_creator_id,
            source_id=self.ucs_source_id,
            user_data=self.ucs_user_data,
        )

        return AudioAnalysisRecord(
            file_name=af.file_name,
            # UCS core
            category=description.category,
            subcategory=description.subcategory,
            cat_id=description.cat_id,
            category_full=description.category_full,
            fx_name=description.fx_name,
            description=description.description,
            keywords=description.keywords,
            sound_events=description.sound_events,
            confidence=description.confidence,
            # UCS identity
            creator_id=self.ucs_creator_id,
            source_id=self.ucs_source_id,
            user_data=self.ucs_user_data,
            suggested_filename=suggested_filename,
            # Internal
            top_labels=top_labels,
            metadata=metadata,
            acoustic_summary=acoustic_summary,
        )
