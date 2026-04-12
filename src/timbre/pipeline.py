"""
pipeline.py
-----------
Evidence-first audio analysis pipeline:

  AudioFile → Features → CLAP Evidence → Analyze (LLM) → Record
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .output.schema import (AudioMetadata, EvidenceEvent, LLMProvenance,
                            AnalysisResult, EvidenceBundle, PromptEvidence,
                            AcousticSummary, AnalysisProvenance,
                            MappingDiagnostics, AudioAnalysisRecord,
                            build_suggested_filename)
from .analysis.analyzer import analyze
from .models.clap_tagger import CLAP_SAMPLE_RATE, CLAPTagger
from .models.label_cache import LabelEmbeddingCache, build_cache_metadata
from .analysis.prompt_bank import build_descriptive_prompt_bank
from .ingestion.audio_loader import AudioFile, load_audio
from .analysis.event_detector import (SoundEvent, detect_events,
                                      detect_events_from_full_clip)
from .analysis.feature_extractor import AcousticFeatures, extract_features

logger = logging.getLogger(__name__)


class AudioAnalysisPipeline:
    """Orchestrates the full evidence-first audio analysis pipeline."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.tagger: Optional[CLAPTagger] = None
        self.cache: Optional[LabelEmbeddingCache] = None

        self.candidate_labels: List[str] = config.get('candidate_labels', [])
        self.label_to_category: Dict[str, str] = config.get('label_to_category', {})
        self.label_to_subcategory: Dict[str, str] = config.get('label_to_subcategory', {})
        self.label_to_cat_id: Dict[str, str] = config.get('label_to_cat_id', {})
        self.label_to_category_full: Dict[str, str] = config.get('label_to_category_full', {})
        self.taxonomy: Dict[str, Dict[str, Dict[str, str]]] = config.get('taxonomy', {})
        self.prompt_entries = build_descriptive_prompt_bank(
            candidate_labels=self.candidate_labels,
            label_to_category=self.label_to_category,
            label_to_subcategory=self.label_to_subcategory,
            label_to_cat_id=self.label_to_cat_id,
            label_to_category_full=self.label_to_category_full,
        )
        self.prompt_label_index = {entry.prompt: entry for entry in self.prompt_entries}

        self.ucs_creator_id: str = config.get('ucs_creator_id', 'UNKNOWN')
        self.ucs_source_id: str = config.get('ucs_source_id', 'NONE')
        self.ucs_user_data: str = config.get('ucs_user_data', '')

        self.target_sr: int = config.get('target_sr', CLAP_SAMPLE_RATE)
        self.window_seconds: float = config.get('window_seconds', 2.0)
        self.hop_seconds: float = config.get('hop_seconds', 0.5)
        self.min_confidence: float = config.get('min_confidence', 0.25)
        self.use_windowed_analysis: bool = config.get('use_windowed_analysis', True)
        self.windowed_min_duration: float = config.get('windowed_min_duration', 2.0)
        self.top_k_categories: int = config.get('top_k_categories', 5)

    def load_model(self) -> None:
        """Load CLAP and the combined label/prompt embedding cache."""
        model_id = self.config.get('model_id', 'laion/larger_clap_general')
        device = self.config.get('device', None)
        fp16 = self.config.get('fp16', True)

        self.tagger = CLAPTagger(model_id=model_id, device=device, fp16=fp16)
        self.tagger.load()

        cache_path = self.config.get('label_cache_path')
        if cache_path:
            self.cache = LabelEmbeddingCache(cache_path)
            expected_metadata = {
                'model_id': self.config.get('model_id'),
                'vocab_sha256': self.config.get('vocab_sha256'),
                'cache_fingerprint': self.config.get('cache_fingerprint'),
                'prompt_bank_version': self.config.get('prompt_bank_version'),
            }
            if self.cache.is_valid(
                expected_label_count=len(self.candidate_labels),
                expected_metadata=expected_metadata,
            ):
                self.cache.load()
                self.prompt_label_index = {
                    entry['prompt']: entry for entry in self.cache.prompt_entries
                }
                logger.info('Label cache loaded (%d labels).', len(self.candidate_labels))
            else:
                logger.info('Label cache missing or stale; rebuilding.')
                self.cache.build(
                    tagger=self.tagger,
                    candidate_labels=self.candidate_labels,
                    label_to_category=self.label_to_category,
                    label_to_subcategory=self.label_to_subcategory,
                    label_to_cat_id=self.label_to_cat_id,
                    label_to_category_full=self.label_to_category_full,
                    metadata=build_cache_metadata(self.config, self.candidate_labels),
                )
                self.prompt_label_index = {
                    entry['prompt']: entry for entry in self.cache.prompt_entries
                }
        else:
            logger.warning(
                'label_cache_path not set; descriptive prompt scoring will be slow.'
            )

    def analyze_file(
        self,
        path: str | Path,
        audio_file: Optional[AudioFile] = None,
    ) -> AudioAnalysisRecord:
        """Analyze a single audio file and return an AudioAnalysisRecord."""
        if self.tagger is None:
            raise RuntimeError('Model not loaded. Call pipeline.load_model() first.')

        t0 = time.perf_counter()
        af = audio_file or load_audio(str(path), target_sr=self.target_sr)
        logger.info('Analyzing: %s (%.2fs)', af.file_name, af.duration)

        features = extract_features(af.waveform, af.sample_rate)
        audio_embed = self.tagger.embed_audio(af.waveform, af.sample_rate)
        full_scores = self._score_base_labels(af, audio_embed)
        prompt_scores = self._score_prompt_bank(af, audio_embed)
        events = self._detect_events(af, full_scores)
        evidence = self._build_evidence_bundle(full_scores, prompt_scores, events, features)

        result, mapping_diagnostics, provenance = analyze(
            evidence,
            self.taxonomy,
            backend=self.config.get('llm_backend', 'openai'),
            model=self.config.get('llm_model', 'gpt-4o-mini'),
            temperature=self.config.get('llm_temperature', 0.1),
            retries=self.config.get('llm_retry_count', 1),
        )

        elapsed = time.perf_counter() - t0
        record = self._assemble_record(
            af=af,
            features=features,
            evidence=evidence,
            result=result,
            mapping_diagnostics=mapping_diagnostics,
            llm_provenance=LLMProvenance(
                backend=provenance['backend'],
                model=provenance['model'],
                attempts=provenance['attempts'],
                repaired=provenance.get('repaired', False),
            ),
            analysis_elapsed_seconds=elapsed,
        )
        logger.info(
            'Done: %s | conf=%.2f | %.1fs elapsed',
            af.file_name,
            record.confidence,
            record.analysis_provenance.analysis_elapsed_seconds,
        )
        return record

    def analyze_batch(
        self,
        paths: List[str | Path],
        skip_errors: bool = True,
        progress_callback=None,
    ) -> List[AudioAnalysisRecord]:
        """Analyze a list of audio files."""
        if self.tagger is None:
            raise RuntimeError('Call pipeline.load_model() first.')

        results: List[AudioAnalysisRecord] = []
        total = len(paths)
        for i, path in enumerate(paths, start=1):
            if progress_callback:
                progress_callback(i, total, Path(path).name)
            try:
                results.append(self.analyze_file(path))
            except Exception as exc:
                if skip_errors:
                    logger.error("Failed '%s': %s", Path(path).name, exc)
                else:
                    raise
        logger.info('Batch complete: %d/%d files analyzed successfully.', len(results), total)
        return results

    def _score_base_labels(self, af: AudioFile, audio_embed) -> Dict[str, float]:
        if self.cache is not None:
            return self.cache.classify(audio_embed, top_k_categories=self.top_k_categories)
        return self.tagger.classify(
            waveform=af.waveform,
            sr=af.sample_rate,
            candidate_labels=self.candidate_labels,
        )

    def _score_prompt_bank(self, af: AudioFile, audio_embed) -> Dict[str, float]:
        if self.cache is not None:
            return self.cache.classify_prompts(audio_embed, top_k_categories=self.top_k_categories)
        prompt_labels = list(self.prompt_label_index.keys())
        return self.tagger.classify(
            waveform=af.waveform,
            sr=af.sample_rate,
            candidate_labels=prompt_labels,
        )

    def _detect_events(self, af: AudioFile, full_scores: Dict[str, float]) -> List[SoundEvent]:
        if self.use_windowed_analysis and af.duration >= self.windowed_min_duration:
            try:
                return detect_events(
                    waveform=af.waveform,
                    sr=af.sample_rate,
                    tagger=self.tagger,
                    label_to_category=self.label_to_category,
                    window_seconds=self.window_seconds,
                    hop_seconds=self.hop_seconds,
                    min_confidence=self.min_confidence,
                    cache=self.cache,
                    top_k_categories=self.top_k_categories,
                )
            except Exception as exc:
                logger.warning(
                    "Windowed event detection failed for '%s': %s. Using full-clip fallback.",
                    af.file_name,
                    exc,
                )
        return detect_events_from_full_clip(
            full_scores=full_scores,
            label_to_category=self.label_to_category,
            duration=af.duration,
            min_confidence=self.min_confidence,
        )

    def _build_evidence_bundle(
        self,
        full_scores: Dict[str, float],
        prompt_scores: Dict[str, float],
        events: List[SoundEvent],
        features: AcousticFeatures,
    ) -> EvidenceBundle:
        prompt_matches: list[PromptEvidence] = []
        for prompt, score in sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True)[:8]:
            entry = self.prompt_label_index.get(prompt)
            if entry is None:
                continue
            if isinstance(entry, dict):
                prompt_matches.append(
                    PromptEvidence(
                        prompt=entry['prompt'],
                        base_label=entry['base_label'],
                        category=entry['category'],
                        subcategory=entry['subcategory'],
                        cat_id=entry['cat_id'],
                        category_full=entry['category_full'],
                        modifiers=list(entry.get('modifiers', [])),
                        score=round(float(score), 6),
                    )
                )
            else:
                prompt_matches.append(
                    PromptEvidence(
                        prompt=entry.prompt,
                        base_label=entry.base_label,
                        category=entry.category,
                        subcategory=entry.subcategory,
                        cat_id=entry.cat_id,
                        category_full=entry.category_full,
                        modifiers=list(entry.modifiers),
                        score=round(float(score), 6),
                    )
                )

        event_items = [
            EvidenceEvent(
                label=event.label,
                category=event.category,
                start_time=round(event.start_time, 3),
                end_time=round(event.end_time, 3),
                confidence=round(event.confidence, 6),
            )
            for event in events
        ]

        acoustic_flags: list[str] = []
        if features.is_percussive:
            acoustic_flags.append('percussive')
        if features.is_tonal:
            acoustic_flags.append('tonal')
        if features.is_noisy:
            acoustic_flags.append('noisy')
        if features.is_low_frequency_heavy:
            acoustic_flags.append('low_frequency_heavy')
        if features.is_broadband:
            acoustic_flags.append('broadband')
        if features.silence_ratio > 0.35:
            acoustic_flags.append('sparse')

        return EvidenceBundle(
            base_label_scores=dict(
                (label, round(float(score), 6))
                for label, score in sorted(full_scores.items(), key=lambda item: item[1], reverse=True)[:20]
            ),
            descriptive_prompt_matches=prompt_matches,
            sound_events=event_items,
            acoustic_flags=acoustic_flags,
            dominant_frequency_band=features.dominant_frequency_band,
        )

    def _assemble_record(
        self,
        *,
        af: AudioFile,
        features: AcousticFeatures,
        evidence: EvidenceBundle,
        result: AnalysisResult,
        mapping_diagnostics: MappingDiagnostics,
        llm_provenance: LLMProvenance,
        analysis_elapsed_seconds: float,
    ) -> AudioAnalysisRecord:
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
            dominant_frequency_band=features.dominant_frequency_band,
        )
        analysis_provenance = AnalysisProvenance(
            model_id=self.config.get('model_id', 'unknown'),
            config_path=self.config.get('config_path', 'unknown'),
            vocab_path=self.config.get('vocab_path', 'unknown'),
            vocab_sha256=self.config.get('vocab_sha256', 'unknown'),
            analysis_elapsed_seconds=round(analysis_elapsed_seconds, 3),
            profile_name=self.config.get('profile_name', 'default'),
            profile_fingerprint=self.config.get('profile_fingerprint'),
            cache_path=self.config.get('label_cache_path'),
            cache_fingerprint=self.config.get('cache_fingerprint'),
            prompt_bank_version=self.config.get('prompt_bank_version'),
            prompt_bank_fingerprint=self.config.get('prompt_bank_fingerprint'),
        )

        suggested_filename = build_suggested_filename(
            cat_id=result.cat_id,
            fx_name=result.fx_name,
            creator_id=self.ucs_creator_id,
            source_id=self.ucs_source_id,
            user_data=self.ucs_user_data,
        )

        top_labels = dict(list(evidence.base_label_scores.items())[:10])
        classification_confidence = self._compute_classification_confidence(
            evidence, result, mapping_diagnostics,
        )
        description_confidence = self._compute_description_confidence(
            evidence, result,
        )
        metadata_confidence = self._compute_metadata_confidence(
            classification_confidence, mapping_diagnostics,
        )
        confidence = round(
            max(0.0, min(1.0,
                0.45 * classification_confidence
                + 0.25 * description_confidence
                + 0.30 * metadata_confidence,
                         )),
            3,
        )
        review_threshold = float(self.config.get('review_confidence_threshold', 0.45))
        review_required = any(
            value < review_threshold
            for value in (
                classification_confidence,
                description_confidence,
                metadata_confidence,
            )
        ) or bool(mapping_diagnostics.conflict_flags)

        return AudioAnalysisRecord(
            file_name=af.file_name,
            category=result.category,
            subcategory=result.subcategory,
            cat_id=result.cat_id,
            category_full=result.category_full,
            fx_name=result.fx_name,
            description=result.description,
            keywords=result.keywords,
            sound_events=result.sound_events,
            confidence=confidence,
            classification_confidence=classification_confidence,
            description_confidence=description_confidence,
            metadata_confidence=metadata_confidence,
            review_required=review_required,
            creator_id=self.ucs_creator_id,
            source_id=self.ucs_source_id,
            user_data=self.ucs_user_data,
            suggested_filename=suggested_filename,
            top_labels=top_labels,
            evidence=evidence,
            analysis_result=result,
            mapping_diagnostics=mapping_diagnostics,
            llm_provenance=llm_provenance,
            metadata=metadata,
            acoustic_summary=acoustic_summary,
            analysis_provenance=analysis_provenance,
        )

    def _compute_classification_confidence(
        self,
        evidence: EvidenceBundle,
        result: AnalysisResult,
        diagnostics: MappingDiagnostics,
    ) -> float:
        scores = list(evidence.base_label_scores.values())
        if not scores:
            return 0.0
        primary_score = scores[0]
        runner_up = scores[1] if len(scores) > 1 else 0.0
        margin = max(0.0, primary_score - runner_up)

        prompt_support_hits = [
            item for item in evidence.descriptive_prompt_matches
            if item.category == result.category and item.subcategory == result.subcategory
        ]
        prompt_support = (
            sum(item.score for item in prompt_support_hits[:3]
                ) / max(1, len(prompt_support_hits[:3]))
            if prompt_support_hits else 0.0
        )

        event_hits = [
            event for event in evidence.sound_events
            if event.category == result.category
        ]
        event_support = len(event_hits) / max(1, len(evidence.sound_events)
                                              ) if evidence.sound_events else 0.5

        conflict_penalty = min(0.3, 0.08 * len(diagnostics.conflict_flags))
        alt_penalty = 0.05 * min(3, len(diagnostics.ranked_alternatives))

        confidence = (
            0.45 * primary_score
            + 0.20 * min(1.0, margin * 5)
            + 0.20 * min(1.0, prompt_support * 6)
            + 0.15 * event_support
            - conflict_penalty
            - alt_penalty
        )
        return round(max(0.0, min(1.0, confidence)), 3)

    def _compute_description_confidence(
        self,
        evidence: EvidenceBundle,
        result: AnalysisResult,
    ) -> float:
        primary_score = next(iter(evidence.base_label_scores.values()), 0.0)
        acoustic_support = min(1.0, 0.2 * len(evidence.acoustic_flags))
        structure_support = 0.0
        if result.description:
            structure_support += 0.3
        if result.keywords:
            structure_support += 0.2
        if result.sound_events:
            structure_support += 0.2
        uncertainty_penalty = min(0.35, 0.1 * len(result.uncertainty_notes))
        confidence = (
            0.45 * primary_score
            + 0.20 * min(1.0, structure_support)
            + 0.20 * acoustic_support
            + 0.15 * min(1.0, 0.1 * len(result.keywords))
            - uncertainty_penalty
        )
        return round(max(0.0, min(1.0, confidence)), 3)

    def _compute_metadata_confidence(
        self,
        classification_confidence: float,
        diagnostics: MappingDiagnostics,
    ) -> float:
        conflict_penalty = min(0.35, 0.08 * len(diagnostics.conflict_flags))
        alt_penalty = 0.06 * min(3, len(diagnostics.ranked_alternatives))
        confidence = classification_confidence - conflict_penalty - alt_penalty
        return round(max(0.0, min(1.0, confidence)), 3)
