from __future__ import annotations

from timbre.pipeline import AudioAnalysisPipeline
from timbre.output.schema import (AudioMetadata, EvidenceEvent, LLMProvenance,
                                  EvidenceBundle, PromptEvidence,
                                  AcousticSummary, RankedAlternative,
                                  ValidationSummary, AnalysisProvenance,
                                  DescriptionDetails, MappingDiagnostics,
                                  AudioAnalysisRecord)


def _pipeline() -> AudioAnalysisPipeline:
    return AudioAnalysisPipeline({
        'candidate_labels': ['metal impact'],
        'label_to_category': {'metal impact': 'IMPACTS'},
        'label_to_subcategory': {'metal impact': 'METAL'},
        'label_to_cat_id': {'metal impact': 'IMPMtl'},
        'label_to_category_full': {'metal impact': 'IMPACTS-METAL'},
        'taxonomy': {
            'IMPACTS': {
                'METAL': {'cat_id': 'IMPMtl', 'category_full': 'IMPACTS-METAL'}
            }
        },
    })


def test_pipeline_confidence_penalizes_conflicts() -> None:
    pipeline = _pipeline()
    evidence = EvidenceBundle(
        base_label_scores={'metal impact': 0.8, 'clang': 0.1},
        descriptive_prompt_matches=[
            PromptEvidence(
                prompt='sharp metal impact',
                base_label='metal impact',
                category='IMPACTS',
                subcategory='METAL',
                cat_id='IMPMtl',
                category_full='IMPACTS-METAL',
                modifiers=['sharp'],
                score=0.5,
            )
        ],
        sound_events=[
            EvidenceEvent(
                label='metal impact',
                category='IMPACTS',
                start_time=0.0,
                end_time=0.4,
                confidence=0.9,
            )
        ],
        acoustic_flags=['percussive'],
        dominant_frequency_band='mid',
    )
    mapped = {
        'category': 'IMPACTS',
        'subcategory': 'METAL',
    }

    clean = pipeline._compute_confidence(
        evidence,
        mapped,
        MappingDiagnostics(ranked_alternatives=[], conflict_flags=[]),
    )
    conflicted = pipeline._compute_confidence(
        evidence,
        mapped,
        MappingDiagnostics(
            ranked_alternatives=[
                RankedAlternative(
                    category='IMPACTS',
                    subcategory='METAL',
                    cat_id='IMPMtl',
                    category_full='IMPACTS-METAL',
                    reason='close second',
                )
            ],
            conflict_flags=['ambiguous_source'],
        ),
    )

    assert clean > conflicted


def test_brief_output_stays_catalog_focused() -> None:
    record = AudioAnalysisRecord(
        file_name='impact.wav',
        category='IMPACTS',
        subcategory='METAL',
        cat_id='IMPMtl',
        category_full='IMPACTS-METAL',
        fx_name='Metal Hit',
        description='A short sharp metal hit.',
        keywords=['metal', 'impact'],
        sound_events=['metal impact'],
        confidence=0.88,
        creator_id='UNKNOWN',
        source_id='NONE',
        user_data='',
        suggested_filename='IMPMtl_Metal Hit_UNKNOWN_NONE',
        top_labels={'metal impact': 0.8},
        evidence=EvidenceBundle(
            base_label_scores={'metal impact': 0.8},
            descriptive_prompt_matches=[],
            sound_events=[],
            acoustic_flags=[],
            dominant_frequency_band='mid',
        ),
        description_details=DescriptionDetails(
            description='A short sharp metal hit.',
            salient_attributes=['sharp'],
            uncertain_attributes=[],
            negative_claims=[],
            keyword_candidates=['metal', 'impact'],
        ),
        mapping_diagnostics=MappingDiagnostics(),
        llm_provenance=LLMProvenance(
            description_backend='openai',
            description_model='gpt-4o-mini',
            description_attempts=1,
            description_repaired=False,
            metadata_backend='openai',
            metadata_model='gpt-4o-mini',
            metadata_attempts=1,
            metadata_repaired=False,
        ),
        validation_summary=ValidationSummary(
            backend='openai',
            model='gpt-4o',
            mode='audit',
            consistency_score=0.94,
            issues=[],
            notes='validated inline',
        ),
        metadata=AudioMetadata(
            file_name='impact.wav',
            file_path='/tmp/impact.wav',
            format='wav',
            duration_seconds=1.0,
            sample_rate_hz=48000,
            original_sample_rate_hz=48000,
            num_channels=1,
            num_samples=48000,
        ),
        acoustic_summary=AcousticSummary(
            rms_mean=0.1,
            spectral_centroid_mean_hz=1200,
            spectral_flatness_mean=0.2,
            is_percussive=True,
            is_tonal=False,
            is_noisy=False,
            silence_ratio=0.1,
            dynamic_range_db=12.0,
            dominant_frequency_band='mid',
        ),
        analysis_provenance=AnalysisProvenance(
            model_id='laion/larger_clap_general',
            config_path='/tmp/config.yaml',
            vocab_path='/tmp/vocabulary.yaml',
            vocab_sha256='abc',
            analysis_elapsed_seconds=0.5,
        ),
    )

    brief = record.to_brief_dict()

    assert 'evidence' not in brief
    assert 'mapping_diagnostics' not in brief
    assert brief['category'] == 'IMPACTS'
    assert brief['validation_summary']['validated_inline'] is True
    assert brief['validation_summary']['consistency_score'] == 0.94
