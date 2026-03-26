from __future__ import annotations

from pathlib import Path

from timbre.output.schema import (AudioMetadata, AcousticSummary,
                                  AnalysisProvenance, AudioAnalysisRecord)
from timbre.output.serializer import save_csv, save_markdown
from timbre.output.catalog_builder import build_catalog_markdown


def _build_record(experiment_name: str, fingerprint: str) -> AudioAnalysisRecord:
    return AudioAnalysisRecord(
        file_name=f'{experiment_name}.wav',
        category='IMPACTS',
        subcategory='METAL',
        cat_id='IMPMtl',
        category_full='IMPACTS-METAL',
        fx_name='Metal Impact',
        description='A metallic impact with a short ring.',
        keywords=['impacts', 'metal', 'metallic impact'],
        sound_events=['metallic impact'],
        confidence=0.82,
        creator_id='UNKNOWN',
        source_id='NONE',
        user_data='',
        suggested_filename=f'IMPMtl_Metal Impact_UNKNOWN_NONE_{experiment_name}',
        top_labels={'metallic impact': 0.82},
        metadata=AudioMetadata(
            file_name=f'{experiment_name}.wav',
            file_path=f'/tmp/{experiment_name}.wav',
            format='wav',
            duration_seconds=1.2,
            sample_rate_hz=48000,
            original_sample_rate_hz=48000,
            num_channels=1,
            num_samples=57600,
        ),
        acoustic_summary=AcousticSummary(
            rms_mean=0.1,
            spectral_centroid_mean_hz=1234.5,
            spectral_flatness_mean=0.02,
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
            vocab_sha256='abcdef1234567890',
            experiment_name=experiment_name,
            experiment_fingerprint=fingerprint,
            cache_path='/tmp/cache.pt',
            cache_fingerprint='cache12345678',
        ),
    )


def test_serializer_outputs_include_experiment_provenance(tmp_path: Path) -> None:
    record = _build_record('fast', 'expfast123456')

    csv_path = save_csv([record], tmp_path / 'catalog.csv')
    markdown_path = save_markdown(record, tmp_path / 'markdown')

    csv_text = csv_path.read_text(encoding='utf-8')
    markdown_text = markdown_path.read_text(encoding='utf-8')

    assert 'experiment_name' in csv_text
    assert 'fast' in csv_text
    assert 'Experiment' in markdown_text
    assert 'expfast123456' in markdown_text


def test_catalog_groups_provenance_by_experiment(tmp_path: Path) -> None:
    fast_record = _build_record('fast', 'expfast123456')
    precise_record = _build_record('precise', 'expprecise789')

    output_path = build_catalog_markdown(
        [fast_record, precise_record],
        tmp_path / 'catalog.md',
    )
    catalog_text = output_path.read_text(encoding='utf-8')

    assert 'Mixed analysis profiles detected: 2' in catalog_text
    assert 'experiment=`fast`' in catalog_text
    assert 'experiment=`precise`' in catalog_text
