from __future__ import annotations

from pathlib import Path

from cli.validate import build_user_message, _default_report_path


def test_default_report_path_uses_analyzed_file_name_for_single_record() -> None:
    report_path = _default_report_path(
        Path('out/fast/validation/validation_report.json'),
        Path('out/fast/json/metal_impact_01.json'),
        [(
            Path('out/fast/json/metal_impact_01.json'),
            {'file_name': 'metal_impact_01.wav'},
        )],
    )

    assert report_path == Path('out/fast/validation/metal_impact_01.json')


def test_default_report_path_falls_back_to_input_stem_when_file_name_missing() -> None:
    report_path = _default_report_path(
        Path('out/fast/validation/validation_report.json'),
        Path('out/fast/json/metal_impact_01.json'),
        [(Path('out/fast/json/metal_impact_01.json'), {})],
    )

    assert report_path == Path('out/fast/validation/metal_impact_01.json')


def test_default_report_path_uses_directory_name_for_multi_record_runs() -> None:
    report_path = _default_report_path(
        Path('out/fast/validation/validation_report.json'),
        Path('out/fast/json'),
        [
            (Path('out/fast/json/metal_impact_01.json'), {'file_name': 'metal_impact_01.wav'}),
            (Path('out/fast/json/metal_impact_02.json'), {'file_name': 'metal_impact_02.wav'}),
        ],
    )

    assert report_path == Path('out/fast/validation/json_validation_report.json')


def test_build_user_message_excludes_raw_evidence_fields() -> None:
    prompt = build_user_message({
        'file_name': 'cane mangia.wav',
        'category': 'FOOD & DRINK',
        'subcategory': 'EATING',
        'cat_id': 'FOODEat',
        'category_full': 'FOOD & DRINK-EATING',
        'fx_name': 'EatingSounds',
        'description': 'Repeated eating sounds.',
        'keywords': ['eating'],
        'sound_events': ['eating'],
        'confidence': 0.44,
        'classification_confidence': 0.41,
        'description_confidence': 0.52,
        'metadata_confidence': 0.47,
        'review_required': True,
        'structured_description': {'primary_action': 'eating'},
        'mapping_diagnostics': {'conflict_flags': []},
        'validation_summary': None,
        'evidence': {'base_label_scores': {'pumping': 0.5}},
        'llm_provenance': {'description_model': 'gpt'},
    })

    assert 'structured_description' in prompt
    assert 'classification_confidence' in prompt
    assert 'evidence' not in prompt
    assert 'llm_provenance' not in prompt
    assert 'pumping' not in prompt
