from __future__ import annotations

from pathlib import Path

from cli.validate import _default_report_path


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
