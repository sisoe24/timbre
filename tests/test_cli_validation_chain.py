from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

from click.testing import CliRunner

import cli.batch as batch_cli
import cli.analyze as analyze_cli
import cli.validate as validate_cli
from cli.batch import main as batch_main
from cli.analyze import main as analyze_main


def _install_command_fakes(
    monkeypatch,
    tmp_path: Path,
    discovered_audio_paths: list[Path] | None = None,
    analyze_failures: set[str] | None = None,
) -> None:
    discovered_audio_paths = discovered_audio_paths or []
    analyze_failures = analyze_failures or set()

    config_loader = ModuleType('timbre.config_loader')

    def load_config(config_path=None, vocab_path=None, profile_name=None):
        effective_profile = profile_name if profile_name is not None else 'balanced'
        return {
            'profile_name': effective_profile,
            'profile_fingerprint': f'fp-{effective_profile}',
            'model_id': 'fake-model',
            'vocab_path': str(tmp_path / 'vocabulary.yaml'),
            'vocab_sha256': 'abcdef1234567890',
            'vocab_source': 'test',
            'target_sr': 48000,
            'device': None,
            'fp16': False,
            'label_cache_path': None,
            'output': {'save_per_file_markdown': False, 'save_validation_report': False},
        }

    config_loader.load_config = load_config
    config_loader.setup_logging = lambda cfg, debug=False: None
    config_loader.refresh_runtime_metadata = lambda cfg: None

    pipeline = ModuleType('timbre.pipeline')

    class FakeRecord:
        def __init__(self, path: str, profile_name: str):
            self.file_name = Path(path).name
            self.category = 'IMPACTS'
            self.subcategory = 'METAL'
            self.cat_id = 'IMPMtl'
            self.category_full = 'IMPACTS-METAL'
            self.fx_name = 'Metal Hit'
            self.description = 'A short metal hit.'
            self.keywords = ['metal', 'impact']
            self.sound_events = ['metal impact']
            self.confidence = 0.82
            self.suggested_filename = 'IMPMtl_Metal Hit_UNKNOWN_NONE'
            self.source_id = 'NONE'
            self.creator_id = 'UNKNOWN'
            self.top_labels = {'metal impact': 0.82}
            self.metadata = SimpleNamespace(
                duration_seconds=1.0,
                sample_rate_hz=48000,
                format='wav',
            )
            self.analysis_provenance = SimpleNamespace(
                profile_name=profile_name,
                analysis_elapsed_seconds=0.42,
            )
            self.validation_summary = None

        def to_full_dict(self):
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
                'analysis_provenance': {'profile_name': self.analysis_provenance.profile_name},
                'validation_summary': self.validation_summary,
            }

        def model_copy(self, update=None):
            copied = FakeRecord(self.file_name, self.analysis_provenance.profile_name)
            copied.validation_summary = (update or {}).get('validation_summary')
            return copied

    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        def load_model(self) -> None:
            return None

        def analyze_file(self, path, audio_file=None):
            if Path(path).name in analyze_failures:
                raise RuntimeError(f'boom:{Path(path).name}')
            return FakeRecord(path, self.cfg['profile_name'])

    pipeline.AudioAnalysisPipeline = FakePipeline

    output_paths = ModuleType('timbre.output_paths')

    def resolve_output_paths(cfg, explicit_output_dir=None):
        root_name = Path(explicit_output_dir).name if explicit_output_dir else 'out'
        root = tmp_path / root_name / cfg['profile_name']
        return {
            'root': root,
            'json_dir': root / 'json',
            'markdown_dir': root / 'markdown',
            'catalog_markdown': root / 'catalog.md',
            'catalog_csv': root / 'catalog.csv',
            'batch_json': root / 'batch_results.json',
            'validation_report': root / 'validation' / 'validation_report.json',
        }

    output_paths.resolve_output_paths = resolve_output_paths

    serializer = ModuleType('timbre.output.serializer')
    saved_json_paths: list[Path] = []

    def save_json(record, output_dir, full=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{Path(record.file_name).stem}.json'
        output_path.write_text('{"file_name": "test"}\n', encoding='utf-8')
        saved_json_paths.append(output_path)
        return output_path

    def save_json_batch(records, output_path, full=False):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('[]\n', encoding='utf-8')
        return output_path

    serializer.save_json = save_json
    serializer.save_markdown = lambda record, output_dir: None
    serializer.save_json_batch = save_json_batch

    audio_loader = ModuleType('timbre.ingestion.audio_loader')
    audio_loader.load_audio = lambda path, target_sr=48000: f'audio:{Path(path).name}:{target_sr}'
    audio_loader.discover_audio_files = (
        lambda input_dir, recursive=True: [str(path) for path in discovered_audio_paths]
    )

    catalog_builder = ModuleType('timbre.output.catalog_builder')
    catalog_builder.build_catalog_markdown = lambda records, output_path: None
    catalog_builder.build_catalog_csv = lambda records, output_path: None

    monkeypatch.setitem(sys.modules, 'timbre.config_loader', config_loader)
    monkeypatch.setitem(sys.modules, 'timbre.pipeline', pipeline)
    monkeypatch.setitem(sys.modules, 'timbre.output_paths', output_paths)
    monkeypatch.setitem(sys.modules, 'timbre.output.serializer', serializer)
    monkeypatch.setitem(sys.modules, 'timbre.ingestion.audio_loader', audio_loader)
    monkeypatch.setitem(sys.modules, 'timbre.output.catalog_builder', catalog_builder)

    monkeypatch.setattr(analyze_cli, 'remember_vocab', lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_cli, 'remember_vocab', lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_cli, '_print_batch_summary', lambda records: None)

    return saved_json_paths


def test_analyze_validate_uses_in_memory_record_and_optional_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / 'impact.wav'
    audio_path.write_text('stub', encoding='utf-8')
    report_path = tmp_path / 'validation.json'
    validation_calls: list[dict] = []
    report_calls: list[dict] = []

    saved = _install_command_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        validate_cli,
        'validate_record',
        lambda record, **kwargs: (validation_calls.append({'record': record, **kwargs}) or ({
            'file_name': record.file_name,
            'consistency_score': 0.95,
            'issues': [],
            'notes': 'ok',
        }, record.to_full_dict())),
    )
    monkeypatch.setattr(
        validate_cli,
        'maybe_write_validation_report',
        lambda results, report=None, config=None: (
            report_calls.append({'results': results, 'report': report, 'config': config}) or report
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        analyze_main,
        [
            '--quiet',
            '--validate',
            '--validate-backend',
            'openai',
            '--validate-model',
            'gpt-5.4-mini',
            '--validate-mode',
            'autocorrect',
            '--validate-temp',
            '0.3',
            '--validate-report',
            str(report_path),
            str(audio_path),
        ],
    )

    assert result.exit_code == 0
    assert len(validation_calls) == 1
    assert validation_calls[0]['record'].file_name == 'impact.wav'
    assert validation_calls[0]['backend'] == 'openai'
    assert validation_calls[0]['model'] == 'gpt-5.4-mini'
    assert validation_calls[0]['mode'] == 'autocorrect'
    assert validation_calls[0]['temp'] == 0.3
    assert len(report_calls) == 1
    assert report_calls[0]['report'] == report_path
    assert len(saved) == 1
    assert saved[0] == tmp_path / 'out' / 'balanced' / 'json' / 'impact.json'


def test_analyze_validate_failure_blocks_save(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / 'impact.wav'
    audio_path.write_text('stub', encoding='utf-8')
    saved = _install_command_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        validate_cli,
        'validate_record',
        lambda record, **kwargs: (_ for _ in ()).throw(RuntimeError('validator down')),
    )
    monkeypatch.setattr(
        validate_cli,
        'maybe_write_validation_report',
        lambda results, report=None, config=None: report,
    )

    runner = CliRunner()
    result = runner.invoke(analyze_main, ['--quiet', '--validate', str(audio_path)])

    assert result.exit_code != 0
    assert not saved
    assert 'validator down' in result.output


def test_batch_validate_uses_in_memory_records(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / 'clips'
    input_dir.mkdir()
    clip_path = input_dir / 'impact.wav'
    clip_path.write_text('stub', encoding='utf-8')
    validation_calls: list[dict] = []
    report_calls: list[dict] = []

    saved = _install_command_fakes(
        monkeypatch,
        tmp_path,
        discovered_audio_paths=[clip_path],
    )
    monkeypatch.setattr(
        validate_cli,
        'validate_record',
        lambda record, **kwargs: (validation_calls.append({'record': record, **kwargs}) or ({
            'file_name': record.file_name,
            'consistency_score': 0.9,
            'issues': [],
            'notes': 'ok',
        }, record.to_full_dict())),
    )
    monkeypatch.setattr(
        validate_cli,
        'maybe_write_validation_report',
        lambda results, report=None, config=None: (
            report_calls.append({'results': results, 'report': report, 'config': config}) or report
        ),
    )

    runner = CliRunner()
    result = runner.invoke(batch_main, ['--validate', str(input_dir)])

    assert result.exit_code == 0
    assert len(validation_calls) == 1
    assert validation_calls[0]['record'].file_name == 'impact.wav'
    assert len(report_calls) == 1
    assert report_calls[0]['report'] is None
    assert len(saved) == 1
    assert saved[0] == tmp_path / 'out' / 'balanced' / 'json' / 'impact.json'


def test_batch_validate_failure_skips_file_and_continues(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / 'clips'
    input_dir.mkdir()
    good = input_dir / 'good.wav'
    bad = input_dir / 'bad.wav'
    good.write_text('stub', encoding='utf-8')
    bad.write_text('stub', encoding='utf-8')

    saved = _install_command_fakes(
        monkeypatch,
        tmp_path,
        discovered_audio_paths=[good, bad],
    )

    def fake_validate(record, **kwargs):
        if record.file_name == 'bad.wav':
            raise RuntimeError('bad validation')
        return ({
            'file_name': record.file_name,
            'consistency_score': 0.9,
            'issues': [],
            'notes': 'ok',
        }, record.to_full_dict())

    monkeypatch.setattr(validate_cli, 'validate_record', fake_validate)
    monkeypatch.setattr(
        validate_cli,
        'maybe_write_validation_report',
        lambda results, report=None, config=None: report,
    )

    runner = CliRunner()
    result = runner.invoke(batch_main, ['--validate', str(input_dir)])

    assert result.exit_code == 0
    assert len(saved) == 1
    assert saved[0].name == 'good.json'
    assert 'Skipped bad.wav: bad validation' in result.output


def test_analyze_help_does_not_advertise_multi_profile_features() -> None:
    runner = CliRunner()
    result = runner.invoke(analyze_main, ['--help'])

    assert result.exit_code == 0
    assert '--all-profiles' not in result.output
    assert '--list-profiles' not in result.output
    assert '--profile TEXT' in result.output


def test_batch_help_does_not_advertise_multi_profile_features() -> None:
    runner = CliRunner()
    result = runner.invoke(batch_main, ['--help'])

    assert result.exit_code == 0
    assert '--all-profiles' not in result.output
    assert '--list-profiles' not in result.output
    assert '--profile TEXT' in result.output
