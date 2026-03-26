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
    profiles_to_run: list[str | None],
    discovered_audio_paths: list[Path] | None = None,
) -> None:
    default_profile = 'balanced'
    discovered_audio_paths = discovered_audio_paths or []

    config_loader = ModuleType('timbre.config_loader')

    def load_config(config_path=None, vocab_path=None, profile_name=None):
        effective_profile = profile_name if profile_name is not None else default_profile
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
            'output': {'save_per_file_markdown': False},
        }

    config_loader.load_config = load_config
    config_loader.list_profiles = lambda config_path=None: ['balanced', 'fast', 'precise']
    config_loader.setup_logging = lambda cfg, debug=False: None
    config_loader.refresh_runtime_metadata = lambda cfg: None
    config_loader.resolve_requested_profiles = (
        lambda config_path=None, requested_profiles=(), all_profiles=False: list(profiles_to_run)
    )

    pipeline = ModuleType('timbre.pipeline')

    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tagger = None
            self.cache = None

        def load_model(self) -> None:
            self.tagger = 'tagger'
            self.cache = 'cache'

        def analyze_file(self, path, audio_file=None):
            return SimpleNamespace(
                file_name=Path(path).name,
                analysis_provenance=SimpleNamespace(profile_name=self.cfg['profile_name']),
            )

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

    def save_json(record, output_dir, full=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{Path(record.file_name).stem}.json'
        output_path.write_text('{"file_name": "test"}\n', encoding='utf-8')
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


def test_analyze_validate_passes_saved_json_and_validation_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / 'impact.wav'
    audio_path.write_text('stub', encoding='utf-8')
    report_path = tmp_path / 'validation.json'
    calls: list[dict] = []

    _install_command_fakes(monkeypatch, tmp_path, profiles_to_run=[None])
    monkeypatch.setattr(validate_cli, 'run_validation', lambda **kwargs: calls.append(kwargs))

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
    assert len(calls) == 1
    assert calls[0]['input_path'] == tmp_path / 'out' / 'balanced' / 'json' / 'impact.json'
    assert calls[0]['backend'] == 'openai'
    assert calls[0]['model'] == 'gpt-5.4-mini'
    assert calls[0]['mode'] == 'autocorrect'
    assert calls[0]['temp'] == 0.3
    assert calls[0]['report'] == report_path
    assert calls[0]['profile'] == 'balanced'


def test_analyze_validate_runs_once_per_profile(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / 'impact.wav'
    audio_path.write_text('stub', encoding='utf-8')
    calls: list[dict] = []

    _install_command_fakes(monkeypatch, tmp_path, profiles_to_run=['fast', 'precise'])
    monkeypatch.setattr(validate_cli, 'run_validation', lambda **kwargs: calls.append(kwargs))

    runner = CliRunner()
    result = runner.invoke(
        analyze_main,
        ['--quiet', '--validate', '--profile', 'fast', '--profile', 'precise', str(audio_path)],
    )

    assert result.exit_code == 0
    assert [call['profile'] for call in calls] == ['fast', 'precise']
    assert [call['input_path'] for call in calls] == [
        tmp_path / 'out' / 'fast' / 'json' / 'impact.json',
        tmp_path / 'out' / 'precise' / 'json' / 'impact.json',
    ]


def test_analyze_validate_report_is_rejected_for_multi_profile_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / 'impact.wav'
    audio_path.write_text('stub', encoding='utf-8')

    _install_command_fakes(monkeypatch, tmp_path, profiles_to_run=['fast', 'precise'])

    runner = CliRunner()
    result = runner.invoke(
        analyze_main,
        [
            '--validate',
            '--validate-report',
            str(tmp_path / 'validation.json'),
            '--profile',
            'fast',
            '--profile',
            'precise',
            str(audio_path),
        ],
    )

    assert result.exit_code != 0
    assert '--validate-report cannot be used with multiple profiles' in result.output


def test_batch_validate_uses_profile_json_directory(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / 'clips'
    input_dir.mkdir()
    clip_path = input_dir / 'impact.wav'
    clip_path.write_text('stub', encoding='utf-8')
    calls: list[dict] = []

    _install_command_fakes(
        monkeypatch,
        tmp_path,
        profiles_to_run=[None],
        discovered_audio_paths=[clip_path],
    )
    monkeypatch.setattr(validate_cli, 'run_validation', lambda **kwargs: calls.append(kwargs))

    runner = CliRunner()
    result = runner.invoke(batch_main, ['--validate', str(input_dir)])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]['input_path'] == tmp_path / 'out' / 'balanced' / 'json'
    assert calls[0]['profile'] == 'balanced'


def test_batch_validate_runs_once_per_profile(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / 'clips'
    input_dir.mkdir()
    clip_path = input_dir / 'impact.wav'
    clip_path.write_text('stub', encoding='utf-8')
    calls: list[dict] = []

    _install_command_fakes(
        monkeypatch,
        tmp_path,
        profiles_to_run=['fast', 'precise'],
        discovered_audio_paths=[clip_path],
    )
    monkeypatch.setattr(validate_cli, 'run_validation', lambda **kwargs: calls.append(kwargs))

    runner = CliRunner()
    result = runner.invoke(
        batch_main,
        ['--validate', '--profile', 'fast', '--profile', 'precise', str(input_dir)],
    )

    assert result.exit_code == 0
    assert [call['profile'] for call in calls] == ['fast', 'precise']
    assert [call['input_path'] for call in calls] == [
        tmp_path / 'out' / 'fast' / 'json',
        tmp_path / 'out' / 'precise' / 'json',
    ]
