from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from cli.main import main
from cli.batch import main as batch_main
from cli.analyze import main as analyze_main
from cli.profile import main as profile_main


def _write_vocab(path: Path) -> None:
    path.write_text(
        """
categories:
  IMPACTS:
    METAL:
      cat_id: "IMPMtl"
      labels:
        - "metal clang"
""".strip()
        + '\n',
        encoding='utf-8',
    )


def _write_config(path: Path) -> None:
    path.write_text(
        """
default_profile: balanced
base:
  model:
    model_id: "laion/larger_clap_general"
    vocab_file: "vocabulary.yaml"
  analysis:
    use_windowed_analysis: true
    windowed_min_duration: 2.0
    window_seconds: 2.0
    hop_seconds: 0.5
    min_confidence: 0.25
    top_k_categories: 5
profiles:
  balanced:
    label: "Balanced"
    description: "Default review profile."
  fast:
    label: "Fast"
    description: "Quick pass profile."
    analysis:
      hop_seconds: 1.0
""".strip()
        + '\n',
        encoding='utf-8',
    )


def test_analyze_help_omits_legacy_profile_listing_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(analyze_main, ['--help'])

    assert result.exit_code == 0
    assert '--list-profiles' not in result.output
    assert '--all-profiles' not in result.output


def test_validate_is_registered_on_root_command() -> None:
    """The documented validation command is reachable from the root CLI."""
    result = CliRunner().invoke(main, ['validate', '--help'])

    assert result.exit_code == 0
    assert '--input' in result.output


def test_batch_help_omits_legacy_profile_listing_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(batch_main, ['--help'])

    assert result.exit_code == 0
    assert '--list-profiles' not in result.output
    assert '--all-profiles' not in result.output


def test_profile_list_renders_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.yaml'
    vocab_path = tmp_path / 'vocabulary.yaml'
    _write_config(config_path)
    _write_vocab(vocab_path)

    runner = CliRunner()
    result = runner.invoke(profile_main, ['list', '--config', str(config_path)])

    assert result.exit_code == 0
    assert 'Balanced' in result.output
    assert 'Quick pass profile.' in result.output


def test_profile_inspect_shows_label_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.yaml'
    vocab_path = tmp_path / 'vocabulary.yaml'
    _write_config(config_path)
    _write_vocab(vocab_path)

    runner = CliRunner()
    result = runner.invoke(
        profile_main,
        ['inspect', 'fast', '--config', str(config_path), '--vocab', str(vocab_path)],
    )

    assert result.exit_code == 0
    assert 'Quick pass profile.' in result.output
    assert 'hop_seconds' in result.output
    assert '1.0' in result.output
