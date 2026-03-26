from __future__ import annotations

from pathlib import Path

import pytest

from timbre.output_paths import resolve_output_paths
from timbre.config_loader import (load_config, refresh_runtime_metadata,
                                  resolve_requested_experiments)


def _write_vocab(path: Path) -> None:
    path.write_text(
        """
categories:
  IMPACTS:
    METAL:
      cat_id: "IMPMtl"
      labels:
        - "metallic impact"
        - "metal clang"
""".strip()
        + '\n',
        encoding='utf-8',
    )


def _write_config(path: Path) -> None:
    path.write_text(
        """
default_experiment: balanced
base:
  model:
    model_id: "laion/larger_clap_general"
    device: null
    fp16: true
    vocab_file: "vocabulary.yaml"
    label_cache_path: ".cache/test_cache.pt"
  audio:
    target_sr: 48000
  analysis:
    use_windowed_analysis: true
    windowed_min_duration: 2.0
    window_seconds: 2.0
    hop_seconds: 0.5
    min_confidence: 0.25
    top_k_categories: 5
  output:
    output_dir: "./out"
    json_dir: "./out/json"
    markdown_dir: "./out/markdown"
    catalog_markdown: "./out/catalog.md"
    catalog_csv: "./out/catalog.csv"
    batch_json: "./out/batch_results.json"
    validation_report: "./out/validation/validation_report.json"
  logging:
    level: "INFO"
experiments:
  balanced: {}
  fast:
    analysis:
      hop_seconds: 1.0
      top_k_categories: 3
""".strip()
        + '\n',
        encoding='utf-8',
    )


@pytest.fixture()
def temp_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / 'config.yaml'
    vocab_path = tmp_path / 'vocabulary.yaml'
    _write_config(config_path)
    _write_vocab(vocab_path)
    return config_path, vocab_path


def test_load_config_applies_default_experiment(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config

    cfg = load_config(config_path=config_path, vocab_path=vocab_path)

    assert cfg['experiment_name'] == 'balanced'
    assert cfg['experiment_source'] == 'default'
    assert cfg['hop_seconds'] == 0.5
    assert cfg['available_experiments'] == ['balanced', 'fast']


def test_load_config_applies_named_experiment_override(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config

    cfg = load_config(
        config_path=config_path,
        vocab_path=vocab_path,
        experiment_name='fast',
    )

    assert cfg['experiment_name'] == 'fast'
    assert cfg['experiment_source'] == 'explicit'
    assert cfg['hop_seconds'] == 1.0
    assert cfg['top_k_categories'] == 3


def test_load_config_rejects_unknown_experiment(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config

    with pytest.raises(ValueError, match="Unknown experiment 'missing'"):
        load_config(
            config_path=config_path,
            vocab_path=vocab_path,
            experiment_name='missing',
        )


def test_experiment_fingerprint_is_stable_and_updates_on_override(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, vocab_path = temp_config

    cfg_default = load_config(config_path=config_path, vocab_path=vocab_path)
    cfg_explicit = load_config(
        config_path=config_path,
        vocab_path=vocab_path,
        experiment_name='balanced',
    )

    assert cfg_default['experiment_fingerprint'] == cfg_explicit['experiment_fingerprint']

    cfg_default['use_windowed_analysis'] = False
    refresh_runtime_metadata(cfg_default)

    assert cfg_default['experiment_fingerprint'] != cfg_explicit['experiment_fingerprint']


def test_output_paths_are_scoped_by_experiment(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config
    cfg = load_config(
        config_path=config_path,
        vocab_path=vocab_path,
        experiment_name='fast',
    )

    paths = resolve_output_paths(cfg)
    explicit = resolve_output_paths(cfg, explicit_output_dir='custom-out')

    assert paths['root'] == Path('out') / 'fast'
    assert paths['json_dir'] == Path('out') / 'fast' / 'json'
    assert explicit['root'] == Path('custom-out') / 'fast'
    assert explicit['validation_report'] == (
        Path('custom-out') / 'fast' / 'validation' / 'validation_report.json'
    )


def test_resolve_requested_experiments_defaults_to_default_selection(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    names = resolve_requested_experiments(config_path=config_path)

    assert names == [None]


def test_resolve_requested_experiments_all_profiles(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    names = resolve_requested_experiments(
        config_path=config_path,
        all_experiments=True,
    )

    assert names == ['balanced', 'fast']


def test_resolve_requested_experiments_rejects_mixed_selection(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    with pytest.raises(ValueError, match='Use either --experiment or --all-experiments'):
        resolve_requested_experiments(
            config_path=config_path,
            requested_experiments=['balanced'],
            all_experiments=True,
        )
