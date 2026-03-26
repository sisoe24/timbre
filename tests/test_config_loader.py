from __future__ import annotations

from pathlib import Path

import pytest

from timbre.output_paths import resolve_output_paths
from timbre.config_loader import (load_config, get_profile_catalog,
                                  get_profile_definition,
                                  refresh_runtime_metadata,
                                  resolve_requested_profiles)


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
default_profile: balanced
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
profiles:
  balanced:
    label: "Balanced"
    description: "Default review profile."
  fast:
    label: "Fast"
    description: "Quick pass profile."
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


def test_load_config_applies_default_profile(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config

    cfg = load_config(config_path=config_path, vocab_path=vocab_path)

    assert cfg['profile_name'] == 'balanced'
    assert cfg['profile_label'] == 'Balanced'
    assert cfg['profile_description'] == 'Default review profile.'
    assert cfg['profile_source'] == 'default'
    assert cfg['hop_seconds'] == 0.5
    assert cfg['available_profiles'] == ['balanced', 'fast']


def test_load_config_applies_named_profile_override(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config

    cfg = load_config(
        config_path=config_path,
        vocab_path=vocab_path,
        profile_name='fast',
    )

    assert cfg['profile_name'] == 'fast'
    assert cfg['profile_source'] == 'explicit'
    assert cfg['hop_seconds'] == 1.0
    assert cfg['top_k_categories'] == 3


def test_load_config_rejects_unknown_profile(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config

    with pytest.raises(ValueError, match="Unknown profile 'missing'"):
        load_config(
            config_path=config_path,
            vocab_path=vocab_path,
            profile_name='missing',
        )


def test_profile_fingerprint_is_stable_and_updates_on_override(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, vocab_path = temp_config

    cfg_default = load_config(config_path=config_path, vocab_path=vocab_path)
    cfg_explicit = load_config(
        config_path=config_path,
        vocab_path=vocab_path,
        profile_name='balanced',
    )

    assert cfg_default['profile_fingerprint'] == cfg_explicit['profile_fingerprint']

    cfg_default['use_windowed_analysis'] = False
    refresh_runtime_metadata(cfg_default)

    assert cfg_default['profile_fingerprint'] != cfg_explicit['profile_fingerprint']


def test_output_paths_are_scoped_by_profile(temp_config: tuple[Path, Path]) -> None:
    config_path, vocab_path = temp_config
    cfg = load_config(
        config_path=config_path,
        vocab_path=vocab_path,
        profile_name='fast',
    )

    paths = resolve_output_paths(cfg)
    explicit = resolve_output_paths(cfg, explicit_output_dir='custom-out')

    assert paths['root'] == Path('out') / 'fast'
    assert paths['json_dir'] == Path('out') / 'fast' / 'json'
    assert explicit['root'] == Path('custom-out') / 'fast'
    assert explicit['validation_report'] == (
        Path('custom-out') / 'fast' / 'validation' / 'validation_report.json'
    )


def test_resolve_requested_profiles_defaults_to_default_selection(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    names = resolve_requested_profiles(config_path=config_path)

    assert names == [None]


def test_resolve_requested_profiles_all_profiles(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    names = resolve_requested_profiles(
        config_path=config_path,
        all_profiles=True,
    )

    assert names == ['balanced', 'fast']


def test_resolve_requested_profiles_rejects_mixed_selection(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    with pytest.raises(ValueError, match='Use either --profile or --all-profiles'):
        resolve_requested_profiles(
            config_path=config_path,
            requested_profiles=['balanced'],
            all_profiles=True,
        )


def test_profile_catalog_exposes_label_and_description(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    catalog = get_profile_catalog(config_path)

    assert catalog[0]['name'] == 'balanced'
    assert catalog[0]['label'] == 'Balanced'
    assert catalog[0]['description'] == 'Default review profile.'
    assert catalog[0]['is_default'] is True


def test_profile_definition_returns_metadata_and_overrides(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, _ = temp_config

    definition = get_profile_definition(config_path, 'fast')

    assert definition['metadata']['name'] == 'fast'
    assert definition['metadata']['label'] == 'Fast'
    assert definition['metadata']['description'] == 'Quick pass profile.'
    assert definition['overrides']['analysis']['hop_seconds'] == 1.0
