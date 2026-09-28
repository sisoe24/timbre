from __future__ import annotations

from pathlib import Path

import yaml
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


def test_prompt_bank_version_updates_cache_fingerprint(
    temp_config: tuple[Path, Path],
) -> None:
    config_path, vocab_path = temp_config

    cfg = load_config(config_path=config_path, vocab_path=vocab_path)
    original_cache = cfg['cache_fingerprint']
    original_prompt = cfg['prompt_bank_fingerprint']

    cfg['prompt_bank_version'] = 'v2'
    refresh_runtime_metadata(cfg)

    assert cfg['cache_fingerprint'] != original_cache
    assert cfg['prompt_bank_fingerprint'] != original_prompt


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


@pytest.mark.parametrize('profile,active,explicit,expected', [
    (None, False, False, 'vocabulary.yaml'),
    ('fast', False, False, 'profile.yaml'),
    ('fast', True, False, 'active.yaml'),
    ('fast', True, True, 'explicit.yaml'),
])
def test_vocab_selection_respects_profile_and_override_precedence(
    temp_config: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    profile: str | None,
    active: bool,
    explicit: bool,
    expected: str,
) -> None:
    """Vocabulary selection uses the effective profile unless overridden."""
    config_path, _ = temp_config
    for name in ('profile.yaml', 'active.yaml', 'explicit.yaml'):
        _write_vocab(config_path.parent / name)
    document = yaml.safe_load(config_path.read_text())
    document['profiles']['fast']['model'] = {'vocab_file': 'profile.yaml'}
    config_path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(
        'timbre.config_loader.get_active_vocab_path',
        lambda: config_path.parent / 'active.yaml' if active else None,
    )

    cfg = load_config(
        config_path=config_path,
        profile_name=profile,
        vocab_path=config_path.parent / 'explicit.yaml' if explicit else None,
    )

    assert Path(cfg['vocab_path']) == config_path.parent / expected
    assert cfg['vocab_source'] == ('explicit' if explicit else 'active' if active else 'config')


def test_llm_settings_and_fingerprint_follow_effective_profile(
    temp_config: tuple[Path, Path],
) -> None:
    """The analyzer receives the selected provider, model, and temperature."""
    config_path, vocab_path = temp_config
    document = yaml.safe_load(config_path.read_text())
    document['profiles']['fast']['llm'] = {
        'backend': 'ollama', 'model': 'local-model', 'temperature': 0.7,
    }
    config_path.write_text(yaml.safe_dump(document))
    cfg = load_config(config_path, vocab_path, profile_name='fast')

    assert (cfg['llm_backend'], cfg['llm_model'], cfg['llm_temperature']) == (
        'ollama', 'local-model', 0.7,
    )
    for key, value in (
        ('llm_backend', 'anthropic'), ('llm_model', 'other-model'), ('llm_temperature', 0.2),
    ):
        changed = dict(cfg, **{key: value})
        refresh_runtime_metadata(changed)
        assert changed['profile_fingerprint'] != cfg['profile_fingerprint']


@pytest.mark.parametrize('legacy_key', ['description_backend', 'metadata_model'])
def test_legacy_llm_settings_require_explicit_migration(
    temp_config: tuple[Path, Path], legacy_key: str,
) -> None:
    """Old two-stage settings cannot silently select the default provider."""
    config_path, vocab_path = temp_config
    document = yaml.safe_load(config_path.read_text())
    document['base']['llm'] = {legacy_key: 'old-setting'}
    config_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match='llm.backend, llm.model, and llm.temperature'):
        load_config(config_path, vocab_path)
