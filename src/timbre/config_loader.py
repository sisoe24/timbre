"""
config_loader.py
----------------
Loads and merges config.yaml + vocabulary.yaml into a single runtime
config dict consumed by AudioAnalysisPipeline.

UCS vocabulary structure
------------------------
The vocabulary.yaml now uses a nested UCS hierarchy:

    categories:
      <CATEGORY>:
        <SUBCATEGORY>:
          cat_id: <CatID>
          labels:
            - "clap label 1"
            - "clap label 2"

This is flattened into four parallel lookup dicts:
    label_to_category     : label → UCS Category  (e.g. "IMPACTS")
    label_to_subcategory  : label → UCS SubCategory (e.g. "METAL")
    label_to_cat_id       : label → UCS CatID       (e.g. "IMPMtl")
    label_to_category_full: label → "CATEGORY-SUBCATEGORY" (e.g. "IMPACTS-METAL")
"""

from __future__ import annotations

import copy
import json
import hashlib
import logging
import logging.handlers
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT
from .vocab_state import get_active_vocab_path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'
DEFAULT_VOCAB_PATH = PROJECT_ROOT / 'config' / 'vocabulary.yaml'
NOISY_LOGGERS = [
    'filelock',
    'httpcore',
    'httpx',
    'huggingface_hub',
    'urllib3',
]

CONFIG_SECTION_KEYS = ('model', 'audio', 'analysis', 'output', 'logging', 'ucs')
PROFILE_METADATA_KEYS = ('label', 'description')
PROFILE_FINGERPRINT_KEYS = (
    'model_id',
    'device',
    'fp16',
    'target_sr',
    'use_windowed_analysis',
    'windowed_min_duration',
    'window_seconds',
    'hop_seconds',
    'min_confidence',
    'top_k_categories',
    'vocab_sha256',
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _fingerprinted_cache_path(base_path: Path, fingerprint: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{fingerprint}{base_path.suffix}")


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _default_profile_label(name: str) -> str:
    return name.replace('_', ' ').title()


def _split_profile_entry(name: str, entry: dict | None) -> tuple[dict, dict]:
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise ValueError(f"Profile '{name}' must be a mapping.")

    metadata = {
        'name': name,
        'label': _default_profile_label(name),
        'description': '',
    }
    overrides = {}
    for key, value in entry.items():
        if key in PROFILE_METADATA_KEYS:
            metadata[key] = value or metadata[key]
        else:
            overrides[key] = copy.deepcopy(value)
    return metadata, overrides


def _split_config_document(raw_cfg: dict) -> tuple[dict, dict, str | None]:
    profiles = raw_cfg.get('profiles')
    if profiles is None:
        profiles = raw_cfg.get('experiments') or {}
    if not isinstance(profiles, dict):
        raise ValueError("'profiles' must be a mapping of profile names to overrides.")

    default_profile = raw_cfg.get('default_profile')
    if default_profile is None:
        default_profile = raw_cfg.get('default_experiment')
    if 'base' in raw_cfg:
        base_cfg = raw_cfg.get('base') or {}
        if not isinstance(base_cfg, dict):
            raise ValueError("'base' must be a mapping when present in config.yaml.")
    else:
        base_cfg = {
            key: copy.deepcopy(raw_cfg.get(key, {}))
            for key in CONFIG_SECTION_KEYS
            if key in raw_cfg
        }

    return base_cfg, profiles, default_profile


def list_profiles(config_path: Optional[str | Path] = None) -> List[str]:
    """Return the configured profile names in declaration order."""
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(resolved_config_path)
    _, profiles, _ = _split_config_document(cfg)
    return list(profiles.keys())


def get_default_profile_name(config_path: Optional[str | Path] = None) -> str | None:
    """Return the configured default profile name, if any."""
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(resolved_config_path)
    _, _, default_profile = _split_config_document(cfg)
    return default_profile


def get_profile_catalog(config_path: Optional[str | Path] = None) -> List[dict]:
    """Return profile metadata for all configured profiles."""
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(resolved_config_path)
    _, profiles, default_profile = _split_config_document(cfg)

    items = []
    for name, entry in profiles.items():
        metadata, _ = _split_profile_entry(name, entry)
        metadata['is_default'] = name == default_profile
        items.append(metadata)
    return items


def get_profile_definition(
    config_path: Optional[str | Path] = None,
    profile_name: Optional[str] = None,
) -> dict:
    """Return metadata and raw overrides for a specific profile."""
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(resolved_config_path)
    _, profiles, default_profile = _split_config_document(cfg)

    selected_name = profile_name or default_profile
    if selected_name is None:
        raise ValueError('No profile name provided and no default_profile is configured.')
    if selected_name not in profiles:
        raise ValueError(
            f"Unknown profile '{selected_name}'. "
            f"Available profiles: {', '.join(profiles.keys()) or 'none'}"
        )

    metadata, overrides = _split_profile_entry(selected_name, profiles[selected_name])
    metadata['is_default'] = selected_name == default_profile
    return {
        'metadata': metadata,
        'overrides': overrides,
    }


def resolve_requested_profiles(
    config_path: Optional[str | Path] = None,
    requested_profiles: Optional[List[str] | tuple[str, ...]] = None,
    all_profiles: bool = False,
) -> List[str | None]:
    """
    Resolve which profiles a CLI invocation should run.

    Returns a list of profile names. `None` means "use default selection".
    """
    requested = list(requested_profiles or [])

    if all_profiles and requested:
        raise ValueError('Use either --profile or --all-profiles, not both.')

    if all_profiles:
        names = list_profiles(config_path)
        if not names:
            raise ValueError('No named profiles configured in config.yaml.')
        return names

    if not requested:
        return [None]

    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(requested))


def resolve_effective_config(
    raw_cfg: dict,
    profile_name: Optional[str] = None,
) -> tuple[dict, str, str, List[str]]:
    """
    Resolve the effective config document for the selected profile.

    Returns:
        merged_cfg, resolved_profile_name, profile_source, available_profiles
    """
    base_cfg, profiles, default_profile = _split_config_document(raw_cfg)
    available_profiles = list(profiles.keys())

    if profile_name is not None:
        if profile_name not in profiles:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available profiles: {', '.join(available_profiles) or 'none'}"
            )
        _, profile_overrides = _split_profile_entry(profile_name, profiles[profile_name])
        return (
            _deep_merge_dicts(base_cfg, profile_overrides),
            profile_name,
            'explicit',
            available_profiles,
        )

    if default_profile:
        if default_profile not in profiles:
            raise ValueError(
                f"default_profile '{default_profile}' is not defined in 'profiles'."
            )
        _, profile_overrides = _split_profile_entry(default_profile, profiles[default_profile])
        return (
            _deep_merge_dicts(base_cfg, profile_overrides),
            default_profile,
            'default',
            available_profiles,
        )

    return base_cfg, 'default', 'default', available_profiles


def _compute_profile_fingerprint(runtime: dict) -> str:
    payload = {
        key: runtime.get(key)
        for key in PROFILE_FINGERPRINT_KEYS
    }
    stable_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return _sha256_text(stable_json)[:12]


def refresh_runtime_metadata(runtime: dict) -> dict:
    """Refresh derived cache and profile metadata after runtime overrides."""
    cache_fingerprint = _sha256_text(
        f"{runtime.get('model_id')}:{runtime.get('vocab_sha256')}"
    )[:12]
    runtime['cache_fingerprint'] = cache_fingerprint

    base_cache_path = runtime.get('label_cache_base_path')
    if base_cache_path:
        runtime['label_cache_path'] = str(
            _fingerprinted_cache_path(Path(base_cache_path), cache_fingerprint)
        )
    else:
        runtime['label_cache_path'] = None

    runtime['profile_fingerprint'] = _compute_profile_fingerprint(runtime)
    return runtime


def load_config(
    config_path: Optional[str | Path] = None,
    vocab_path: Optional[str | Path] = None,
    profile_name: Optional[str] = None,
) -> dict:
    """
    Load and merge configuration + vocabulary into a single dict.

    The returned dict has the following structure:
        {
            # Model
            "model_id": ...,
            "device": ...,
            "fp16": ...,
            # Audio
            "target_sr": ...,
            # Analysis
            "use_windowed_analysis": ...,
            "windowed_min_duration": ...,
            "window_seconds": ...,
            "hop_seconds": ...,
            "min_confidence": ...,
            "n_fft": ...,
            "hop_length": ...,
            "silence_threshold_db": ...,
            # Vocabulary (flat lookups)
            "candidate_labels": [...],
            "label_to_category": {...},
            "label_to_subcategory": {...},
            "label_to_cat_id": {...},
            "label_to_category_full": {...},
            # UCS identity
            "ucs_creator_id": ...,
            "ucs_source_id": ...,
            "ucs_user_data": ...,
            # Output
            "output": {...},
        }
    """
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)

    # Resolve vocab_path: explicit arg > active vocab context > config.yaml > default
    if vocab_path is None:
        active_vocab_path = get_active_vocab_path()
        if active_vocab_path is not None:
            vocab_path = active_vocab_path
            vocab_source = 'active'
        else:
            cfg_peek = _load_yaml(config_path)
            vocab_file = cfg_peek.get('model', {}).get('vocab_file')
            if vocab_file:
                vocab_path = config_path.parent / vocab_file
                vocab_source = 'config'
            else:
                vocab_path = DEFAULT_VOCAB_PATH
                vocab_source = 'default'
    else:
        vocab_source = 'explicit'
    vocab_path = Path(vocab_path)

    raw_cfg = _load_yaml(config_path)
    cfg, resolved_profile_name, profile_source, available_profiles = (
        resolve_effective_config(raw_cfg, profile_name=profile_name)
    )
    vocab = _load_yaml(vocab_path)

    # --- Flatten UCS vocabulary into lookup dicts -------------------------
    candidate_labels: List[str] = []
    label_to_category: Dict[str, str] = {}
    label_to_subcategory: Dict[str, str] = {}
    label_to_cat_id: Dict[str, str] = {}
    label_to_category_full: Dict[str, str] = {}

    for category, subcategories in vocab.get('categories', {}).items():
        if not isinstance(subcategories, dict):
            continue
        for subcategory, subcat_data in subcategories.items():
            if not isinstance(subcat_data, dict):
                continue
            cat_id = subcat_data.get('cat_id', '')
            category_full = f"{category}-{subcategory}"
            labels = subcat_data.get('labels', [])
            for label in labels:
                if label not in label_to_category:  # first mapping wins on collision
                    candidate_labels.append(label)
                    label_to_category[label] = category
                    label_to_subcategory[label] = subcategory
                    label_to_cat_id[label] = cat_id
                    label_to_category_full[label] = category_full

    # --- Flatten nested config into a single dict -------------------------
    model_cfg = cfg.get('model', {})
    audio_cfg = cfg.get('audio', {})
    analysis_cfg = cfg.get('analysis', {})
    output_cfg = cfg.get('output', {})
    log_cfg = cfg.get('logging', {})
    ucs_cfg = cfg.get('ucs', {})

    model_id = model_cfg.get('model_id', 'laion/larger_clap_general')
    resolved_config_path = config_path.resolve()
    resolved_vocab_path = vocab_path.resolve()
    vocab_sha256 = _sha256_file(resolved_vocab_path)
    cache_fingerprint = _sha256_text(f'{model_id}:{vocab_sha256}')[:12]

    raw_cache_path = model_cfg.get('label_cache_path')
    if raw_cache_path:
        cache_base_path = (PROJECT_ROOT / raw_cache_path).resolve()
        label_cache_base_path = str(cache_base_path)
        label_cache_path = str(_fingerprinted_cache_path(cache_base_path, cache_fingerprint))
    else:
        label_cache_base_path = None
        label_cache_path = None

    runtime = {
        # Model
        'model_id': model_id,
        'device': model_cfg.get('device', None),
        'fp16': model_cfg.get('fp16', True),
        'label_cache_path': label_cache_path,
        'label_cache_base_path': label_cache_base_path,
        'cache_fingerprint': cache_fingerprint,
        'profile_name': resolved_profile_name,
        'profile_source': profile_source,
        'available_profiles': available_profiles,
        'profile_label': '',
        'profile_description': '',
        'config_path': str(resolved_config_path),
        'vocab_path': str(resolved_vocab_path),
        'vocab_file': resolved_vocab_path.name,
        'vocab_source': vocab_source,
        'vocab_sha256': vocab_sha256,
        # Audio
        'target_sr': audio_cfg.get('target_sr', 48000),
        # Analysis
        'use_windowed_analysis': analysis_cfg.get('use_windowed_analysis', True),
        'windowed_min_duration': analysis_cfg.get('windowed_min_duration', 2.0),
        'window_seconds': analysis_cfg.get('window_seconds', 2.0),
        'hop_seconds': analysis_cfg.get('hop_seconds', 0.5),
        'min_confidence': analysis_cfg.get('min_confidence', 0.25),
        'top_k_categories': analysis_cfg.get('top_k_categories', 5),
        'n_fft': analysis_cfg.get('n_fft', 2048),
        'hop_length': analysis_cfg.get('hop_length', 512),
        'silence_threshold_db': analysis_cfg.get('silence_threshold_db', -50.0),
        # Vocabulary (flat UCS lookups)
        'candidate_labels': candidate_labels,
        'label_to_category': label_to_category,
        'label_to_subcategory': label_to_subcategory,
        'label_to_cat_id': label_to_cat_id,
        'label_to_category_full': label_to_category_full,
        # UCS identity
        'ucs_creator_id': ucs_cfg.get('creator_id', 'UNKNOWN'),
        'ucs_source_id': ucs_cfg.get('source_id', 'NONE'),
        'ucs_user_data': ucs_cfg.get('user_data', ''),
        # Output
        'output': output_cfg,
        # Logging
        'log_level': log_cfg.get('level', 'INFO'),
        'log_format': log_cfg.get(
            'format', '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ),
        'log_file': log_cfg.get('log_file', None),
    }
    if resolved_profile_name != 'default' and resolved_profile_name in available_profiles:
        metadata, _ = _split_profile_entry(
            resolved_profile_name,
            dict(raw_cfg.get('profiles', raw_cfg.get('experiments', {}))).get(
                resolved_profile_name
            ),
        )
        runtime['profile_label'] = metadata.get('label', '')
        runtime['profile_description'] = metadata.get('description', '')
    else:
        runtime['profile_label'] = _default_profile_label(runtime['profile_name'])
        runtime['profile_description'] = ''
    refresh_runtime_metadata(runtime)

    logger.debug(
        'Config loaded: profile=%s labels=%d categories=%d.',
        runtime['profile_name'],
        len(candidate_labels),
        len(set(label_to_category.values())),
    )

    return runtime


def setup_logging(config: dict, debug: bool = False) -> None:
    """Configure the root logger from the loaded config dict."""
    level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    if debug:
        level = logging.DEBUG
    fmt = config.get(
        'log_format', '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    handlers = [logging.StreamHandler()]

    log_file = config.get('log_file')
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)

    noisy_level = logging.DEBUG if debug else logging.WARNING
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(noisy_level)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}
