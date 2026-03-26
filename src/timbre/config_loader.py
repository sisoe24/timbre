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
EXPERIMENT_FINGERPRINT_KEYS = (
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


def _split_config_document(raw_cfg: dict) -> tuple[dict, dict, str | None]:
    experiments = raw_cfg.get('experiments') or {}
    if not isinstance(experiments, dict):
        raise ValueError("'experiments' must be a mapping of experiment names to overrides.")

    default_experiment = raw_cfg.get('default_experiment')
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

    return base_cfg, experiments, default_experiment


def list_experiments(config_path: Optional[str | Path] = None) -> List[str]:
    """Return the configured experiment names in declaration order."""
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(resolved_config_path)
    _, experiments, _ = _split_config_document(cfg)
    return list(experiments.keys())


def resolve_requested_experiments(
    config_path: Optional[str | Path] = None,
    requested_experiments: Optional[List[str] | tuple[str, ...]] = None,
    all_experiments: bool = False,
) -> List[str | None]:
    """
    Resolve which experiments a CLI invocation should run.

    Returns a list of experiment names. `None` means "use default selection".
    """
    requested = list(requested_experiments or [])

    if all_experiments and requested:
        raise ValueError('Use either --experiment or --all-experiments, not both.')

    if all_experiments:
        names = list_experiments(config_path)
        if not names:
            raise ValueError('No named experiments configured in config.yaml.')
        return names

    if not requested:
        return [None]

    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(requested))


def resolve_effective_config(
    raw_cfg: dict,
    experiment_name: Optional[str] = None,
) -> tuple[dict, str, str, List[str]]:
    """
    Resolve the effective config document for the selected experiment.

    Returns:
        merged_cfg, resolved_experiment_name, experiment_source, available_experiments
    """
    base_cfg, experiments, default_experiment = _split_config_document(raw_cfg)
    available_experiments = list(experiments.keys())

    if experiment_name is not None:
        if experiment_name not in experiments:
            raise ValueError(
                f"Unknown experiment '{experiment_name}'. "
                f"Available experiments: {', '.join(available_experiments) or 'none'}"
            )
        return (
            _deep_merge_dicts(base_cfg, experiments[experiment_name]),
            experiment_name,
            'explicit',
            available_experiments,
        )

    if default_experiment:
        if default_experiment not in experiments:
            raise ValueError(
                f"default_experiment '{default_experiment}' is not defined in 'experiments'."
            )
        return (
            _deep_merge_dicts(base_cfg, experiments[default_experiment]),
            default_experiment,
            'default',
            available_experiments,
        )

    return base_cfg, 'default', 'default', available_experiments


def _compute_experiment_fingerprint(runtime: dict) -> str:
    payload = {
        key: runtime.get(key)
        for key in EXPERIMENT_FINGERPRINT_KEYS
    }
    stable_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return _sha256_text(stable_json)[:12]


def refresh_runtime_metadata(runtime: dict) -> dict:
    """Refresh derived cache and experiment metadata after runtime overrides."""
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

    runtime['experiment_fingerprint'] = _compute_experiment_fingerprint(runtime)
    return runtime


def load_config(
    config_path: Optional[str | Path] = None,
    vocab_path: Optional[str | Path] = None,
    experiment_name: Optional[str] = None,
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
    cfg, resolved_experiment_name, experiment_source, available_experiments = (
        resolve_effective_config(raw_cfg, experiment_name=experiment_name)
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
        'experiment_name': resolved_experiment_name,
        'experiment_source': experiment_source,
        'available_experiments': available_experiments,
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
    refresh_runtime_metadata(runtime)

    logger.debug(
        'Config loaded: experiment=%s labels=%d categories=%d.',
        runtime['experiment_name'],
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
