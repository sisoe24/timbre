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

import hashlib
import logging
import logging.handlers
from typing import Dict, List, Optional
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = ROOT / 'config' / 'config.yaml'
DEFAULT_VOCAB_PATH = ROOT / 'config' / 'vocabulary.yaml'
NOISY_LOGGERS = [
    'filelock',
    'httpcore',
    'httpx',
    'huggingface_hub',
    'urllib3',
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _fingerprinted_cache_path(base_path: Path, fingerprint: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{fingerprint}{base_path.suffix}")


def load_config(
    config_path: Optional[str | Path] = None,
    vocab_path: Optional[str | Path] = None,
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

    # Resolve vocab_path: explicit arg > config.yaml vocab_file > default
    if vocab_path is None:
        cfg_peek = _load_yaml(config_path)
        vocab_file = cfg_peek.get('model', {}).get('vocab_file')
        if vocab_file:
            vocab_path = config_path.parent / vocab_file
        else:
            vocab_path = DEFAULT_VOCAB_PATH
    vocab_path = Path(vocab_path)

    cfg = _load_yaml(config_path)
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

    # Resolve label_cache_path relative to the project root so generated
    # artifacts can live outside the editable config directory.
    raw_cache_path = model_cfg.get('label_cache_path')
    if raw_cache_path:
        cache_base_path = (ROOT / raw_cache_path).resolve()
        label_cache_path = str(_fingerprinted_cache_path(cache_base_path, cache_fingerprint))
        label_cache_base_path = str(cache_base_path)
    else:
        label_cache_path = None
        label_cache_base_path = None

    runtime = {
        # Model
        'model_id': model_id,
        'device': model_cfg.get('device', None),
        'fp16': model_cfg.get('fp16', True),
        'label_cache_path': label_cache_path,
        'label_cache_base_path': label_cache_base_path,
        'cache_fingerprint': cache_fingerprint,
        'config_path': str(resolved_config_path),
        'vocab_path': str(resolved_vocab_path),
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

    logger.debug(
        'Config loaded: %d candidate labels across %d UCS categories.',
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
