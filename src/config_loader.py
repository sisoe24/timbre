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

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
DEFAULT_VOCAB_PATH = Path(__file__).parent.parent / "config" / "vocabulary.yaml"


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
    vocab_path = Path(vocab_path or DEFAULT_VOCAB_PATH)

    cfg = _load_yaml(config_path)
    vocab = _load_yaml(vocab_path)

    # --- Flatten UCS vocabulary into lookup dicts -------------------------
    candidate_labels: List[str] = []
    label_to_category: Dict[str, str] = {}
    label_to_subcategory: Dict[str, str] = {}
    label_to_cat_id: Dict[str, str] = {}
    label_to_category_full: Dict[str, str] = {}

    for category, subcategories in vocab.get("categories", {}).items():
        if not isinstance(subcategories, dict):
            continue
        for subcategory, subcat_data in subcategories.items():
            if not isinstance(subcat_data, dict):
                continue
            cat_id = subcat_data.get("cat_id", "")
            category_full = f"{category}-{subcategory}"
            labels = subcat_data.get("labels", [])
            for label in labels:
                if label not in label_to_category:  # first mapping wins on collision
                    candidate_labels.append(label)
                    label_to_category[label] = category
                    label_to_subcategory[label] = subcategory
                    label_to_cat_id[label] = cat_id
                    label_to_category_full[label] = category_full

    # --- Flatten nested config into a single dict -------------------------
    model_cfg = cfg.get("model", {})
    audio_cfg = cfg.get("audio", {})
    analysis_cfg = cfg.get("analysis", {})
    output_cfg = cfg.get("output", {})
    log_cfg = cfg.get("logging", {})
    ucs_cfg = cfg.get("ucs", {})

    # Resolve label_cache_path relative to the config file's directory
    raw_cache_path = model_cfg.get("label_cache_path")
    if raw_cache_path:
        label_cache_path = str(config_path.parent / raw_cache_path)
    else:
        label_cache_path = None

    runtime = {
        # Model
        "model_id": model_cfg.get("model_id", "laion/larger_clap_general"),
        "device": model_cfg.get("device", None),
        "fp16": model_cfg.get("fp16", True),
        "label_cache_path": label_cache_path,
        # Audio
        "target_sr": audio_cfg.get("target_sr", 48000),
        # Analysis
        "use_windowed_analysis": analysis_cfg.get("use_windowed_analysis", True),
        "windowed_min_duration": analysis_cfg.get("windowed_min_duration", 2.0),
        "window_seconds": analysis_cfg.get("window_seconds", 2.0),
        "hop_seconds": analysis_cfg.get("hop_seconds", 0.5),
        "min_confidence": analysis_cfg.get("min_confidence", 0.25),
        "n_fft": analysis_cfg.get("n_fft", 2048),
        "hop_length": analysis_cfg.get("hop_length", 512),
        "silence_threshold_db": analysis_cfg.get("silence_threshold_db", -50.0),
        # Vocabulary (flat UCS lookups)
        "candidate_labels": candidate_labels,
        "label_to_category": label_to_category,
        "label_to_subcategory": label_to_subcategory,
        "label_to_cat_id": label_to_cat_id,
        "label_to_category_full": label_to_category_full,
        # UCS identity
        "ucs_creator_id": ucs_cfg.get("creator_id", "UNKNOWN"),
        "ucs_source_id": ucs_cfg.get("source_id", "NONE"),
        "ucs_user_data": ucs_cfg.get("user_data", ""),
        # Output
        "output": output_cfg,
        # Logging
        "log_level": log_cfg.get("level", "INFO"),
        "log_format": log_cfg.get(
            "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ),
        "log_file": log_cfg.get("log_file", None),
    }

    logger.debug(
        "Config loaded: %d candidate labels across %d UCS categories.",
        len(candidate_labels),
        len(set(label_to_category.values())),
    )

    return runtime


def setup_logging(config: dict) -> None:
    """Configure the root logger from the loaded config dict."""
    level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    fmt = config.get(
        "log_format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handlers = [logging.StreamHandler()]

    log_file = config.get("log_file")
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
