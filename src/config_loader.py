"""
config_loader.py
----------------
Loads and merges config.yaml + vocabulary.yaml into a single runtime
config dict consumed by AudioAnalysisPipeline.
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
            "model_id": ...,
            "device": ...,
            "fp16": ...,
            "target_sr": ...,
            "window_seconds": ...,
            "hop_seconds": ...,
            "min_confidence": ...,
            "use_windowed_analysis": ...,
            "windowed_min_duration": ...,
            "n_fft": ...,
            "hop_length": ...,
            "silence_threshold_db": ...,
            "candidate_labels": [...],    # flat list of all vocabulary labels
            "label_to_category": {...},   # label → category mapping
            "output": {...},              # output settings
        }
    """
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    vocab_path = Path(vocab_path or DEFAULT_VOCAB_PATH)

    cfg = _load_yaml(config_path)
    vocab = _load_yaml(vocab_path)

    # --- Flatten vocabulary into candidate_labels + label_to_category ---
    candidate_labels: List[str] = []
    label_to_category: Dict[str, str] = {}

    for category, labels in vocab.get("categories", {}).items():
        for label in labels:
            if label not in label_to_category:  # first category wins on collision
                candidate_labels.append(label)
                label_to_category[label] = category

    # --- Flatten nested config into a single dict -----------------------
    model_cfg = cfg.get("model", {})
    audio_cfg = cfg.get("audio", {})
    analysis_cfg = cfg.get("analysis", {})
    output_cfg = cfg.get("output", {})
    log_cfg = cfg.get("logging", {})

    runtime = {
        # Model
        "model_id": model_cfg.get("model_id", "laion/larger_clap_general"),
        "device": model_cfg.get("device", None),
        "fp16": model_cfg.get("fp16", True),
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
        # Vocabulary
        "candidate_labels": candidate_labels,
        "label_to_category": label_to_category,
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
        "Config loaded: %d candidate labels across %d categories.",
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
