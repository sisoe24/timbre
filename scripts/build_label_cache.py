#!/usr/bin/env python3
"""
build_label_cache.py
--------------------
Pre-computes CLAP text embeddings for the full UCS vocabulary and writes them
to a .pt cache file.  Run this once after changing vocabulary.yaml.

Usage
-----
    python scripts/build_label_cache.py
    python scripts/build_label_cache.py --vocab config/vocabulary.yaml
    python scripts/build_label_cache.py --force   # rebuild even if cache exists

The cache is saved to the path specified in config/config.yaml under
  model.label_cache_path
(default: config/label_cache.pt)

Why this exists
---------------
Without a cache, CLAP re-encodes all ~5 500 text labels on every audio file
(and again for every sliding window).  With the cache, text is encoded once
and inference becomes a fast cosine similarity against the cached matrix.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path when called from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, setup_logging
from src.models.clap_tagger import CLAPTagger
from src.models.label_cache import LabelEmbeddingCache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute CLAP label embeddings for the UCS vocabulary."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Path to vocabulary YAML (overrides config default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if it already exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Labels per text-encoder forward pass (reduce if OOM, default: 64)",
    )
    args = parser.parse_args()

    # --- Load config + vocab -------------------------------------------
    config = load_config(config_path=args.config, vocab_path=args.vocab)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    cache_path = Path(config.get("label_cache_path",
                                  PROJECT_ROOT / "config" / "label_cache.pt"))
    candidate_labels = config["candidate_labels"]

    logger.info("Vocabulary: %d labels across %d categories",
                len(candidate_labels),
                len(set(config["label_to_category"].values())))

    # --- Check if cache already valid ----------------------------------
    cache = LabelEmbeddingCache(cache_path)
    if not args.force and cache.is_valid(expected_label_count=len(candidate_labels)):
        logger.info(
            "Cache already up to date at %s — use --force to rebuild.", cache_path
        )
        return

    # --- Load model ----------------------------------------------------
    logger.info("Loading CLAP model (this may take a moment)…")
    tagger = CLAPTagger(
        model_id=config.get("model_id", "laion/larger_clap_general"),
        device=config.get("device", None),
        fp16=config.get("fp16", True),
    )
    tagger.load()

    # --- Build cache ---------------------------------------------------
    t0 = time.perf_counter()
    cache.build(
        tagger=tagger,
        candidate_labels=candidate_labels,
        label_to_category=config["label_to_category"],
        label_to_subcategory=config["label_to_subcategory"],
        label_to_cat_id=config["label_to_cat_id"],
        label_to_category_full=config["label_to_category_full"],
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - t0

    logger.info("Done in %.1fs.  Cache written to: %s", elapsed, cache_path)
    logger.info(
        "From now on, 'timbre' will use pre-computed embeddings — "
        "each file analysis only needs one audio forward pass."
    )


if __name__ == "__main__":
    main()
