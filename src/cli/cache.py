"""CLI for building the label embedding cache."""

from __future__ import annotations

import time
import logging
from pathlib import Path

from timbre.models.clap_tagger import CLAPTagger
from timbre.models.label_cache import LabelEmbeddingCache, build_cache_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_cache_for_config(
    config: dict,
    force: bool = False,
    batch_size: int = 64,
) -> Path:
    """Build or validate the cache for an already-loaded config."""
    logger = logging.getLogger(__name__)

    cache_path = Path(
        config.get('label_cache_path', PROJECT_ROOT / '.cache' / 'label_cache.pt')
    )
    candidate_labels = config['candidate_labels']

    logger.info(
        'Vocabulary: %d labels across %d categories',
        len(candidate_labels),
        len(set(config['label_to_category'].values())),
    )

    cache = LabelEmbeddingCache(cache_path)
    expected_metadata = {
        'model_id': config.get('model_id'),
        'vocab_sha256': config.get('vocab_sha256'),
        'cache_fingerprint': config.get('cache_fingerprint'),
    }
    if not force and cache.is_valid(
        expected_label_count=len(candidate_labels),
        expected_metadata=expected_metadata,
    ):
        logger.info('Cache already up to date at %s — use --force to rebuild.', cache_path)
        return cache_path

    logger.info('Loading CLAP model (this may take a moment)…')
    tagger = CLAPTagger(
        model_id=config.get('model_id', 'laion/larger_clap_general'),
        device=config.get('device', None),
        fp16=config.get('fp16', True),
    )
    tagger.load()

    start = time.perf_counter()
    cache.build(
        tagger=tagger,
        candidate_labels=candidate_labels,
        label_to_category=config['label_to_category'],
        label_to_subcategory=config['label_to_subcategory'],
        label_to_cat_id=config['label_to_cat_id'],
        label_to_category_full=config['label_to_category_full'],
        batch_size=batch_size,
        metadata=build_cache_metadata(config, candidate_labels),
    )
    elapsed = time.perf_counter() - start

    logger.info('Done in %.1fs. Cache written to: %s', elapsed, cache_path)
    return cache_path
