"""CLI for building the label embedding cache."""

from __future__ import annotations

import time
import logging
from pathlib import Path

import click

from timbre.config_loader import load_config, setup_logging
from timbre.models.clap_tagger import CLAPTagger
from timbre.models.label_cache import LabelEmbeddingCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@click.command()
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--vocab',
    default=None,
    help='Path to vocabulary YAML (overrides config default)',
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Rebuild cache even if it already exists',
)
@click.option(
    '--batch-size',
    type=int,
    default=64,
    help='Labels per text-encoder forward pass (reduce if OOM, default: 64)',
)
def main(config: str, vocab: str | None, force: bool, batch_size: int) -> None:
    """Pre-compute CLAP label embeddings for the UCS vocabulary."""

    config = load_config(config_path=config, vocab_path=vocab)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    cache_path = Path(
        config.get('label_cache_path', PROJECT_ROOT / 'config' / 'label_cache.pt')
    )
    candidate_labels = config['candidate_labels']

    logger.info(
        'Vocabulary: %d labels across %d categories',
        len(candidate_labels),
        len(set(config['label_to_category'].values())),
    )

    cache = LabelEmbeddingCache(cache_path)
    if not force and cache.is_valid(expected_label_count=len(candidate_labels)):
        logger.info('Cache already up to date at %s — use --force to rebuild.', cache_path)
        return

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
    )
    elapsed = time.perf_counter() - start

    logger.info('Done in %.1fs. Cache written to: %s', elapsed, cache_path)
