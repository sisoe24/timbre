"""CLI for inspecting label cache provenance."""

from __future__ import annotations

from pathlib import Path

import click
from rich.table import Table
from rich.console import Console

from timbre.config_loader import load_config
from timbre.models.label_cache import LabelEmbeddingCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
console = Console()


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
def main(config: str, vocab: str | None) -> None:
    """Show which label cache this configuration resolves to."""
    cfg = load_config(config_path=config, vocab_path=vocab)
    cache_path = Path(cfg['label_cache_path'])
    cache = LabelEmbeddingCache(cache_path)
    metadata = cache.read_metadata() if cache_path.exists() else {}

    table = Table(title='Label Cache Info', show_lines=True)
    table.add_column('Field', style='cyan', no_wrap=True)
    table.add_column('Value', style='white')
    table.add_row('Resolved cache path', str(cache_path))
    table.add_row('Cache exists', 'yes' if cache_path.exists() else 'no')
    table.add_row('Model', str(cfg.get('model_id')))
    table.add_row('Vocabulary path', str(cfg.get('vocab_path')))
    table.add_row('Vocabulary SHA256', str(cfg.get('vocab_sha256')))
    table.add_row('Cache fingerprint', str(cfg.get('cache_fingerprint')))

    if metadata:
        table.add_row('Cached model', str(metadata.get('model_id')))
        table.add_row('Cached vocab path', str(metadata.get('vocab_path')))
        table.add_row('Cached vocab SHA256', str(metadata.get('vocab_sha256')))
        table.add_row('Cached fingerprint', str(metadata.get('cache_fingerprint')))
        table.add_row('Cached label count', str(metadata.get('label_count')))
        table.add_row('Cached category count', str(metadata.get('category_count')))
        table.add_row('Created at', str(metadata.get('created_at')))

    console.print(table)
