"""CLI commands for managing vocabulary context and caches."""

from __future__ import annotations

import shutil
from pathlib import Path

import click
from rich.table import Table
from rich.console import Console

from timbre.paths import PROJECT_ROOT
from timbre.vocab_state import (remember_vocab, list_known_vocabs,
                                clear_active_vocab,
                                remember_vocab_with_metadata)
from timbre.config_loader import load_config, setup_logging
from timbre.models.label_cache import LabelEmbeddingCache

from .cache import build_cache_for_config

CONFIG_DIR = PROJECT_ROOT / 'config'
console = Console()


def _render_info_table(cfg: dict) -> None:
    cache_path_value = cfg.get('label_cache_path')
    cache_path = Path(cache_path_value) if cache_path_value else None
    cache = LabelEmbeddingCache(cache_path) if cache_path else None
    metadata = cache.read_metadata() if cache_path and cache_path.exists() else {}

    table = Table(title='Vocabulary Context', show_lines=True)
    table.add_column('Field', style='cyan', no_wrap=True)
    table.add_column('Value', style='white')
    table.add_row('Profile', str(cfg.get('profile_name')))
    table.add_row('Profile label', str(cfg.get('profile_label') or '—'))
    table.add_row('Profile description', str(cfg.get('profile_description') or '—'))
    table.add_row('Profile fingerprint', str(cfg.get('profile_fingerprint')))
    table.add_row('Resolved source', str(cfg.get('vocab_source')))
    table.add_row('Vocabulary file', str(cfg.get('vocab_file')))
    table.add_row('Vocabulary path', str(cfg.get('vocab_path')))
    table.add_row('Vocabulary SHA256', str(cfg.get('vocab_sha256')))
    table.add_row('Cache fingerprint', str(cfg.get('cache_fingerprint')))
    table.add_row('Resolved cache path', str(cache_path or '—'))
    table.add_row('Cache exists', 'yes' if cache_path and cache_path.exists() else 'no')
    table.add_row('Model', str(cfg.get('model_id')))

    if metadata:
        table.add_row('Cached model', str(metadata.get('model_id')))
        table.add_row('Cached vocab path', str(metadata.get('vocab_path')))
        table.add_row('Cached vocab SHA256', str(metadata.get('vocab_sha256')))
        table.add_row('Cached fingerprint', str(metadata.get('cache_fingerprint')))
        table.add_row('Cached label count', str(metadata.get('label_count')))
        table.add_row('Cached category count', str(metadata.get('category_count')))
        table.add_row('Created at', str(metadata.get('created_at')))

    console.print(table)


def _select_known_vocab_interactively() -> Path:
    items = [item for item in list_known_vocabs() if item.get('exists')]
    if not items:
        raise click.ClickException('No known vocabularies. Use `timbre vocab add <file>` first.')

    table = Table(title='Select Vocabulary', show_lines=True)
    table.add_column('Index', justify='right')
    table.add_column('Active', justify='center')
    table.add_column('Name', style='cyan')
    table.add_column('Managed', justify='center')
    table.add_column('Exists', justify='center')
    table.add_column('SHA256', style='white')
    table.add_column('Path', style='dim')

    for index, item in enumerate(items, start=1):
        table.add_row(
            str(index),
            '*' if item.get('is_active') else '',
            item.get('name', '?'),
            'yes' if item.get('managed') else 'no',
            'yes' if item.get('exists') else 'no',
            (item.get('sha256') or '')[:12],
            item.get('path', ''),
        )

    console.print(table)
    selected = click.prompt('Select vocabulary index', type=click.IntRange(1, len(items)))
    return Path(items[selected - 1]['path'])


@click.group()
def main() -> None:
    """Manage vocabulary context and cache lifecycle."""


@main.command('info')
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--vocab',
    default=None,
    help='Path to vocabulary YAML (overrides active/config selection)',
)
@click.option(
    '--profile',
    default=None,
    help='Named profile to load from config.yaml',
)
def info(config: str, vocab: str | None, profile: str | None) -> None:
    """Show the resolved vocabulary and cache information."""
    cfg = load_config(config_path=config, vocab_path=vocab, profile_name=profile)
    _render_info_table(cfg)


@main.command('current')
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--profile',
    default=None,
    help='Named profile to load from config.yaml',
)
def current(config: str, profile: str | None) -> None:
    """Alias for `timbre vocab info`."""
    cfg = load_config(config_path=config, profile_name=profile)
    _render_info_table(cfg)


@main.command('list')
def list_command() -> None:
    """List known vocabularies."""
    table = Table(title='Known Vocabularies', show_lines=True)
    table.add_column('Index', justify='right')
    table.add_column('Active', justify='center')
    table.add_column('Name', style='cyan')
    table.add_column('Managed', justify='center')
    table.add_column('Exists', justify='center')
    table.add_column('SHA256', style='white')
    table.add_column('Last Used', style='white')
    table.add_column('Path', style='dim')

    for index, item in enumerate(list_known_vocabs(), start=1):
        table.add_row(
            str(index),
            '*' if item.get('is_active') else '',
            item.get('name', '?'),
            'yes' if item.get('managed') else 'no',
            'yes' if item.get('exists') else 'no',
            (item.get('sha256') or '')[:12],
            item.get('last_used_at', ''),
            item.get('path', ''),
        )

    console.print(table)


@main.command('use')
@click.argument('target', required=False)
def use(target: str | None) -> None:
    """Activate a vocabulary by path, name, index, or interactive selection."""
    if target is None:
        vocab_path = _select_known_vocab_interactively()
    elif target.isdigit():
        items = list_known_vocabs()
        index = int(target)
        if index < 1 or index > len(items):
            raise click.ClickException(f'Index out of range: {index}')
        vocab_path = Path(items[index - 1]['path'])
    else:
        candidate_path = Path(target)
        if candidate_path.exists():
            vocab_path = candidate_path.resolve()
        else:
            items = list_known_vocabs()
            match = next((item for item in items if item.get('name') == target), None)
            if match is None:
                raise click.ClickException(f'Unknown vocabulary target: {target}')
            vocab_path = Path(match['path'])

    if not vocab_path.exists():
        raise click.ClickException(f'Vocabulary path does not exist: {vocab_path}')

    remember_vocab_with_metadata(vocab_path, make_active=True)
    console.print(f"[green]Active vocabulary set:[/green] {vocab_path}")


@main.command('add')
@click.argument('source_path', type=click.Path(exists=True, path_type=Path))
@click.option('--name', default=None, help='Optional filename to use inside config/')
@click.option('--force', is_flag=True, default=False, help='Overwrite existing managed file')
@click.option('--activate', is_flag=True, default=False, help='Also make the added vocab active')
@click.option('--batch-size', type=int, default=64, help='Cache build batch size')
@click.option('--debug', is_flag=True, default=False, help='Enable verbose debug logging')
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--profile',
    default=None,
    help='Named profile to load from config.yaml',
)
def add(
    source_path: Path,
    name: str | None,
    force: bool,
    activate: bool,
    batch_size: int,
    debug: bool,
    config: str,
    profile: str | None,
) -> None:
    """Copy a vocab into config/, register it, and build its cache."""
    destination = CONFIG_DIR / (name or source_path.name)
    if destination.exists() and not force:
        raise click.ClickException(
            f'Vocabulary already exists at {destination}. Use --force to overwrite.'
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    remember_vocab_with_metadata(destination, make_active=activate, managed=True, source='added')

    cfg = load_config(
        config_path=config,
        vocab_path=destination,
        profile_name=profile,
    )
    setup_logging(cfg, debug=debug)
    build_cache_for_config(cfg, force=force, batch_size=batch_size)

    console.print(f"[green]Vocabulary added:[/green] {destination}")
    console.print(f"[green]Cache ready:[/green] {cfg['label_cache_path']}")
    if activate:
        console.print(f"[green]Active vocabulary set:[/green] {destination}")


@main.command('cache')
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--vocab',
    default=None,
    help='Path to vocabulary YAML (overrides active/config selection)',
)
@click.option(
    '--profile',
    default=None,
    help='Named profile to load from config.yaml',
)
@click.option('--force', is_flag=True, default=False, help='Rebuild cache even if it exists')
@click.option('--batch-size', type=int, default=64, help='Cache build batch size')
@click.option('--debug', is_flag=True, default=False, help='Enable verbose debug logging')
def cache(
    config: str,
    vocab: str | None,
    profile: str | None,
    force: bool,
    batch_size: int,
    debug: bool,
) -> None:
    """Build or refresh the cache for the resolved vocabulary."""
    cfg = load_config(config_path=config, vocab_path=vocab, profile_name=profile)
    setup_logging(cfg, debug=debug)
    remember_vocab(cfg['vocab_path'], make_active=bool(vocab))
    build_cache_for_config(cfg, force=force, batch_size=batch_size)


@main.command('clear')
def clear() -> None:
    """Clear the active vocabulary and fall back to config/default resolution."""
    clear_active_vocab()
    console.print('[green]Active vocabulary cleared.[/green]')
