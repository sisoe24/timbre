"""CLI for batch-analyzing an entire folder of audio files."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import (Progress, BarColumn, TextColumn, SpinnerColumn,
                           TimeElapsedColumn, MofNCompleteColumn)

from timbre.vocab_state import remember_vocab

console = Console()


@click.command()
@click.argument('input_dir', required=False, type=click.Path(exists=True, file_okay=False))
@click.option(
    '--output-dir',
    '-o',
    default=None,
    help='Root output directory (default: ./outputs/)',
)
@click.option('--config', '-c', default=None, help='Path to config.yaml')
@click.option('--vocab', '-v', default=None, help='Path to vocabulary.yaml')
@click.option(
    '--profile',
    multiple=True,
    help='Named profile to load from config.yaml. Repeat to run several.',
)
@click.option(
    '--all-profiles',
    is_flag=True,
    default=False,
    help='Run all named profiles from config.yaml',
)
@click.option(
    '--list-profiles',
    is_flag=True,
    default=False,
    help='List profile names from config.yaml and exit',
)
@click.option(
    '--recursive',
    '-r',
    is_flag=True,
    default=True,
    help='Recurse into sub-directories (default: true)',
)
@click.option(
    '--catalog',
    is_flag=True,
    default=True,
    help='Generate a Markdown catalog from all results (default: true)',
)
@click.option(
    '--csv',
    'save_csv',
    is_flag=True,
    default=True,
    help='Generate a CSV catalog (default: true)',
)
@click.option(
    '--markdown',
    'save_per_file_markdown',
    is_flag=True,
    default=False,
    help='Save per-file Markdown review reports',
)
@click.option(
    '--full',
    is_flag=True,
    default=False,
    help='Save full JSON (with metadata + acoustics) per file',
)
@click.option(
    '--no-windowed',
    is_flag=True,
    default=False,
    help='Disable sliding-window event detection',
)
@click.option(
    '--skip-errors',
    is_flag=True,
    default=True,
    help='Skip files that fail to load/analyze (default: true)',
)
@click.option('--limit', default=None, type=int, help='Limit to the first N files')
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable verbose debug logging, including third-party request logs',
)
def main(
    input_dir: str | None,
    output_dir: str,
    config: str,
    vocab: str,
    profile: tuple[str, ...],
    all_profiles: bool,
    list_profiles: bool,
    recursive: bool,
    catalog: bool,
    save_csv: bool,
    save_per_file_markdown: bool,
    full: bool,
    no_windowed: bool,
    skip_errors: bool,
    limit: int,
    debug: bool,
) -> None:
    """Batch analyze all audio files in INPUT_DIR."""

    from timbre.pipeline import AudioAnalysisPipeline
    from timbre.output_paths import resolve_output_paths
    from timbre.config_loader import load_config
    from timbre.config_loader import list_profiles as list_config_profiles
    from timbre.config_loader import (setup_logging, refresh_runtime_metadata,
                                      resolve_requested_profiles)
    from timbre.output.serializer import save_json
    from timbre.output.serializer import save_markdown as save_md
    from timbre.output.serializer import save_json_batch
    from timbre.ingestion.audio_loader import discover_audio_files
    from timbre.output.catalog_builder import (build_catalog_csv,
                                               build_catalog_markdown)

    if list_profiles:
        names = list_config_profiles(config)
        if names:
            console.print('\n'.join(names))
        else:
            console.print('[yellow]No named profiles configured.[/yellow]')
        return

    if input_dir is None:
        raise click.UsageError('Missing argument: INPUT_DIR')

    audio_paths = discover_audio_files(input_dir, recursive=recursive)
    if not audio_paths:
        console.print(f"[red]No supported audio files found in: {input_dir}[/red]")
        sys.exit(1)

    if limit:
        audio_paths = audio_paths[:limit]

    console.print(f"\nFound [bold]{len(audio_paths)}[/bold] audio files.\n")

    try:
        profiles_to_run = resolve_requested_profiles(
            config_path=config,
            requested_profiles=profile,
            all_profiles=all_profiles,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    shared_resources: dict[tuple, tuple] = {}

    for profile_name in profiles_to_run:
        cfg = load_config(
            config_path=config,
            vocab_path=vocab,
            profile_name=profile_name,
        )
        if no_windowed:
            cfg['use_windowed_analysis'] = False
            refresh_runtime_metadata(cfg)
        setup_logging(cfg, debug=debug)
        remember_vocab(cfg['vocab_path'], make_active=bool(vocab))

        vocab_file = Path(cfg['vocab_path']).name
        vocab_sha = cfg['vocab_sha256'][:12]
        vocab_source = cfg['vocab_source']

        output_paths = resolve_output_paths(cfg, explicit_output_dir=output_dir)
        out_root = output_paths['root']
        json_dir = output_paths['json_dir']
        md_dir = output_paths['markdown_dir']
        catalog_md = output_paths['catalog_markdown']
        catalog_csv_path = output_paths['catalog_csv']
        batch_json_path = output_paths['batch_json']

        console.print(
            Panel.fit(
                f"[bold cyan]Audio Analyzer — Batch Mode[/bold cyan]\n"
                f"Input: [green]{input_dir}[/green]\n"
                f"Output: [yellow]{out_root}[/yellow]\n"
                f"Profile: [blue]{cfg['profile_name']}[/blue] "
                f"[dim]({cfg['profile_fingerprint']})[/dim]\n"
                f"Model: [yellow]{cfg['model_id']}[/yellow]\n"
                f"Vocab: [magenta]{vocab_file}[/magenta] "
                f"[dim]({vocab_sha}, {vocab_source})[/dim]",
                title='🎧 Batch Analysis',
            )
        )

        pipeline = AudioAnalysisPipeline(cfg)
        resource_key = _resource_cache_key(cfg)
        if resource_key in shared_resources:
            pipeline.tagger, pipeline.cache = shared_resources[resource_key]
            console.print('[green]✓[/green] Reusing loaded CLAP model.\n')
        else:
            with console.status('Loading CLAP model…'):
                pipeline.load_model()
            console.print('[green]✓[/green] Model loaded.\n')
            shared_resources[resource_key] = (pipeline.tagger, pipeline.cache)

        records = []
        failed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Analyzing [{cfg['profile_name']}]…",
                total=len(audio_paths),
            )

            for path in audio_paths:
                progress.update(task, description=f"[cyan]{Path(path).name}[/cyan]")
                try:
                    record = pipeline.analyze_file(path)
                    records.append(record)
                    save_json(record, json_dir, full=full)
                    if save_per_file_markdown:
                        save_md(record, md_dir)
                except Exception as exc:
                    failed += 1
                    if not skip_errors:
                        raise
                    console.print(f"[yellow]⚠ Skipped {Path(path).name}: {exc}[/yellow]")
                progress.advance(task)

        console.print(
            f"\n[bold green]✓ Analyzed {len(records)}/{len(audio_paths)} files[/bold green]"
            + (f" ([yellow]{failed} failed[/yellow])" if failed else '')
        )

        if not records:
            console.print('[red]No records produced for this profile.[/red]')
            if len(profiles_to_run) == 1:
                sys.exit(1)
            continue

        save_json_batch(records, batch_json_path, full=full)
        console.print(f"[dim]Batch JSON → {batch_json_path}[/dim]")

        if catalog:
            build_catalog_markdown(records, catalog_md)
            console.print(f"[dim]Catalog   → {catalog_md}[/dim]")

        if save_csv:
            build_catalog_csv(records, catalog_csv_path)
            console.print(f"[dim]CSV       → {catalog_csv_path}[/dim]")

        _print_batch_summary(records)


def _print_batch_summary(records) -> None:
    console.print()
    table = Table(
        title=f"Batch Results ({len(records)} files)",
        show_header=True,
        header_style='bold cyan',
    )
    table.add_column('File', style='white', no_wrap=True, max_width=30)
    table.add_column('CatID', style='yellow', no_wrap=True)
    table.add_column('Category', style='cyan')
    table.add_column('SubCategory', style='green')
    table.add_column('Profile', style='blue')
    table.add_column('Conf', justify='right')
    table.add_column('FXName', max_width=40)

    for record in sorted(records, key=lambda item: (item.category, item.subcategory)):
        table.add_row(
            record.file_name,
            record.cat_id,
            record.category,
            record.subcategory,
            record.analysis_provenance.profile_name,
            f"{record.confidence:.2f}",
            record.fx_name[:40] + ('…' if len(record.fx_name) > 40 else ''),
        )

    console.print(table)


def _resource_cache_key(config: dict) -> tuple:
    return (
        config.get('model_id'),
        config.get('device'),
        config.get('fp16'),
        config.get('label_cache_path'),
        config.get('vocab_sha256'),
    )
