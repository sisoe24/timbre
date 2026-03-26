"""CLI for analyzing a single audio file."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from timbre.vocab_state import remember_vocab

console = Console()


@click.command()
@click.argument('audio_file', required=False, type=click.Path(exists=True))
@click.option(
    '--output-dir',
    '-o',
    default=None,
    help='Directory to save output files (default: ./out/)',
)
@click.option(
    '--config',
    '-c',
    default=None,
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--vocab',
    '-v',
    default=None,
    help='Path to vocabulary.yaml (default: config/vocabulary.yaml)',
)
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
    '--full',
    is_flag=True,
    default=False,
    help='Save full JSON (with metadata + acoustics) instead of brief spec format',
)
@click.option(
    '--markdown',
    'save_markdown',
    is_flag=True,
    default=False,
    help='Also save a per-file Markdown review report',
)
@click.option(
    '--no-windowed',
    is_flag=True,
    default=False,
    help='Disable sliding-window event detection (faster, less temporal detail)',
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    default=False,
    help='Suppress console output (only errors shown)',
)
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable verbose debug logging, including third-party request logs',
)
def main(
    audio_file: str | None,
    output_dir: str,
    config: str,
    vocab: str,
    profile: tuple[str, ...],
    all_profiles: bool,
    list_profiles: bool,
    full: bool,
    save_markdown: bool,
    no_windowed: bool,
    quiet: bool,
    debug: bool,
) -> None:
    """Analyze a single AUDIO_FILE and produce a catalog-ready description."""

    from timbre.pipeline import AudioAnalysisPipeline
    from timbre.output_paths import resolve_output_paths
    from timbre.config_loader import load_config
    from timbre.config_loader import list_profiles as list_config_profiles
    from timbre.config_loader import (setup_logging, refresh_runtime_metadata,
                                      resolve_requested_profiles)
    from timbre.output.serializer import save_json
    from timbre.output.serializer import save_markdown as save_md
    from timbre.ingestion.audio_loader import load_audio

    if list_profiles:
        names = list_config_profiles(config)
        if names:
            console.print('\n'.join(names))
        else:
            console.print('[yellow]No named profiles configured.[/yellow]')
        return

    if audio_file is None:
        raise click.UsageError('Missing argument: AUDIO_FILE')

    try:
        profiles_to_run = resolve_requested_profiles(
            config_path=config,
            requested_profiles=profile,
            all_profiles=all_profiles,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    shared_resources: dict[tuple, tuple] = {}
    loaded_audio_by_sr = {}

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
        out_dir = output_paths['json_dir']

        if not quiet:
            console.print(
                Panel.fit(
                    f"[bold cyan]Audio Analyzer — UCS[/bold cyan]\n"
                    f"File: [green]{audio_file}[/green]\n"
                    f"Profile: [blue]{cfg['profile_name']}[/blue] "
                    f"[dim]({cfg['profile_fingerprint']})[/dim]\n"
                    f"Model: [yellow]{cfg['model_id']}[/yellow]\n"
                    f"Vocab: [magenta]{vocab_file}[/magenta] "
                    f"[dim]({vocab_sha}, {vocab_source})[/dim]",
                    title='🎧 Analysis',
                )
            )

        pipeline = AudioAnalysisPipeline(cfg)
        resource_key = _resource_cache_key(cfg)
        if resource_key in shared_resources:
            pipeline.tagger, pipeline.cache = shared_resources[resource_key]
            if not quiet:
                console.print('[green]✓[/green] Reusing loaded CLAP model.')
        else:
            if not quiet:
                with console.status('Loading CLAP model…'):
                    pipeline.load_model()
                console.print('[green]✓[/green] Model loaded.')
            else:
                pipeline.load_model()
            shared_resources[resource_key] = (pipeline.tagger, pipeline.cache)

        target_sr = cfg['target_sr']
        if target_sr not in loaded_audio_by_sr:
            loaded_audio_by_sr[target_sr] = load_audio(audio_file, target_sr=target_sr)

        if not quiet:
            with console.status(f"Analyzing {Path(audio_file).name}…"):
                record = pipeline.analyze_file(
                    audio_file,
                    audio_file=loaded_audio_by_sr[target_sr],
                )
        else:
            record = pipeline.analyze_file(
                audio_file,
                audio_file=loaded_audio_by_sr[target_sr],
            )

        json_path = save_json(record, out_dir, full=full)

        if save_markdown or cfg['output'].get('save_per_file_markdown', False):
            save_md(record, output_paths['markdown_dir'])

        if not quiet:
            _print_record(record)
            console.print(f"\n[dim]JSON saved → {json_path}[/dim]")


def _print_record(record) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold]{record.fx_name}[/bold]\n\n{record.description}",
            title=f"[cyan]{record.file_name}[/cyan]",
            subtitle=(
                f"confidence: {record.confidence:.2f}  |  "
                f"{record.cat_id}  |  {record.category_full}"
            ),
        )
    )
    console.print(
        f"\n[bold]UCS:[/bold] [yellow]{record.cat_id}[/yellow]  "
        f"[dim]{record.category} → {record.subcategory}[/dim]"
    )
    console.print(
        f"[bold]Suggested filename:[/bold] [green]{record.suggested_filename}[/green]"
    )
    kw_str = '  '.join(f"[cyan]{k}[/cyan]" for k in record.keywords[:8])
    console.print(f"[bold]Keywords:[/bold] {kw_str}")

    if record.sound_events:
        events_str = ' → '.join(record.sound_events[:6])
        console.print(f"[bold]Events:[/bold] {events_str}")

    table = Table(title='CLAP Classification', show_header=True, header_style='bold')
    table.add_column('Label', style='cyan')
    table.add_column('Score', justify='right')
    table.add_column('UCS Category', style='dim')
    table.add_column('Bar', justify='left')

    for label, score in sorted(
        record.top_labels.items(), key=lambda item: item[1], reverse=True
    )[:8]:
        bar = '█' * int(score * 20)
        table.add_row(label, f"{score:.3f}", record.category, f"[green]{bar}[/green]")

    console.print(table)
    console.print(
        f"\n[dim]Duration: {record.metadata.duration_seconds:.2f}s  |  "
        f"Sample rate: {record.metadata.sample_rate_hz} Hz  |  "
        f"Format: {record.metadata.format.upper()}  |  "
        f"Profile: {record.analysis_provenance.profile_name}  |  "
        f"Creator: {record.creator_id}  |  Source: {record.source_id}  |  "
        f"Vocab: {Path(record.analysis_provenance.vocab_path).name} "
        f"({record.analysis_provenance.vocab_sha256[:12]})[/dim]"
    )


def _resource_cache_key(config: dict) -> tuple:
    return (
        config.get('model_id'),
        config.get('device'),
        config.get('fp16'),
        config.get('label_cache_path'),
        config.get('vocab_sha256'),
    )
