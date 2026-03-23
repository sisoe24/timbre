"""CLI for analyzing a single audio file."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

console = Console()


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
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
def main(
    audio_file: str,
    output_dir: str,
    config: str,
    vocab: str,
    full: bool,
    save_markdown: bool,
    no_windowed: bool,
    quiet: bool,
) -> None:
    """Analyze a single AUDIO_FILE and produce a catalog-ready description."""

    from timbre.pipeline import AudioAnalysisPipeline
    from timbre.config_loader import load_config, setup_logging
    from timbre.output.serializer import save_json
    from timbre.output.serializer import save_markdown as save_md

    cfg = load_config(config_path=config, vocab_path=vocab)
    if no_windowed:
        cfg['use_windowed_analysis'] = False

    setup_logging(cfg)

    out_dir = Path(output_dir or cfg['output'].get('json_dir', './outputs/json'))

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold cyan]Audio Analyzer — UCS[/bold cyan]\n"
                f"File: [green]{audio_file}[/green]\n"
                f"Model: [yellow]{cfg['model_id']}[/yellow]",
                title='🎧 Analysis',
            )
        )

    pipeline = AudioAnalysisPipeline(cfg)

    if not quiet:
        with console.status('Loading CLAP model…'):
            pipeline.load_model()
        console.print('[green]✓[/green] Model loaded.')
    else:
        pipeline.load_model()

    if not quiet:
        with console.status(f"Analyzing {Path(audio_file).name}…"):
            record = pipeline.analyze_file(audio_file)
    else:
        record = pipeline.analyze_file(audio_file)

    json_path = save_json(record, out_dir, full=full)

    if save_markdown or cfg['output'].get('save_per_file_markdown', False):
        md_dir = Path(cfg['output'].get('markdown_dir', './outputs/markdown'))
        save_md(record, md_dir)

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
        f"Creator: {record.creator_id}  |  Source: {record.source_id}[/dim]"
    )
