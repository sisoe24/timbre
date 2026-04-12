"""CLI for batch-analyzing a folder of audio files."""

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
from timbre.output.schema import ValidationSummary

from .validation_chain import add_validation_chain_options

console = Console()


@click.command()
@click.argument('input_dir', required=False, type=click.Path(exists=True, file_okay=False))
@click.option(
    '--output-dir',
    '-o',
    default=None,
    help='Root output directory (default: ./out/)',
)
@click.option('--config', '-c', default=None, help='Path to config.yaml')
@click.option('--vocab', '-v', default=None, help='Advanced override for vocabulary.yaml')
@click.option(
    '--profile',
    default=None,
    help='Optional named profile from config.yaml',
)
@click.option(
    '--recursive',
    '-r',
    is_flag=True,
    default=True,
    help='Recurse into sub-directories (default: true)',
)
@click.option(
    '--csv',
    'save_csv',
    is_flag=True,
    default=False,
    help='Generate a CSV catalog',
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
    help='Skip files that fail to load, validate, or analyze (default: true)',
)
@click.option('--limit', default=None, type=int, help='Limit to the first N files')
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable verbose debug logging, including third-party request logs',
)
@add_validation_chain_options
def main(
    input_dir: str | None,
    output_dir: str | None,
    config: str | None,
    vocab: str | None,
    profile: str | None,
    recursive: bool,
    save_csv: bool,
    full: bool,
    no_windowed: bool,
    skip_errors: bool,
    limit: int | None,
    debug: bool,
    validate_output: bool,
    validate_backend: str,
    validate_model: str | None,
    validate_mode: str,
    validate_temp: float,
    validate_report: Path | None,
) -> None:
    """Batch analyze all audio files in INPUT_DIR."""

    from timbre.pipeline import AudioAnalysisPipeline
    from timbre.output_paths import resolve_output_paths
    from timbre.config_loader import (load_config, setup_logging,
                                      refresh_runtime_metadata)
    from timbre.output.serializer import save_json, save_json_batch
    from timbre.ingestion.audio_loader import discover_audio_files
    from timbre.output.catalog_builder import build_catalog_csv

    from .validate import validate_record, maybe_write_validation_report

    if input_dir is None:
        raise click.UsageError('Missing argument: INPUT_DIR')

    audio_paths = discover_audio_files(input_dir, recursive=recursive)
    if not audio_paths:
        console.print(f"[red]No supported audio files found in: {input_dir}[/red]")
        sys.exit(1)
    if limit:
        audio_paths = audio_paths[:limit]

    cfg = load_config(config_path=config, vocab_path=vocab, profile_name=profile)
    if no_windowed:
        cfg['use_windowed_analysis'] = False
        refresh_runtime_metadata(cfg)

    setup_logging(cfg, debug=debug)
    remember_vocab(cfg['vocab_path'], make_active=bool(vocab))
    output_paths = resolve_output_paths(cfg, explicit_output_dir=output_dir)
    validation_report_target = None
    if validate_output:
        if validate_report is not None:
            validation_report_target = validate_report
        elif cfg['output'].get('save_validation_report'):
            validation_report_target = output_paths['validation_report']

    console.print(f"\nFound [bold]{len(audio_paths)}[/bold] audio files.\n")
    console.print(
        Panel.fit(
            f"[bold cyan]Timbre Batch[/bold cyan]\n"
            f"Input: [green]{input_dir}[/green]\n"
            f"Output: [yellow]{output_paths['root']}[/yellow]\n"
            f"Profile: [blue]{cfg['profile_name']}[/blue] "
            f"[dim]({cfg['profile_fingerprint']})[/dim]\n"
            f"Model: [yellow]{cfg['model_id']}[/yellow]",
            title='Batch Analysis',
        )
    )

    pipeline = AudioAnalysisPipeline(cfg)
    with console.status('Loading CLAP model…'):
        pipeline.load_model()
    console.print('[green]✓[/green] Model loaded.\n')

    records = []
    validation_results: list[dict] = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task('Analyzing…', total=len(audio_paths))

        for path in audio_paths:
            progress.update(task, description=f"[cyan]{Path(path).name}[/cyan]")
            try:
                record = pipeline.analyze_file(path)
                if validate_output:
                    validation, _ = validate_record(
                        record,
                        backend=validate_backend,
                        model=validate_model,
                        mode=validate_mode,
                        temp=validate_temp,
                    )
                    validation_results.append(validation)
                    record = record.model_copy(update={
                        'validation_summary': ValidationSummary(
                            backend=validation.get('backend', validate_backend),
                            model=validation.get('model', validate_model or 'unknown'),
                            mode=validation.get('mode', validate_mode),
                            consistency_score=validation.get('consistency_score', 0.0),
                            issues=validation.get('issues', []),
                            notes=validation.get('notes', ''),
                            report_path=(
                                str(validation_report_target)
                                if validation_report_target is not None else None
                            ),
                        )
                    })

                records.append(record)
                save_json(record, output_paths['json_dir'], full=full)
            except Exception as exc:
                failed += 1
                if not skip_errors:
                    raise
                console.print(f"[yellow]⚠ Skipped {Path(path).name}: {exc}[/yellow]")
            progress.advance(task)

    if validate_output:
        report_path = maybe_write_validation_report(
            validation_results,
            report=validation_report_target,
            config=cfg,
        )
        if report_path is not None:
            console.print(f"[dim]Validation report → {report_path}[/dim]")

    console.print(
        f"\n[bold green]✓ Analyzed {len(records)}/{len(audio_paths)} files[/bold green]"
        + (f" ([yellow]{failed} failed[/yellow])" if failed else '')
    )

    if not records:
        console.print('[red]No records produced.[/red]')
        sys.exit(1)

    save_json_batch(records, output_paths['batch_json'], full=full)
    console.print(f"[dim]Batch JSON → {output_paths['batch_json']}[/dim]")

    if save_csv:
        build_catalog_csv(records, output_paths['catalog_csv'])
        console.print(f"[dim]CSV       → {output_paths['catalog_csv']}[/dim]")

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
