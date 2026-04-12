"""CLI for analyzing a single audio file."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from timbre.vocab_state import remember_vocab
from timbre.output.schema import ValidationSummary

from .validation_chain import add_validation_chain_options

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
    help='Advanced override for vocabulary.yaml',
)
@click.option(
    '--profile',
    default=None,
    help='Optional named profile from config.yaml',
)
@click.option(
    '--full',
    is_flag=True,
    default=False,
    help='Save full JSON (with metadata + acoustics) instead of brief spec format',
)
@click.option(
    '--no-windowed',
    is_flag=True,
    default=False,
    help='Disable sliding-window event detection',
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
@add_validation_chain_options
def main(
    audio_file: str | None,
    output_dir: str | None,
    config: str | None,
    vocab: str | None,
    profile: str | None,
    full: bool,
    no_windowed: bool,
    quiet: bool,
    debug: bool,
    validate_output: bool,
    validate_backend: str,
    validate_model: str | None,
    validate_mode: str,
    validate_temp: float,
    validate_report: Path | None,
) -> None:
    """Analyze one AUDIO_FILE and save the generated catalog record."""

    from timbre.pipeline import AudioAnalysisPipeline
    from timbre.output_paths import resolve_output_paths
    from timbre.config_loader import (load_config, setup_logging,
                                      refresh_runtime_metadata)
    from timbre.output.serializer import save_json
    from timbre.ingestion.audio_loader import load_audio

    from .validate import validate_record, maybe_write_validation_report

    if audio_file is None:
        raise click.UsageError('Missing argument: AUDIO_FILE')

    cfg = load_config(config_path=config, vocab_path=vocab, profile_name=profile)
    if no_windowed:
        cfg['use_windowed_analysis'] = False
        refresh_runtime_metadata(cfg)

    setup_logging(cfg, debug=debug)
    remember_vocab(cfg['vocab_path'], make_active=bool(vocab))

    output_paths = resolve_output_paths(cfg, explicit_output_dir=output_dir)
    out_dir = output_paths['json_dir']

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold cyan]Timbre Analyze[/bold cyan]\n"
                f"File: [green]{audio_file}[/green]\n"
                f"Profile: [blue]{cfg['profile_name']}[/blue] "
                f"[dim]({cfg['profile_fingerprint']})[/dim]\n"
                f"Model: [yellow]{cfg['model_id']}[/yellow]",
                title='Analysis',
            )
        )

    pipeline = AudioAnalysisPipeline(cfg)
    if not quiet:
        with console.status('Loading CLAP model…'):
            pipeline.load_model()
        console.print('[green]✓[/green] Model loaded.')
    else:
        pipeline.load_model()

    loaded_audio = load_audio(audio_file, target_sr=cfg['target_sr'])
    if not quiet:
        with console.status(f"Analyzing {Path(audio_file).name}…"):
            record = pipeline.analyze_file(audio_file, audio_file=loaded_audio)
    else:
        record = pipeline.analyze_file(audio_file, audio_file=loaded_audio)

    validation = None
    validation_summary = None
    if validate_output:
        try:
            if not quiet:
                with console.status('Validating generated record…'):
                    validation, _ = validate_record(
                        record,
                        backend=validate_backend,
                        model=validate_model,
                        mode=validate_mode,
                        temp=validate_temp,
                    )
            else:
                validation, _ = validate_record(
                    record,
                    backend=validate_backend,
                    model=validate_model,
                    mode=validate_mode,
                    temp=validate_temp,
                )

            report_path = maybe_write_validation_report(
                [validation],
                report=validate_report,
                config=cfg,
            )
            validation_summary = ValidationSummary(
                backend=validation.get('backend', validate_backend),
                model=validation.get('model', validate_model or 'unknown'),
                mode=validation.get('mode', validate_mode),
                consistency_score=validation.get('consistency_score', 0.0),
                issues=validation.get('issues', []),
                notes=validation.get('notes', ''),
                report_path=str(report_path) if report_path is not None else None,
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        if not quiet:
            console.print(
                f"[green]✓[/green] Validation score={validation.get('consistency_score', 0.0):.2f} "
                f"issues={len(validation.get('issues', []))}"
            )
            if report_path is not None:
                console.print(f"[dim]Validation report → {report_path}[/dim]")

    if validation_summary is not None:
        record = record.model_copy(update={'validation_summary': validation_summary})

    json_path = save_json(record, out_dir, full=full)

    if not quiet:
        _print_record(record)
        console.print(f"\n[dim]JSON saved → '{json_path}'[/dim]")


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
    source_id = record.source_id or '—'
    console.print(
        f"\n[dim]Clip:[/dim] "
        f"[dim]length {record.metadata.duration_seconds:.2f}s[/dim]  |  "
        f"[dim]{record.metadata.sample_rate_hz} Hz[/dim]  |  "
        f"[dim]{record.metadata.format.upper()}[/dim]\n"
        f"[dim]Analysis:[/dim] "
        f"[dim]completed in "
        f"{record.analysis_provenance.analysis_elapsed_seconds:.2f}s[/dim]  |  "
        f"[dim]profile {record.analysis_provenance.profile_name}[/dim]  |  "
        f"[dim]creator {record.creator_id}[/dim]  |  "
        f"[dim]source {source_id}[/dim]"
    )
