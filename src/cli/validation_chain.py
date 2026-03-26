"""Shared CLI helpers for opt-in analysis-to-validation chaining."""

from __future__ import annotations

from pathlib import Path

import click


def add_validation_chain_options(func):
    """Attach chained validation options to an analysis command."""
    options = [
        click.option(
            '--validate',
            'validate_output',
            is_flag=True,
            default=False,
            help='Run LLM validation against generated JSON after analysis completes',
        ),
        click.option(
            '--validate-backend',
            type=click.Choice(['ollama', 'openai', 'anthropic']),
            default='ollama',
            show_default=True,
            help='LLM backend to use for chained validation',
        ),
        click.option(
            '--validate-model',
            default=None,
            help='Model name to use for chained validation',
        ),
        click.option(
            '--validate-mode',
            type=click.Choice(['audit', 'autocorrect']),
            default='audit',
            show_default=True,
            help='Validation mode for chained validation',
        ),
        click.option(
            '--validate-temp',
            default=0.1,
            type=float,
            show_default=True,
            help='Validation model temperature if supported',
        ),
        click.option(
            '--validate-report',
            default=None,
            type=click.Path(path_type=Path),
            help='Path to save the chained validation report',
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func


def ensure_validate_report_is_unambiguous(
    profiles_to_run: list[str | None],
    validate_report: Path | None,
) -> None:
    """Reject a single explicit report path for multi-profile chained runs."""
    if validate_report is not None and len(profiles_to_run) > 1:
        raise click.UsageError(
            '--validate-report cannot be used with multiple profiles; '
            'run one profile at a time or omit the flag to use per-profile defaults.'
        )
