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
            help='Run LLM validation on the generated record before writing outputs',
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
            help='Model name to use for inline validation',
        ),
        click.option(
            '--validate-mode',
            type=click.Choice(['audit', 'autocorrect']),
            default='audit',
            show_default=True,
            help='Validation mode for inline validation',
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
            help='Optional path to save inline validation results',
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func
