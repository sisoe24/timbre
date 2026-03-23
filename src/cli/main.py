"""Top-level timbre CLI with subcommands."""

from __future__ import annotations

import click

from .batch import main as batch_command
from .vocab import main as vocab_command
from .analyze import main as analyze_command
from .validate import main as validate_command


@click.group()
def main() -> None:
    """Timbre command-line interface."""


main.add_command(analyze_command, name='analyze')
main.add_command(batch_command, name='batch')
main.add_command(vocab_command, name='vocab')
main.add_command(validate_command, name='validate')
