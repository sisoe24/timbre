"""Top-level timbre CLI with subcommands."""

from __future__ import annotations

import click

from scripts.validate_clap import main as validate_command

from .batch import main as batch_command
from .cache import main as cache_command
from .vocab import main as vocab_command
from .analyze import main as analyze_command
from .cache_info import main as cache_info_command

cache_command.hidden = True
cache_info_command.hidden = True


@click.group()
def main() -> None:
    """Timbre command-line interface."""


main.add_command(analyze_command, name='analyze')
main.add_command(batch_command, name='batch')
main.add_command(cache_command, name='cache')
main.add_command(cache_info_command, name='cache-info')
main.add_command(vocab_command, name='vocab')
main.add_command(validate_command, name='validate')
