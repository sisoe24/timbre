"""CLI commands for inspecting configured analysis profiles."""

from __future__ import annotations

import json

import yaml
import click
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from timbre.paths import PROJECT_ROOT
from timbre.config_loader import (load_config, get_profile_catalog,
                                  get_profile_definition,
                                  get_default_profile_name)

console = Console()


@click.group()
def main() -> None:
    """Inspect configured profile definitions."""


@main.command('list')
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
def list_command(config: str) -> None:
    """List configured profiles with labels and descriptions."""
    profiles = get_profile_catalog(config)
    default_profile = get_default_profile_name(config)

    table = Table(title='Configured Profiles', show_lines=True)
    table.add_column('Default', justify='center')
    table.add_column('Name', style='cyan', no_wrap=True)
    table.add_column('Label', style='white')
    table.add_column('Description', style='dim')

    for profile in profiles:
        table.add_row(
            '*' if profile['name'] == default_profile else '',
            profile['name'],
            profile.get('label', ''),
            profile.get('description', ''),
        )

    console.print(table)


@main.command('inspect')
@click.argument('profile_name', required=False)
@click.option(
    '--config',
    default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--vocab',
    default=None,
    help='Path to vocabulary.yaml (overrides active/config selection)',
)
@click.option(
    '--json',
    'as_json',
    is_flag=True,
    default=False,
    help='Print profile information as JSON',
)
def inspect_command(
    profile_name: str | None,
    config: str,
    vocab: str | None,
    as_json: bool,
) -> None:
    """Inspect one configured profile. Defaults to the config default profile."""
    definition = get_profile_definition(config, profile_name)
    selected_name = definition['metadata']['name']
    cfg = load_config(config_path=config, vocab_path=vocab, profile_name=selected_name)

    payload = {
        'name': selected_name,
        'label': definition['metadata'].get('label', ''),
        'description': definition['metadata'].get('description', ''),
        'is_default': definition['metadata'].get('is_default', False),
        'profile_fingerprint': cfg.get('profile_fingerprint'),
        'model_id': cfg.get('model_id'),
        'device': cfg.get('device'),
        'fp16': cfg.get('fp16'),
        'windowed_analysis': cfg.get('use_windowed_analysis'),
        'windowed_min_duration': cfg.get('windowed_min_duration'),
        'window_seconds': cfg.get('window_seconds'),
        'hop_seconds': cfg.get('hop_seconds'),
        'min_confidence': cfg.get('min_confidence'),
        'top_k_categories': cfg.get('top_k_categories'),
        'target_sr': cfg.get('target_sr'),
        'vocab_file': cfg.get('vocab_file'),
        'overrides': definition['overrides'],
    }
    if as_json:
        console.print_json(json.dumps(payload, indent=2))
        return

    panel_lines = [
        f"[bold]{payload['label'] or selected_name}[/bold]",
        f"Name: [cyan]{selected_name}[/cyan]",
        f"Default: {'yes' if payload['is_default'] else 'no'}",
        f"Fingerprint: [magenta]{payload['profile_fingerprint']}[/magenta]",
    ]
    if payload['description']:
        panel_lines.append(f"Description: {payload['description']}")
    console.print(Panel.fit('\n'.join(panel_lines), title='Profile'))

    settings = Table(title='Effective Settings', show_lines=True)
    settings.add_column('Field', style='cyan', no_wrap=True)
    settings.add_column('Value', style='white')
    settings.add_row('Model', str(payload['model_id']))
    settings.add_row('Device', str(payload['device']))
    settings.add_row('fp16', str(payload['fp16']))
    settings.add_row('Windowed analysis', str(payload['windowed_analysis']))
    settings.add_row('Windowed min duration', str(payload['windowed_min_duration']))
    settings.add_row('Window seconds', str(payload['window_seconds']))
    settings.add_row('Hop seconds', str(payload['hop_seconds']))
    settings.add_row('Min confidence', str(payload['min_confidence']))
    settings.add_row('Top-K categories', str(payload['top_k_categories']))
    settings.add_row('Target sample rate', str(payload['target_sr']))
    settings.add_row('Vocabulary file', str(payload['vocab_file']))
    console.print(settings)

    overrides = definition['overrides']
    if overrides:
        console.print(Panel(yaml.safe_dump(overrides, sort_keys=False).rstrip(), title='Overrides'))
    else:
        console.print(Panel('No explicit overrides.', title='Overrides'))
