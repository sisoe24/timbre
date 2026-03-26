"""
LLM-as-Judge validator for CLAP-generated UCS audio analysis records.

Reads JSON output files produced by the audio analyzer and passes each
record to an LLM to check for internal consistency, keyword quality,
category correctness, and description accuracy.

Supports two backends:
  - Ollama  (local, free, requires Ollama server running)
  - OpenAI  (cloud, requires OPENAI_API_KEY env var)

Two modes:
  - audit       : produces a validation report, original files untouched
  - autocorrect : produces corrected JSON records alongside the report

Usage
-----
  # Audit a single file (Ollama):
  timbre validate --input outputs/metal_impact_01.json

  # Audit a directory (OpenAI):
  timbre validate --input outputs/ --backend openai

  # Override the model for the chosen backend:
  timbre validate --input outputs/ --backend openai --model gpt-5.4-mini

  # Auto-correct mode:
  timbre validate --input outputs/ --mode autocorrect

  # Save report to a custom path:
  timbre validate --input outputs/ --report out/validation_report.json
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

import click
from rich.table import Table
from rich.console import Console

from timbre.output_paths import resolve_output_paths
from timbre.config_loader import load_config

console = Console()
logger = logging.getLogger(__name__)


TEMP = 0.1
"""Defualt model temperature for consistent output."""

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert audio metadata reviewer specialising in the Universal Category System (UCS) v8.2.1.

Your job is to review a single audio analysis record and check it for:
1. Keyword relevance  — do the keywords accurately reflect the description and sound events?
2. Keyword redundancy — are any keywords duplicates or near-duplicates? (e.g. "impact" and "metallic impact")
3. Category / subcategory fit — does the UCS category and subcategory match the description?
4. fx_name accuracy — does the short title (~25 chars) correctly summarise the sound?
5. sound_events consistency — do the temporal events match what the description says?
6. Confidence plausibility — is the confidence score reasonable given the description quality?

UCS reference (top-level categories):
    AIR, AIRCRAFT, ALARMS, AMBIENCE, ANIMALS, ARCHIVED, BEEPS, BELLS, BIRDS,
    BOATS, BULLETS, CARTOON, CERAMICS, CHAINS, CHEMICALS, CLOCKS, CLOTH, COMMUNICATIONS,
    COMPUTERS, CREATURES, CROWDS, DESIGNED, DESTRUCTION, DIRT & SAND, DOORS, DRAWERS,
    ELECTRICITY, EQUIPMENT, EXPLOSIONS, FARTS, FIGHT, FIRE, FIREWORKS, FOLEY, FOOD & DRINK,
    FOOTSTEPS, GAMES, GEOTHERMAL, GLASS, GORE, GUNS, HORNS, HUMAN, ICE, LASERS, LEATHER, LIQUID & MUD,
    MACHINES, MAGIC, MECHANICAL, METAL, MOTORS, MOVEMENT, MUSICAL, NATURAL DISASTER, OBJECTS,
    PAPER, PLASTIC, RAIN, ROBOTS, ROCKS, ROPE, RUBBER, SCIFI, SNOW, SPORTS, SWOOSHES, TOOLS,
    TOYS, TRAINS, USER INTERFACE, VEGETATION, VEHICLES, VOICES, WATER, WEAPONS, WEATHER, WHISTLES,
    WIND, WINDOWS, WINGS, WOOD

Return ONLY valid JSON with this exact structure (no markdown, no explanation outside the JSON):
{
    "consistency_score": <float 0.0-1.0>,
    "file_name": "<same as input>",
    "issues": ["<issue 1>", "<issue 2>"],
    "notes": "<brief overall comment>",
    "suggested_category": "<UCS category>",
    "suggested_filename": "<UCS compliant filename if not already present>",
    "suggested_fx_name": "<short title ~25 chars>",
    "suggested_keywords": ["<kw1>", "<kw2>"],
    "suggested_subcategory": "<UCS subcategory>"
}

If nothing is wrong, return an empty issues list and consistency_score of 1.0.
"""


def build_user_message(record: dict) -> str:
    """Format the record as a clean prompt message."""
    relevant = {k: record.get(k) for k in [
        'file_name', 'category', 'subcategory', 'cat_id', 'category_full',
        'fx_name', 'description', 'keywords', 'sound_events', 'confidence',
    ]}
    return f"Please review this audio analysis record:\n\n```json\n{json.dumps(relevant, indent=2)}\n```"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def query_ollama(record: dict, model: str = 'llama3.1:8b', temp: float = TEMP) -> dict:
    """Send a record to Ollama and return the parsed validation result."""
    try:
        import ollama
    except ImportError:
        console.print('[red]ollama package not installed. Run: pip install ollama[/red]')
        sys.exit(1)

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_message(record)},
        ],
        options={'temperature': temp},  # low temp for consistent structured output
    )
    return _parse_llm_response(response['message']['content'])


def query_openai(record: dict, model: str = 'gpt-4o', temp: float = TEMP) -> dict:
    """Send a record to OpenAI and return the parsed validation result."""
    try:
        from openai import OpenAI
    except ImportError:
        console.print('[red]openai package not installed. Run: pip install openai[/red]')
        sys.exit(1)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        console.print('[red]OPENAI_API_KEY environment variable not set.[/red]')
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_message(record)},
        ],
        temperature=temp,
        response_format={'type': 'json_object'},  # enforces JSON output
    )
    return _parse_llm_response(response.choices[0].message.content)


def query_anthropic(record: dict, model: str = 'claude-sonnet-4-6', temp: float = TEMP) -> dict:
    """Send a record to Anthropic Claude and return the parsed validation result."""
    try:
        import anthropic
    except ImportError:
        console.print('[red]anthropic package not installed. Run: pip install anthropic[/red]')
        sys.exit(1)

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        console.print('[red]ANTHROPIC_API_KEY environment variable not set.[/red]')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': build_user_message(record)}],
    )
    return _parse_llm_response(response.content[0].text)


def _parse_llm_response(raw: str) -> dict:
    """Extract and parse JSON from the LLM response."""
    raw = raw.strip()
    # Strip markdown code fences if model ignored the instruction
    if raw.startswith('```'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}\nRaw:\n{raw}")
        return {'error': 'Failed to parse LLM response', 'raw': raw}


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def load_records(input_path: Path) -> list[tuple[Path, dict]]:
    """Load one or more JSON records from a file or directory."""
    records = []
    if input_path.is_file():
        with open(input_path) as f:
            records.append((input_path, json.load(f)))
    elif input_path.is_dir():
        for p in sorted(input_path.glob('*.json')):
            with open(p) as f:
                records.append((p, json.load(f)))
    else:
        console.print(f"[red]Input path not found: {input_path}[/red]")
        sys.exit(1)
    return records


def apply_corrections(original: dict, validation: dict) -> dict:
    """Merge LLM suggestions back into a corrected record."""
    corrected = original.copy()
    if validation.get('suggested_keywords'):
        corrected['keywords'] = validation['suggested_keywords']
    if validation.get('suggested_category'):
        corrected['category'] = validation['suggested_category']
    if validation.get('suggested_subcategory'):
        corrected['subcategory'] = validation['suggested_subcategory']
    if validation.get('suggested_fx_name'):
        corrected['fx_name'] = validation['suggested_fx_name']
    return corrected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print a rich summary table to the terminal."""
    table = Table(title='CLAP Validation Summary', show_lines=True)
    table.add_column('File', style='cyan', no_wrap=True)
    table.add_column('Score', justify='center')
    table.add_column('Issues', justify='center')
    table.add_column('Notes', style='dim')

    for r in results:
        score = r.get('consistency_score', 0.0)
        score_str = f"{score:.2f}"
        color = 'green' if score >= 0.85 else ('yellow' if score >= 0.6 else 'red')
        issues = len(r.get('issues', []))
        table.add_row(
            r.get('file_name', '?'),
            f"[{color}]{score_str}[/{color}]",
            str(issues),
            r.get('notes', '')[:80],
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(
    input_path: Path,
    backend: str,
    model: str | None,
    mode: str,
    report: Path | None,
    config: str | Path | None,
    experiment: str | None,
    temp: float = TEMP,
) -> None:
    """Run the validation workflow."""
    cfg = load_config(config_path=config, experiment_name=experiment)
    default_models = {
        'ollama': 'qwen3.5',
        'openai': 'gpt-4o',
        'anthropic': 'claude-sonnet-4-6',
    }
    model = model or default_models[backend]

    query_fn = {'ollama': query_ollama, 'openai': query_openai,
                'anthropic': query_anthropic}[backend]

    records = load_records(input_path)

    inferred_experiment = _infer_experiment_name(records)
    if experiment is None and inferred_experiment:
        try:
            cfg = load_config(config_path=config, experiment_name=inferred_experiment)
        except ValueError:
            pass

    console.print(
        f"\n[bold]Validating {len(records)} record(s): {backend} / {model} / "
        f"experiment: {cfg['experiment_name']} / temp: {temp}[/bold]\n")

    all_results = []
    corrected_records = []

    for path, record in records:
        file_name = record.get('file_name', path.name)
        console.print(f"  Validating [cyan]{file_name}[/cyan]...", end=' ')

        try:
            validation = query_fn(record, model=model, temp=temp)
            validation['file_name'] = file_name
            validation['backend'] = backend
            validation['model'] = model
            validation['analysis_provenance'] = record.get('analysis_provenance', {})
            all_results.append(validation)

            score = validation.get('consistency_score', 0.0)
            issues = len(validation.get('issues', []))
            console.print(f"score={score:.2f}  issues={issues}")

            if mode == 'autocorrect':
                corrected = apply_corrections(record, validation)
                corrected_records.append((path, corrected))

        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            all_results.append({
                'file_name': file_name,
                'backend': backend,
                'model': model,
                'error': str(e),
            })

    # Print summary table
    console.print()
    print_summary(all_results)

    # Save report
    if report:
        report_path = report
    else:
        report_path = resolve_output_paths(cfg)['validation_report']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[green]Report saved:[/green] {report_path}")

    # Save corrected records
    if mode == 'autocorrect' and corrected_records:
        corrected_dir = input_path.parent / 'corrected'
        corrected_dir.mkdir(exist_ok=True)
        for orig_path, corrected in corrected_records:
            out_path = corrected_dir / orig_path.name
            with open(out_path, 'w') as f:
                json.dump(corrected, f, indent=2)
        console.print(f"[green]Corrected records saved to:[/green] {corrected_dir}/")

    console.print()


@click.command()
@click.option(
    '--input',
    'input_path',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to a JSON file or directory of JSON files',
)
@click.option(
    '--backend',
    type=click.Choice(['ollama', 'openai', 'anthropic']),
    default='ollama',
    show_default=True,
    help='LLM backend to use for validation',
)
@click.option(
    '--model',
    default=None,
    help='Model name to use for the selected backend',
)
@click.option(
    '--temp',
    default=0.1,
    help='Model temperature if supported. Default 0.1',
)
@click.option(
    '--config',
    '-c',
    default=None,
    help='Path to config.yaml (default: config/config.yaml)',
)
@click.option(
    '--experiment',
    default=None,
    help='Named experiment profile to load from config.yaml',
)
@click.option(
    '--mode',
    type=click.Choice(['audit', 'autocorrect']),
    default='audit',
    show_default=True,
    help='Validation mode',
)
@click.option(
    '--report',
    default=None,
    type=click.Path(path_type=Path),
    help='Path to save the full JSON report',
)
def main(
    input_path: Path,
    backend: str,
    model: str | None,
    config: str | None,
    experiment: str | None,
    mode: str,
    report: Path | None,
    temp: float = TEMP,
) -> None:
    """LLM-as-Judge validator for CLAP audio analysis records."""
    run_validation(
        input_path=input_path,
        backend=backend,
        model=model,
        config=config,
        experiment=experiment,
        temp=temp,
        mode=mode,
        report=report,
    )


def _infer_experiment_name(records: list[tuple[Path, dict]]) -> str | None:
    names = {
        record.get('analysis_provenance', {}).get('experiment_name')
        for _, record in records
        if record.get('analysis_provenance', {}).get('experiment_name')
    }
    if len(names) == 1:
        return next(iter(names))
    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
