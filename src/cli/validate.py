"""
LLM-as-Judge validator for generated Timbre analysis records.

Offline mode reads saved JSON artifacts from disk. Inline validation in
`analyze` and `batch` calls the same validation helpers on in-memory records
before writing outputs.
"""

from __future__ import annotations

import sys
import json
import logging
from typing import Any, Iterable
from pathlib import Path

import click
from rich.table import Table
from rich.console import Console

from timbre.llm.client import complete_json
from timbre.output_paths import resolve_output_paths
from timbre.config_loader import load_config

console = Console()
logger = logging.getLogger(__name__)

TEMP = 0.1

SYSTEM_PROMPT = """\
You are an expert audio metadata reviewer specialising in the Universal Category System (UCS) v8.2.1.

Your job is to review a single audio analysis record and check it for:
1. Keyword relevance  — do the keywords accurately reflect the description and sound events?
2. Keyword redundancy — are any keywords duplicates or near-duplicates?
3. Category / subcategory fit — does the UCS category and subcategory match the evidence bundle?
4. fx_name accuracy — does the short title (~25 chars) correctly summarise the sound?
5. sound_events consistency — do the temporal events match what the evidence and description say?
6. Confidence plausibility — is the confidence score reasonable given the evidence quality?
7. Mapping diagnostics — do the conflict flags and alternatives indicate unresolved ambiguity?

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
    """Format a record as a validation prompt."""
    relevant = {k: record.get(k) for k in [
        'file_name', 'category', 'subcategory', 'cat_id', 'category_full',
        'fx_name', 'description', 'keywords', 'sound_events', 'confidence',
        'evidence', 'description_details', 'mapping_diagnostics', 'llm_provenance',
    ]}
    return (
        'Please review this audio analysis record:\n\n'
        f"```json\n{json.dumps(relevant, indent=2)}\n```"
    )


def query_ollama(record: dict, model: str = 'llama3.1:8b', temp: float = TEMP) -> dict:
    payload, _ = complete_json(
        backend='ollama',
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_message(record),
        temperature=temp,
        retries=1,
    )
    return payload


def query_openai(record: dict, model: str = 'gpt-4o', temp: float = TEMP) -> dict:
    payload, _ = complete_json(
        backend='openai',
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_message(record),
        temperature=temp,
        retries=1,
    )
    return payload


def query_anthropic(record: dict, model: str = 'claude-sonnet-4-6', temp: float = TEMP) -> dict:
    payload, _ = complete_json(
        backend='anthropic',
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_message(record),
        temperature=temp,
        retries=1,
    )
    return payload


def load_records(input_path: Path) -> list[tuple[Path, dict]]:
    """Load one or more JSON records from a file or directory."""
    records: list[tuple[Path, dict]] = []
    if input_path.is_file():
        with open(input_path, encoding='utf-8') as f:
            records.append((input_path, json.load(f)))
    elif input_path.is_dir():
        for p in sorted(input_path.glob('*.json')):
            with open(p, encoding='utf-8') as f:
                records.append((p, json.load(f)))
    else:
        console.print(f"[red]Input path not found: {input_path}[/red]")
        sys.exit(1)
    return records


def apply_corrections(original: dict, validation: dict) -> dict:
    """Merge validator suggestions into a corrected record copy."""
    corrected = original.copy()
    if validation.get('suggested_keywords'):
        corrected['keywords'] = validation['suggested_keywords']
    if validation.get('suggested_category'):
        corrected['category'] = validation['suggested_category']
    if validation.get('suggested_subcategory'):
        corrected['subcategory'] = validation['suggested_subcategory']
    if validation.get('suggested_fx_name'):
        corrected['fx_name'] = validation['suggested_fx_name']
    if validation.get('suggested_filename'):
        corrected['suggested_filename'] = validation['suggested_filename']
    return corrected


def print_summary(results: list[dict]) -> None:
    """Print a rich validation summary table."""
    table = Table(title='CLAP Validation Summary', show_lines=True)
    table.add_column('File', style='cyan', no_wrap=True)
    table.add_column('Score', justify='center')
    table.add_column('Issues', justify='center')
    table.add_column('Notes', style='dim')

    for result in results:
        score = result.get('consistency_score', 0.0)
        score_str = f"{score:.2f}"
        color = 'green' if score >= 0.85 else ('yellow' if score >= 0.6 else 'red')
        issues = len(result.get('issues', []))
        table.add_row(
            result.get('file_name', '?'),
            f"[{color}]{score_str}[/{color}]",
            str(issues),
            result.get('notes', '')[:80],
        )

    console.print(table)


def validate_record(
    record: Any,
    *,
    backend: str,
    model: str | None,
    mode: str,
    temp: float = TEMP,
) -> tuple[dict, dict]:
    """Validate a single in-memory record and return `(validation, original_dict)`."""
    record_dict = _coerce_record_dict(record)
    default_models = {
        'ollama': 'qwen3.5-validator',
        'openai': 'gpt-4o',
        'anthropic': 'claude-sonnet-4-6',
    }
    selected_model = model or default_models[backend]
    query_fn = {
        'ollama': query_ollama,
        'openai': query_openai,
        'anthropic': query_anthropic,
    }[backend]
    validation = query_fn(record_dict, model=selected_model, temp=temp)
    validation['file_name'] = record_dict.get('file_name', validation.get('file_name', '?'))
    validation['backend'] = backend
    validation['model'] = selected_model
    validation['mode'] = mode
    validation['analysis_provenance'] = record_dict.get('analysis_provenance', {})
    return validation, record_dict


def validate_records(
    records: Iterable[Any],
    *,
    backend: str,
    model: str | None,
    mode: str,
    temp: float = TEMP,
) -> list[tuple[dict, dict]]:
    """Validate several in-memory records."""
    return [
        validate_record(record, backend=backend, model=model, mode=mode, temp=temp)
        for record in records
    ]


def maybe_write_validation_report(
    results: list[dict],
    *,
    report: Path | None,
    config: dict | None = None,
) -> Path | None:
    """Write a validation report only when explicitly requested or configured."""
    should_write = report is not None or bool(
        (config or {}).get('output', {}).get('save_validation_report'))
    if not should_write:
        return None

    report_path = report
    if report_path is None:
        if config is None:
            raise ValueError('A config dict is required when report is not explicitly provided.')
        report_path = resolve_output_paths(config)['validation_report']

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return report_path


def run_validation(
    input_path: Path,
    backend: str,
    model: str | None,
    mode: str,
    report: Path | None,
    config: str | Path | None,
    profile: str | None,
    temp: float = TEMP,
) -> None:
    """Run offline validation against saved JSON files."""
    cfg = load_config(config_path=config, profile_name=profile)
    records = load_records(input_path)

    inferred_profile = _infer_profile_name(records)
    if profile is None and inferred_profile:
        try:
            cfg = load_config(config_path=config, profile_name=inferred_profile)
        except ValueError:
            pass

    default_models = {
        'ollama': 'qwen3.5-validator',
        'openai': 'gpt-4o',
        'anthropic': 'claude-sonnet-4-6',
    }
    selected_model = model or default_models[backend]

    console.print(
        f"\n[bold]Validating {len(records)} record(s): {backend} / {selected_model} / "
        f"profile: {cfg['profile_name']} / temp: {temp}[/bold]\n"
    )

    all_results: list[dict] = []
    corrected_records: list[tuple[Path, dict]] = []

    for path, record in records:
        file_name = record.get('file_name', path.name)
        console.print(f"  Validating [cyan]{file_name}[/cyan]...", end=' ')
        try:
            validation, record_dict = validate_record(
                record,
                backend=backend,
                model=selected_model,
                mode=mode,
                temp=temp,
            )
            all_results.append(validation)
            score = validation.get('consistency_score', 0.0)
            issues = len(validation.get('issues', []))
            console.print(f"score={score:.2f}  issues={issues}")

            if mode == 'autocorrect':
                corrected_records.append((path, apply_corrections(record_dict, validation)))
        except Exception as exc:
            console.print(f"[red]ERROR: {exc}[/red]")
            all_results.append({
                'file_name': file_name,
                'backend': backend,
                'model': selected_model,
                'error': str(exc),
            })

    console.print()
    print_summary(all_results)

    report_path = report
    if report_path is None:
        report_path = _default_report_path(
            resolve_output_paths(cfg)['validation_report'],
            input_path,
            records,
        )
    maybe_write_validation_report(all_results, report=report_path)
    console.print(f"\n[green]Report saved:[/green] {report_path}")

    if mode == 'autocorrect' and corrected_records:
        corrected_dir = input_path.parent / 'corrected'
        corrected_dir.mkdir(exist_ok=True)
        for orig_path, corrected in corrected_records:
            out_path = corrected_dir / orig_path.name
            with open(out_path, 'w', encoding='utf-8') as f:
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
    '--profile',
    default=None,
    help='Named profile to load from config.yaml',
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
    profile: str | None,
    mode: str,
    report: Path | None,
    temp: float = TEMP,
) -> None:
    """Validate previously saved JSON analysis records."""
    run_validation(
        input_path=input_path,
        backend=backend,
        model=model,
        config=config,
        profile=profile,
        temp=temp,
        mode=mode,
        report=report,
    )


def _coerce_record_dict(record: Any) -> dict:
    if isinstance(record, dict):
        return record
    if hasattr(record, 'to_full_dict'):
        return record.to_full_dict()
    if hasattr(record, 'model_dump'):
        return record.model_dump()
    raise TypeError(f'Unsupported record type for validation: {type(record)!r}')


def _infer_profile_name(records: list[tuple[Path, dict]]) -> str | None:
    names = {
        record.get('analysis_provenance', {}).get('profile_name')
        for _, record in records
        if record.get('analysis_provenance', {}).get('profile_name')
    }
    if len(names) == 1:
        return next(iter(names))
    return None


def _default_report_path(
    configured_report_path: Path,
    input_path: Path,
    records: list[tuple[Path, dict]],
) -> Path:
    report_dir = configured_report_path.parent

    if len(records) == 1:
        _, record = records[0]
        source_name = record.get('file_name') or input_path.name
        return report_dir / f'{Path(source_name).stem}.json'

    stem = input_path.name if input_path.is_dir() else input_path.stem
    stem = stem or configured_report_path.stem
    return report_dir / f'{stem}_validation_report.json'


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
