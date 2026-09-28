#!/usr/bin/env python3
"""
enrich_vocabulary.py
--------------------
Enrich vocabulary.yaml with LLM-generated acoustic descriptor phrases using
the official async Batch APIs (OpenAI / Anthropic) for 50% cost savings.

Acoustic descriptor = label with > 6 words describing what a sound SOUNDS LIKE.

Two-phase workflow (OpenAI / Anthropic):
    python scripts/enrich_vocabulary.py submit   [options]
    python scripts/enrich_vocabulary.py retrieve [options]

Ollama fallback (no batch API) — runs immediately with ThreadPoolExecutor:
    python scripts/enrich_vocabulary.py submit --backend ollama [options]

Usage examples:
    # Dry run — see which subcategories would be enriched
    python scripts/enrich_vocabulary.py submit --dry-run

    # Submit batch (OpenAI, ANIMALS category only)
    python scripts/enrich_vocabulary.py submit --backend openai --category ANIMALS

    # Retrieve results (auto-reads .enrich_batch_state.json)
    python scripts/enrich_vocabulary.py retrieve

    # Full run
    python scripts/enrich_vocabulary.py submit --backend openai
    python scripts/enrich_vocabulary.py retrieve --output config/vocabulary_enriched.yaml
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import tempfile
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timezone

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACOUSTIC_DESCRIPTOR_MIN_WORDS = 6
DEFAULT_MIN_DESCRIPTORS = 3
STATE_FILE = Path('.enrich_batch_state.json')

SYSTEM_PROMPT = """\
You are an expert audio librarian for a professional sound effects library.
Generate acoustic descriptor phrases — sentences that describe what a sound SOUNDS LIKE,
not what causes it. Focus on: frequency range, texture, attack/decay, rhythm, intensity.
Return only a valid JSON array of strings. No explanation, no markdown, no keys — just
a raw JSON array like ["phrase one", "phrase two", ...].
"""


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def load_vocabulary(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def save_vocabulary(vocab: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        yaml.dump(vocab, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)


def count_acoustic_descriptors(labels: List[str]) -> int:
    return sum(1 for lbl in labels if len(lbl.split()) > ACOUSTIC_DESCRIPTOR_MIN_WORDS)


def audit_vocabulary(
    vocab: dict,
    min_descriptors: int,
    filter_category: str | None,
) -> List[Tuple[str, str, List[str], int]]:
    """Return (category, subcategory, labels, descriptor_count) sorted by priority."""
    weak: list[tuple[str, str, list[str], int]] = []
    categories = vocab.get('categories', vocab)

    for category, subcats in categories.items():
        if filter_category and category.upper() != filter_category.upper():
            continue
        if not isinstance(subcats, dict):
            continue
        for subcategory, subcat_data in subcats.items():
            if isinstance(subcat_data, dict):
                labels = subcat_data.get('labels', [])
            elif isinstance(subcat_data, list):
                labels = subcat_data
            else:
                continue
            if not isinstance(labels, list):
                continue
            desc_count = count_acoustic_descriptors(labels)
            if desc_count < min_descriptors:
                weak.append((category, subcategory, labels, desc_count))

    return sorted(weak, key=lambda item: (item[3], len(item[2])))


def write_back(
    vocab: dict,
    weak: List[Tuple[str, str, List[str], int]],
    results: Dict[str, List[str]],
    output_path: Path,
) -> int:
    """Merge generated phrases into vocab copy and save. Returns enriched count."""
    import copy
    enriched = copy.deepcopy(vocab)
    target = enriched.get('categories', enriched)
    enriched_count = 0

    for category, subcategory, labels, _ in weak:
        key = f'{category}/{subcategory}'
        new_phrases = results.get(key)
        if not new_phrases:
            continue
        if isinstance(target[category][subcategory], dict):
            target[category][subcategory]['labels'] = labels + new_phrases
        else:
            target[category][subcategory] = labels + new_phrases
        enriched_count += 1

    save_vocabulary(enriched, output_path)
    return enriched_count


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(category: str, subcategory: str, labels: List[str]) -> str:
    sample = labels[:10]
    return (
        f'Category: {category} / {subcategory}\n'
        f'Existing labels: {json.dumps(sample)}\n\n'
        'Generate 6 acoustic descriptor phrases (10–20 words each) that describe the audible '
        'character of sounds in this category. Each phrase must describe the sound itself — '
        'texture, pitch, attack, rhythm, environment. Do not repeat existing labels. '
        'Return only a JSON array of strings.'
    )


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.startswith('```'):
        lines = raw.split('\n')
        if len(lines) >= 3:
            raw = '\n'.join(lines[1:-1])
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                parsed = v
                break
    if not isinstance(parsed, list):
        raise ValueError(f'Expected JSON array, got {type(parsed).__name__}')
    return [str(s).strip() for s in parsed if str(s).strip()]


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

def submit_batch_openai(
    weak: List[Tuple[str, str, List[str], int]],
    model: str,
    temperature: float,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('openai package not installed') from exc

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')

    client = OpenAI(api_key=api_key)

    # Build JSONL content in memory
    lines = []
    for category, subcategory, labels, _ in weak:
        custom_id = f'{category}/{subcategory}'
        request = {
            'custom_id': custom_id,
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': _build_user_prompt(category, subcategory, labels)},
                ],
                'temperature': temperature,
            },
        }
        lines.append(json.dumps(request))

    jsonl_bytes = '\n'.join(lines).encode('utf-8')

    print(f'Uploading batch file ({len(lines)} requests, {len(jsonl_bytes):,} bytes)...')
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
        tmp.write(jsonl_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as fh:
            uploaded = client.files.create(file=fh, purpose='batch')
    finally:
        os.unlink(tmp_path)

    print(f'File uploaded: {uploaded.id}')
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint='/v1/chat/completions',
        completion_window='24h',
    )
    print(f'Batch submitted: {batch.id} (status: {batch.status})')
    return batch.id


def retrieve_batch_openai(batch_id: str, poll_interval: int) -> Dict[str, List[str]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('openai package not installed') from exc

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')

    client = OpenAI(api_key=api_key)

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else '?'
        total = batch.request_counts.total if batch.request_counts else '?'
        print(f'  Status: {status} ({completed}/{total} completed)', flush=True)

        if status == 'completed':
            break
        if status in ('failed', 'expired', 'cancelled'):
            raise RuntimeError(f'Batch {batch_id} ended with status: {status}')

        print(f'  Waiting {poll_interval}s...', flush=True)
        time.sleep(poll_interval)

    # Download results
    output_file_id = batch.output_file_id
    if not output_file_id:
        raise RuntimeError('Batch completed but no output_file_id')

    raw_content = client.files.content(output_file_id).content
    results: Dict[str, List[str]] = {}
    errors: list[str] = []

    for line in raw_content.decode('utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj.get('custom_id', '')
        response = obj.get('response', {})
        if response.get('status_code') != 200:
            errors.append(f'{custom_id}: HTTP {response.get("status_code")}')
            continue
        try:
            content = response['body']['choices'][0]['message']['content']
            results[custom_id] = _parse_json_list(content)
        except Exception as exc:  # noqa: BLE001
            errors.append(f'{custom_id}: parse error — {exc}')

    if errors:
        print(f'\nWarning: {len(errors)} failed items:')
        for e in errors:
            print(f'  {e}')

    return results


# ---------------------------------------------------------------------------
# Anthropic Batch API
# ---------------------------------------------------------------------------

def submit_batch_anthropic(
    weak: List[Tuple[str, str, List[str], int]],
    model: str,
    temperature: float,
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError('anthropic package not installed') from exc

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    client = anthropic.Anthropic(api_key=api_key)

    requests = []
    for category, subcategory, labels, _ in weak:
        requests.append({
            'custom_id': f'{category}/{subcategory}',
            'params': {
                'model': model,
                'max_tokens': 1024,
                'temperature': temperature,
                'system': SYSTEM_PROMPT,
                'messages': [
                    {'role': 'user', 'content': _build_user_prompt(category, subcategory, labels)},
                ],
            },
        })

    print(f'Submitting Anthropic batch ({len(requests)} requests)...')
    batch = client.messages.batches.create(requests=requests)
    print(f'Batch submitted: {batch.id} (status: {batch.processing_status})')
    return batch.id


def retrieve_batch_anthropic(batch_id: str, poll_interval: int) -> Dict[str, List[str]]:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError('anthropic package not installed') from exc

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    client = anthropic.Anthropic(api_key=api_key)

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(
            f'  Status: {status} '
            f'(processing={counts.processing}, succeeded={counts.succeeded}, '
            f'errored={counts.errored})',
            flush=True,
        )

        if status == 'ended':
            break

        print(f'  Waiting {poll_interval}s...', flush=True)
        time.sleep(poll_interval)

    results: Dict[str, List[str]] = {}
    errors: list[str] = []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == 'succeeded':
            try:
                content = result.result.message.content[0].text
                results[custom_id] = _parse_json_list(content)
            except Exception as exc:  # noqa: BLE001
                errors.append(f'{custom_id}: parse error — {exc}')
        else:
            errors.append(f'{custom_id}: {result.result.type}')

    if errors:
        print(f'\nWarning: {len(errors)} failed items:')
        for e in errors:
            print(f'  {e}')

    return results


# ---------------------------------------------------------------------------
# Ollama fallback — ThreadPoolExecutor (no batch API)
# ---------------------------------------------------------------------------

def _call_ollama_single(
    category: str, subcategory: str, labels: List[str], model: str, temperature: float,
) -> List[str]:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError('ollama package not installed') from exc

    resp = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _build_user_prompt(category, subcategory, labels)},
        ],
        options={'temperature': temperature},
    )
    raw = resp['message']['content']
    existing_lower = {lbl.lower() for lbl in labels}
    phrases = _parse_json_list(raw)
    return [p for p in phrases if p.lower() not in existing_lower]


def run_ollama_immediate(
    weak: List[Tuple[str, str, List[str], int]],
    model: str,
    temperature: float,
    workers: int,
) -> Dict[str, List[str]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: Dict[str, List[str]] = {}
    total = len(weak)

    def _task(item: tuple) -> tuple[str, list[str]]:
        category, subcategory, labels, _ = item
        phrases = _call_ollama_single(category, subcategory, labels, model, temperature)
        return f'{category}/{subcategory}', phrases

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {executor.submit(_task, item): f'{item[0]}/{item[1]}' for item in weak}
        completed = 0
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            completed += 1
            try:
                k, phrases = future.result()
                results[k] = phrases
                status = f'added {len(phrases)}' if phrases else 'no new phrases'
                print(f'[{completed}/{total}] {k} ... {status}', flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f'[{completed}/{total}] {key} ... FAILED — {exc}', flush=True)

    return results


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------

def save_state(
    batch_id: str,
    backend: str,
    vocab_path: Path,
    output_path: Path,
    weak: List[Tuple[str, str, List[str], int]],
) -> None:
    state = {
        'batch_id': batch_id,
        'backend': backend,
        'submitted_at': datetime.now(timezone.utc).isoformat(),
        'vocab_path': str(vocab_path),
        'output_path': str(output_path),
        'weak_keys': [f'{c}/{s}' for c, s, _, _ in weak],
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f'State saved to {STATE_FILE}')


def load_state() -> dict:
    if not STATE_FILE.exists():
        raise RuntimeError(
            f'No state file found at {STATE_FILE}. Run `submit` first, '
            'or pass --batch-id and --backend explicitly.'
        )
    return json.loads(STATE_FILE.read_text())


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f'ERROR: vocabulary file not found: {vocab_path}', file=sys.stderr)
        sys.exit(1)

    vocab = load_vocabulary(vocab_path)
    weak = audit_vocabulary(vocab, args.min_descriptors, args.category)

    categories = vocab.get('categories', vocab)
    total_subcats = sum(len(s) for s in categories.values() if isinstance(s, dict))
    print(f'Vocabulary: {total_subcats} subcategories total')
    print(f'Weak subcategories (< {args.min_descriptors} acoustic descriptors): {len(weak)}')

    if not weak:
        print('Nothing to enrich.')
        return

    if args.dry_run:
        print('\n--- DRY RUN (no API calls) ---')
        for category, subcategory, labels, desc_count in weak:
            print(f'  {category}/{subcategory}: {desc_count} descriptors, {len(labels)} labels')
        return

    output_path = Path(args.output)
    backend = args.backend.lower()

    if backend == 'ollama':
        print('Ollama backend — running immediately with ThreadPoolExecutor...')
        results = run_ollama_immediate(weak, args.model, args.temperature, workers=8)
        enriched_count = write_back(vocab, weak, results, output_path)
        print(f'\nDone. {enriched_count}/{len(weak)} subcategories enriched.')
        print(f'Draft written to: {output_path}')
        _print_next_steps(args.vocab, str(output_path))
        return

    if backend == 'openai':
        batch_id = submit_batch_openai(weak, args.model, args.temperature)
    elif backend == 'anthropic':
        batch_id = submit_batch_anthropic(weak, args.model, args.temperature)
    else:
        print(f'ERROR: unsupported backend: {backend}', file=sys.stderr)
        sys.exit(1)

    save_state(batch_id, backend, vocab_path, output_path, weak)
    print(f'\nBatch submitted. Run retrieve when ready:')
    print(f'  python scripts/enrich_vocabulary.py retrieve')
    print(f'  # or: python scripts/enrich_vocabulary.py retrieve --batch-id {batch_id}')


def cmd_retrieve(args: argparse.Namespace) -> None:
    state = load_state() if STATE_FILE.exists() else {}
    if ((args.batch_id and args.batch_id != state.get('batch_id'))
            or (args.backend and args.backend.lower() != state.get('backend'))):
        state = {}

    batch_id = args.batch_id or state.get('batch_id')
    backend = args.backend or state.get('backend')
    vocab = args.vocab or state.get('vocab_path')
    output = args.output or state.get('output_path')
    if not all((batch_id, backend, vocab, output)):
        raise SystemExit(
            'Provide --batch-id, --backend, --vocab, and --output, '
            'or use the matching saved batch state for omitted options.'
        )
    backend = backend.lower()
    vocab_path = Path(vocab)
    output_path = Path(output)

    vocab = load_vocabulary(vocab_path)

    # We need weak list to drive write-back — re-audit with same params
    # If state has weak_keys use them, otherwise re-audit (may differ if vocab changed)
    if state and 'weak_keys' in state:
        # Build minimal weak list from state keys for write-back indexing
        categories = vocab.get('categories', vocab)
        weak_from_state: list[tuple[str, str, list[str], int]] = []
        for key in state['weak_keys']:
            parts = key.split('/', 1)
            if len(parts) != 2:
                continue
            cat, subcat = parts
            subcat_data = categories.get(cat, {}).get(subcat, {})
            labels = subcat_data.get('labels', []) if isinstance(subcat_data, dict) else subcat_data
            desc_count = count_acoustic_descriptors(labels) if isinstance(labels, list) else 0
            weak_from_state.append(
                (cat, subcat, labels if isinstance(labels, list) else [], desc_count))
        weak = weak_from_state
    else:
        # Fallback: re-audit
        weak = audit_vocabulary(vocab, DEFAULT_MIN_DESCRIPTORS, None)

    print(f'Polling batch {batch_id} ({backend})...')

    if backend == 'openai':
        results = retrieve_batch_openai(batch_id, args.poll_interval)
    elif backend == 'anthropic':
        results = retrieve_batch_anthropic(batch_id, args.poll_interval)
    else:
        print(f'ERROR: unsupported backend: {backend}', file=sys.stderr)
        sys.exit(1)

    enriched_count = write_back(vocab, weak, results, output_path)
    print(f'\nDone. {enriched_count}/{len(weak)} subcategories enriched.')
    print(f'Draft written to: {output_path}')
    _print_next_steps(str(vocab_path), str(output_path))

    # Clean up state file
    if state and STATE_FILE.exists():
        STATE_FILE.unlink()
        print(f'State file {STATE_FILE} removed.')


def _print_next_steps(vocab_path: str, output_path: str) -> None:
    print('\nNext steps:')
    print(f'  diff {vocab_path} {output_path}')
    print(f'  # Review, then: cp {output_path} {vocab_path}')
    print('  poetry run timbre vocab cache --force')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Enrich vocabulary.yaml using LLM Batch APIs (OpenAI / Anthropic).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- submit ---
    submit_p = subparsers.add_parser('submit', help='Audit vocab and submit batch to LLM API')
    submit_p.add_argument('--vocab', default='config/vocabulary.yaml')
    submit_p.add_argument('--output', default='config/vocabulary_enriched.yaml')
    submit_p.add_argument('--min-descriptors', type=int, default=DEFAULT_MIN_DESCRIPTORS,
                          dest='min_descriptors')
    submit_p.add_argument('--category', help='Process only this UCS category (e.g. ANIMALS)')
    submit_p.add_argument('--backend', default='openai',
                          help='openai | anthropic | ollama (default: openai)')
    submit_p.add_argument('--model', default='gpt-4o-mini')
    submit_p.add_argument('--temperature', type=float, default=0.3)
    submit_p.add_argument('--dry-run', action='store_true', dest='dry_run',
                          help='Print audit without calling API')

    # --- retrieve ---
    retrieve_p = subparsers.add_parser('retrieve', help='Poll and retrieve batch results')
    retrieve_p.add_argument('--batch-id', dest='batch_id',
                            help='Batch ID (auto-read from state file if omitted)')
    retrieve_p.add_argument('--backend', help='Backend used for submission (auto-read if omitted)')
    retrieve_p.add_argument('--vocab', default=None,
                            help='Vocabulary path (auto-read from state file if omitted)')
    retrieve_p.add_argument('--output', default=None,
                            help='Output path (auto-read from state file if omitted)')
    retrieve_p.add_argument('--poll-interval', type=int, default=30, dest='poll_interval',
                            help='Seconds between status polls (default: 30)')

    args = parser.parse_args()

    if args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'retrieve':
        cmd_retrieve(args)


if __name__ == '__main__':
    main()
