from __future__ import annotations

import json
from pathlib import Path
from argparse import Namespace

import yaml
import pytest

from scripts import enrich_vocabulary as enrich


@pytest.mark.parametrize('saved_batch,explicit', [
    ('batch-1', False), ('batch-1', True), ('other-batch', True), (None, True),
])
def test_retrieval_uses_matching_state_and_preserves_unrelated_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved_batch: str | None,
    explicit: bool,
) -> None:
    """Explicit retrieval reuses matching paths without deleting another job's state."""
    vocab_path = tmp_path / 'vocabulary.yaml'
    output_path = tmp_path / 'enriched.yaml'
    state_path = tmp_path / 'state.json'
    vocab_path.write_text(yaml.safe_dump({
        'categories': {'IMPACTS': {'METAL': {'cat_id': 'IMPMtl', 'labels': ['metal hit']}}},
    }))
    if saved_batch:
        state_path.write_text(json.dumps({
            'batch_id': saved_batch, 'backend': 'openai',
            'vocab_path': str(vocab_path), 'output_path': str(output_path),
            'weak_keys': ['IMPACTS/METAL'],
        }))
    monkeypatch.setattr(enrich, 'STATE_FILE', state_path)
    calls = []

    def retrieve(batch_id: str, poll_interval: int) -> dict[str, list[str]]:
        calls.append(batch_id)
        return {'IMPACTS/METAL': ['a bright metallic ring fading slowly away']}

    monkeypatch.setattr(enrich, 'retrieve_batch_openai', retrieve)
    supply_paths = saved_batch != 'batch-1'
    enrich.cmd_retrieve(Namespace(
        batch_id='batch-1' if explicit else None,
        backend='openai' if explicit else None,
        vocab=str(vocab_path) if supply_paths else None,
        output=str(output_path) if supply_paths else None,
        poll_interval=1,
    ))

    result = yaml.safe_load(output_path.read_text())
    assert len(result['categories']['IMPACTS']['METAL']['labels']) == 2
    assert calls == ['batch-1']
    assert state_path.exists() == (saved_batch == 'other-batch')


def test_retrieval_without_paths_or_state_reports_missing_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Missing recovery paths fail clearly before any provider request."""
    monkeypatch.setattr(enrich, 'STATE_FILE', tmp_path / 'missing.json')
    args = Namespace(batch_id='batch-1', backend='openai', vocab=None, output=None, poll_interval=1)

    with pytest.raises(SystemExit, match='--vocab, and --output'):
        enrich.cmd_retrieve(args)
