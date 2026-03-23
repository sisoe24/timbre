"""Persistent vocabulary selection state for CLI workflows."""

from __future__ import annotations

import json
import hashlib
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from .paths import PROJECT_ROOT

STATE_PATH = PROJECT_ROOT / '.cache' / 'vocab_state.json'


def _default_state() -> dict[str, Any]:
    return {
        'active_vocab_path': None,
        'known_vocabs': [],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return _default_state()


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding='utf-8')


def get_active_vocab_path() -> Path | None:
    raw = load_state().get('active_vocab_path')
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def remember_vocab(path: str | Path, make_active: bool = False) -> None:
    remember_vocab_with_metadata(path, make_active=make_active)


def remember_vocab_with_metadata(
    path: str | Path,
    make_active: bool = False,
    managed: bool = False,
    source: str = 'seen',
) -> None:
    vocab_path = Path(path).resolve()
    state = load_state()
    known = state.get('known_vocabs', [])

    entry = {
        'path': str(vocab_path),
        'name': vocab_path.name,
        'exists': vocab_path.exists(),
        'last_used_at': _now(),
        'sha256': _sha256_file(vocab_path) if vocab_path.exists() else None,
        'managed': managed,
        'source': source,
    }

    known = [item for item in known if item.get('path') != str(vocab_path)]
    known.append(entry)
    known.sort(key=lambda item: item.get('last_used_at', ''), reverse=True)
    state['known_vocabs'] = known

    if make_active:
        state['active_vocab_path'] = str(vocab_path)

    save_state(state)


def clear_active_vocab() -> None:
    state = load_state()
    state['active_vocab_path'] = None
    save_state(state)


def list_known_vocabs() -> list[dict[str, Any]]:
    state = load_state()
    active = state.get('active_vocab_path')
    items = []
    for item in state.get('known_vocabs', []):
        record = dict(item)
        record['exists'] = Path(record.get('path', '')).exists()
        record['is_active'] = record.get('path') == active
        items.append(record)
    return items
