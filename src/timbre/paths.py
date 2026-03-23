"""Project-root path helpers that work in source and installed layouts."""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []

    env_root = os.environ.get('TIMBRE_PROJECT_ROOT')
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[2])

    return candidates


def resolve_project_root() -> Path:
    """Resolve the project root for config and cache discovery."""
    for candidate in _candidate_roots():
        if (candidate / 'config' / 'config.yaml').exists():
            return candidate.resolve()

    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()
