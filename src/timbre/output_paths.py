"""Helpers for profile-scoped output paths."""

from __future__ import annotations

from pathlib import Path

DEFAULT_RELATIVE_OUTPUTS = {
    'json_dir': Path('json'),
    'markdown_dir': Path('markdown'),
    'catalog_markdown': Path('catalog.md'),
    'catalog_csv': Path('catalog.csv'),
    'batch_json': Path('batch_results.json'),
    'validation_report': Path('validation') / 'validation_report.json',
}


def _normalize_relative_path(path: Path) -> Path:
    parts = [part for part in path.parts if part not in ('', '.')]
    return Path(*parts) if parts else Path()


def ensure_profile_output_root(root: str | Path, profile_name: str) -> Path:
    """Append the profile segment unless the path is already scoped."""
    path = Path(root)
    if profile_name in path.parts:
        return path
    return path / profile_name


def _resolve_configured_output_path(
    configured_path: str | None,
    output_root: Path,
    profile_root: Path,
    key: str,
) -> Path:
    if not configured_path:
        return profile_root / DEFAULT_RELATIVE_OUTPUTS[key]

    path = Path(configured_path)
    if profile_root == path or profile_root in path.parents:
        return path

    try:
        rel = path.relative_to(output_root)
    except ValueError:
        rel = _normalize_relative_path(path)

    if not rel.parts:
        rel = DEFAULT_RELATIVE_OUTPUTS[key]

    return profile_root / rel


def resolve_output_paths(
    config: dict,
    explicit_output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve output destinations for the effective profile."""
    profile_name = config.get('profile_name', 'default')
    output_cfg = config.get('output', {})
    output_root = Path(output_cfg.get('output_dir', './out'))

    if explicit_output_dir:
        root = ensure_profile_output_root(explicit_output_dir, profile_name)
        return {
            'root': root,
            'json_dir': root / DEFAULT_RELATIVE_OUTPUTS['json_dir'],
            'markdown_dir': root / DEFAULT_RELATIVE_OUTPUTS['markdown_dir'],
            'catalog_markdown': root / DEFAULT_RELATIVE_OUTPUTS['catalog_markdown'],
            'catalog_csv': root / DEFAULT_RELATIVE_OUTPUTS['catalog_csv'],
            'batch_json': root / DEFAULT_RELATIVE_OUTPUTS['batch_json'],
            'validation_report': root / DEFAULT_RELATIVE_OUTPUTS['validation_report'],
        }

    profile_root = ensure_profile_output_root(output_root, profile_name)
    return {
        'root': profile_root,
        'json_dir': _resolve_configured_output_path(
            output_cfg.get('json_dir'),
            output_root,
            profile_root,
            'json_dir',
        ),
        'markdown_dir': _resolve_configured_output_path(
            output_cfg.get('markdown_dir'),
            output_root,
            profile_root,
            'markdown_dir',
        ),
        'catalog_markdown': _resolve_configured_output_path(
            output_cfg.get('catalog_markdown'),
            output_root,
            profile_root,
            'catalog_markdown',
        ),
        'catalog_csv': _resolve_configured_output_path(
            output_cfg.get('catalog_csv'),
            output_root,
            profile_root,
            'catalog_csv',
        ),
        'batch_json': _resolve_configured_output_path(
            output_cfg.get('batch_json'),
            output_root,
            profile_root,
            'batch_json',
        ),
        'validation_report': _resolve_configured_output_path(
            output_cfg.get('validation_report'),
            output_root,
            profile_root,
            'validation_report',
        ),
    }
