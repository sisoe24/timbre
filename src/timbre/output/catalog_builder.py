"""
catalog_builder.py
------------------
Aggregates multiple AudioAnalysisRecord objects into a single
human-readable catalog file.

Catalog formats
---------------
  Markdown : one entry per file, sorted by category then filename
             → ideal for human review (open in any Markdown viewer)
  CSV      : flat table of all records
             → ideal for spreadsheet review or import into databases

The Markdown catalog is also designed to be useful for search:
each entry is a self-contained block with file name, description,
tags, and events — easy to grep or Ctrl+F through.
"""

from __future__ import annotations

import logging
import datetime
from typing import List
from pathlib import Path

from .schema import AudioAnalysisRecord
from .serializer import save_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown catalog
# ---------------------------------------------------------------------------

def build_catalog_markdown(
    records: List[AudioAnalysisRecord],
    output_path: str | Path,
    title: str = 'Audio Catalog',
) -> Path:
    """
    Write a single Markdown file containing all records, sorted by category.

    Parameters
    ----------
    records     : list of analysis results
    output_path : where to write the .md file
    title       : document title

    Returns the path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by UCS category, then by filename
    sorted_records = sorted(records, key=lambda r: (r.category, r.file_name))

    lines: List[str] = _catalog_header(title, len(records))
    lines.extend(_catalog_provenance_summary(sorted_records))

    # Group by UCS category for better navigation
    current_category = None
    for r in sorted_records:
        if r.category != current_category:
            current_category = r.category
            lines.append(f"\n## {current_category}\n")

        lines.extend(_record_catalog_block(r))
        lines.append('')  # blank line between entries

    lines.append(_catalog_footer())

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Saved catalog Markdown (%d records): %s', len(records), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CSV catalog (delegates to serializer.save_csv)
# ---------------------------------------------------------------------------

def build_catalog_csv(
    records: List[AudioAnalysisRecord],
    output_path: str | Path,
) -> Path:
    """
    Write a flat CSV catalog of all records.
    Delegates to serializer.save_csv for consistent column format.
    """
    return save_csv(records, output_path)


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _catalog_header(title: str, count: int) -> List[str]:
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        f"# {title}",
        '',
        f"Generated: {now}  |  Total files: {count}",
        '',
        '---',
        '',
        '## Table of Contents',
        '',
        '_Files are grouped by primary sound category, then sorted alphabetically._',
        '',
        '---',
    ]
    return lines


def _catalog_provenance_summary(records: List[AudioAnalysisRecord]) -> List[str]:
    profiles = {}
    for record in records:
        provenance = record.analysis_provenance
        key = (
            provenance.profile_name,
            provenance.profile_fingerprint,
            provenance.model_id,
            provenance.config_path,
            provenance.vocab_path,
            provenance.vocab_sha256,
            provenance.cache_fingerprint,
        )
        profiles.setdefault(key, 0)
        profiles[key] += 1

    lines = [
        '## Analysis Provenance',
        '',
    ]

    if len(profiles) == 1:
        (
            profile_name,
            profile_fingerprint,
            model_id,
            config_path,
            vocab_path,
            vocab_sha256,
            cache_fingerprint,
        ), file_count = next(iter(profiles.items()))
        lines.extend([
            f"- Files rendered with one analysis profile ({file_count} files).",
            f"- Profile: `{profile_name}`",
            f"- Profile fingerprint: `{profile_fingerprint or '—'}`",
            f"- Model: `{model_id}`",
            f"- Config: `{config_path}`",
            f"- Vocabulary: `{Path(vocab_path).name}`",
            f"- Vocabulary SHA256: `{vocab_sha256}`",
            f"- Cache fingerprint: `{cache_fingerprint or '—'}`",
        ])
    else:
        lines.append(f"- Mixed analysis profiles detected: {len(profiles)}")
        for (
            profile_name,
            profile_fingerprint,
            model_id,
            config_path,
            vocab_path,
            vocab_sha256,
            cache_fingerprint,
        ), file_count in profiles.items():
            lines.append(
                f"- profile=`{profile_name}` | fp=`{profile_fingerprint or '—'}` "
                f"| model=`{model_id}` | config=`{Path(config_path).name}` "
                f"| vocab=`{Path(vocab_path).name}` | "
                f"sha=`{vocab_sha256[:12]}` | cache=`{cache_fingerprint or '—'}` "
                f"| files={file_count}"
            )

    lines.extend([
        '',
        '---',
        '',
    ])
    return lines


def _record_catalog_block(r: AudioAnalysisRecord) -> List[str]:
    """Format a single record as a compact UCS catalog entry block."""
    kw_str = ', '.join(f"`{k}`" for k in r.keywords[:6])
    event_str = ' → '.join(r.sound_events[:5]) if r.sound_events else '—'
    conf_bar = _confidence_bar(r.confidence)
    vocab_file = Path(r.analysis_provenance.vocab_path).name

    return [
        f"### `{r.file_name}`",
        '',
        f"**{r.fx_name}**",
        '',
        f"{r.description}",
        '',
        f"| | |",
        f"|---|---|",
        f"| **CatID** | `{r.cat_id}` |",
        f"| **SubCategory** | {r.subcategory} |",
        f"| **Duration** | {r.metadata.duration_seconds:.2f}s |",
        f"| **Confidence** | {conf_bar} {r.confidence:.2f} |",
        f"| **Events** | {event_str} |",
        f"| **Keywords** | {kw_str} |",
        f"| **Profile** | `{r.analysis_provenance.profile_name}` |",
        f"| **Profile FP** | "
        f"`{r.analysis_provenance.profile_fingerprint or '—'}` |",
        f"| **Model** | `{r.analysis_provenance.model_id}` |",
        f"| **Config** | `{Path(r.analysis_provenance.config_path).name}` |",
        f"| **Vocabulary** | `{vocab_file}` |",
        f"| **Vocab SHA** | `{r.analysis_provenance.vocab_sha256[:12]}` |",
        f"| **Cache FP** | `{r.analysis_provenance.cache_fingerprint or '—'}` |",
        f"| **Suggested Filename** | `{r.suggested_filename}` |",
        '',
        '---',
    ]


def _catalog_footer() -> str:
    return (
        '\n_This catalog was generated by the Audio Analyzer (UCS)._\n'
        '_Metadata conforms to the Universal Category System v8.2 (universalcategorysystem.com)._\n'
        '_Descriptions are based on CLAP zero-shot classification + acoustic feature analysis._\n'
    )


def _confidence_bar(conf: float, width: int = 5) -> str:
    """Simple ASCII confidence bar: ████░ (filled/empty blocks)."""
    filled = round(conf * width)
    empty = width - filled
    return '█' * filled + '░' * empty
