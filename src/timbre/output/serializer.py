"""
serializer.py
-------------
Saves an AudioAnalysisRecord (or a list of them) to:
  - JSON  (.json)
  - CSV   (.csv)   — UCS-aligned columns
  - Markdown (.md) — single-record review format with UCS fields

Each function writes to the specified output path and returns the path.
"""

from __future__ import annotations

import csv
import json
import logging
from typing import List
from pathlib import Path

from .schema import AudioAnalysisRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(
    record: AudioAnalysisRecord,
    output_dir: str | Path,
    full: bool = False,
) -> Path:
    """
    Save a single analysis record to JSON.

    Parameters
    ----------
    record      : the analysis result
    output_dir  : directory where the file is written
    full        : if True, include full metadata + acoustics;
                  if False, write only UCS core fields

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(record.file_name).stem
    output_path = output_dir / f"{stem}.json"

    data = record.to_full_dict() if full else record.to_brief_dict()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.debug('Saved JSON: %s', output_path)
    return output_path


def save_json_batch(
    records: List[AudioAnalysisRecord],
    output_path: str | Path,
    full: bool = False,
) -> Path:
    """Save a list of records to a single JSON array file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_full_dict() if full else r.to_brief_dict() for r in records]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info('Saved batch JSON (%d records): %s', len(records), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CSV — UCS-aligned columns
# ---------------------------------------------------------------------------

# UCS-aligned CSV columns
CSV_COLUMNS = [
    'file_name',
    'cat_id',
    'category',
    'subcategory',
    'category_full',
    'fx_name',
    'description',
    'keywords',
    'sound_events',
    'confidence',
    'duration_seconds',
    'creator_id',
    'source_id',
    'user_data',
    'suggested_filename',
    'model_id',
    'vocab_file',
    'vocab_sha256',
    'cache_fingerprint',
]


def save_csv(
    records: List[AudioAnalysisRecord],
    output_path: str | Path,
) -> Path:
    """
    Save a list of records to a UCS-aligned flat CSV file.

    Keywords and sound_events are serialized as semicolon-separated strings.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        rows.append({
            'file_name': r.file_name,
            'cat_id': r.cat_id,
            'category': r.category,
            'subcategory': r.subcategory,
            'category_full': r.category_full,
            'fx_name': r.fx_name,
            'description': r.description,
            'keywords': '; '.join(r.keywords),
            'sound_events': '; '.join(r.sound_events),
            'confidence': r.confidence,
            'duration_seconds': round(r.metadata.duration_seconds, 2),
            'creator_id': r.creator_id,
            'source_id': r.source_id,
            'user_data': r.user_data,
            'suggested_filename': r.suggested_filename,
            'model_id': r.analysis_provenance.model_id,
            'vocab_file': Path(r.analysis_provenance.vocab_path).name,
            'vocab_sha256': r.analysis_provenance.vocab_sha256,
            'cache_fingerprint': r.analysis_provenance.cache_fingerprint or '',
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info('Saved CSV (%d records): %s', len(records), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Markdown (single-record review format)
# ---------------------------------------------------------------------------

def save_markdown(
    record: AudioAnalysisRecord,
    output_dir: str | Path,
) -> Path:
    """
    Save a single analysis record to a human-readable Markdown file.
    Ideal for per-file review.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(record.file_name).stem
    output_path = output_dir / f"{stem}.md"

    md = _record_to_markdown(record)
    output_path.write_text(md, encoding='utf-8')

    logger.debug('Saved Markdown: %s', output_path)
    return output_path


def _record_to_markdown(r: AudioAnalysisRecord) -> str:
    """Format a single AudioAnalysisRecord as UCS-aligned Markdown."""
    keyword_list = '\n'.join(f"- `{k}`" for k in r.keywords)
    event_list = '\n'.join(f"{i+1}. {e}" for i, e in enumerate(r.sound_events))

    top_labels_rows = '\n'.join(
        f"| {label} | {score:.3f} |"
        for label, score in sorted(
            r.top_labels.items(), key=lambda x: x[1], reverse=True
        )[:10]
    )

    lines = [
        f"# {r.file_name}",
        '',
        f"> **{r.fx_name}**",
        '',
        '---',
        '',
        '## UCS Classification',
        '',
        f"| Field | Value |",
        f"|---|---|",
        f"| **CatID** | `{r.cat_id}` |",
        f"| **Category** | {r.category} |",
        f"| **SubCategory** | {r.subcategory} |",
        f"| **CategoryFull** | {r.category_full} |",
        f"| **CreatorID** | {r.creator_id} |",
        f"| **SourceID** | {r.source_id} |",
        f"| **UserData** | {r.user_data or '—'} |",
        f"| **Suggested Filename** | `{r.suggested_filename}` |",
        '',
        '## File Info',
        '',
        f"| Field | Value |",
        f"|---|---|",
        f"| **File** | `{r.file_name}` |",
        f"| **Duration** | {r.metadata.duration_seconds:.2f}s |",
        f"| **Format** | {r.metadata.format.upper()} |",
        f"| **Sample Rate** | {r.metadata.sample_rate_hz} Hz |",
        f"| **Confidence** | {r.confidence:.3f} |",
        '',
        '## Analysis Provenance',
        '',
        f"| Field | Value |",
        f"|---|---|",
        f"| **Model** | `{r.analysis_provenance.model_id}` |",
        f"| **Vocabulary File** | `{Path(r.analysis_provenance.vocab_path).name}` |",
        f"| **Vocabulary SHA256** | `{r.analysis_provenance.vocab_sha256}` |",
        f"| **Cache Fingerprint** | `{r.analysis_provenance.cache_fingerprint or '—'}` |",
        f"| **Cache Path** | `{r.analysis_provenance.cache_path or '—'}` |",
        '',
        '## FXName',
        '',
        f"`{r.fx_name}`",
        '',
        '## Description',
        '',
        r.description,
        '',
        '## Sound Events',
        '',
        event_list if event_list else '_No distinct events detected._',
        '',
        '## Keywords',
        '',
        keyword_list if keyword_list else '_No keywords._',
        '',
        '## CLAP Classification Scores',
        '',
        '| Label | Score |',
        '|---|---|',
        top_labels_rows,
        '',
        '## Acoustic Summary',
        '',
        f"| Property | Value |",
        f"|---|---|",
        f"| RMS Mean | {r.acoustic_summary.rms_mean:.5f} |",
        f"| Spectral Centroid | {r.acoustic_summary.spectral_centroid_mean_hz:.1f} Hz |",
        f"| Spectral Flatness | {r.acoustic_summary.spectral_flatness_mean:.4f} |",
        f"| Percussive | {r.acoustic_summary.is_percussive} |",
        f"| Tonal | {r.acoustic_summary.is_tonal} |",
        f"| Silence Ratio | {r.acoustic_summary.silence_ratio:.2f} |",
        f"| Dynamic Range | {r.acoustic_summary.dynamic_range_db:.1f} dB |",
        f"| Dominant Band | {r.acoustic_summary.dominant_frequency_band} |",
        '',
        '---',
        f"_Analyzed at: {r.analyzed_at}_",
    ]
    return '\n'.join(lines)
