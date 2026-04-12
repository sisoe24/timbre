"""
analyzer.py
-----------
Single-call LLM analysis: CLAP evidence → description + UCS metadata.

Replaces the two-stage description_generator + metadata_mapper flow with
one LLM call that produces both the natural language description and the
UCS-compliant classification in a single structured JSON response.
"""

from __future__ import annotations

import json
from typing import Dict, List

from ..llm.client import complete_json
from ..output.schema import (AnalysisResult, EvidenceBundle, RankedAlternative,
                             MappingDiagnostics)

SYSTEM_PROMPT = """\
You analyze audio evidence and produce a complete UCS-compliant catalog record.

Rules:
- Use only the provided evidence. Do not invent sources, materials, or actions absent from it.
- Write a clear, accurate natural language description of the sound.
- Choose exactly one category/subcategory pair from the provided taxonomy.
- Keep fx_name short and catalog-friendly (~25 chars).
- Keywords must be specific, deduplicated, and search-oriented.
- sound_events must be short event labels aligned with the evidence.
- Express uncertainty explicitly in uncertainty_notes.
- Return only valid JSON.
"""


def analyze(
    evidence: EvidenceBundle,
    taxonomy: Dict[str, Dict[str, Dict[str, str]]],
    *,
    backend: str,
    model: str,
    temperature: float,
    retries: int,
) -> tuple[AnalysisResult, MappingDiagnostics, dict]:
    """Single LLM call: evidence → description + UCS metadata."""
    user_prompt = (
        'Analyze this audio clip and produce a UCS catalog record.\n\n'
        f"Taxonomy:\n{json.dumps(taxonomy, indent=2)}\n\n"
        f"Evidence:\n{json.dumps(_evidence_payload(evidence), indent=2)}\n\n"
        'Return JSON with keys: description, category, subcategory, fx_name, '
        'keywords, sound_events, uncertainty_notes, alternatives, '
        'conflict_flags, mapper_notes.'
    )

    def repair_callback(raw: str) -> tuple[str, str]:
        repair_system = SYSTEM_PROMPT + '\nFix invalid or out-of-taxonomy JSON.'
        repair_user = (
            'Repair this analysis response. category and subcategory must be '
            'valid values from the taxonomy.\n\n'
            f"Taxonomy:\n{json.dumps(taxonomy, indent=2)}\n\n"
            f"Evidence:\n{json.dumps(_evidence_payload(evidence), indent=2)}\n\n"
            f"Invalid response:\n{raw}"
        )
        return repair_system, repair_user

    payload, provenance = complete_json(
        backend=backend,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        retries=retries,
        repair_callback=repair_callback,
    )

    repaired_for_taxonomy = False
    try:
        result, diagnostics = _validate_payload(payload, taxonomy)
    except ValueError:
        repaired_for_taxonomy = True
        repair_system, repair_user = repair_callback(json.dumps(payload))
        payload, repaired_provenance = complete_json(
            backend=backend,
            model=model,
            system_prompt=repair_system,
            user_prompt=repair_user,
            temperature=temperature,
            retries=0,
        )
        result, diagnostics = _validate_payload(payload, taxonomy)
        provenance['attempts'] += repaired_provenance.get('attempts', 1)
        provenance['repaired'] = True

    diagnostics.repair_attempted = bool(provenance.get('repaired') or repaired_for_taxonomy)
    diagnostics.repair_succeeded = diagnostics.repair_attempted
    return result, diagnostics, provenance


def _evidence_payload(evidence: EvidenceBundle) -> dict:
    return {
        'top_labels': dict(list(evidence.base_label_scores.items())[:10]),
        'prompt_matches': [
            {
                'prompt': item.prompt,
                'base_label': item.base_label,
                'category': item.category,
                'subcategory': item.subcategory,
                'score': item.score,
                'modifiers': item.modifiers,
            }
            for item in evidence.descriptive_prompt_matches[:8]
        ],
        'sound_events': [
            {
                'label': event.label,
                'category': event.category,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'confidence': event.confidence,
            }
            for event in evidence.sound_events[:8]
        ],
        'acoustic_flags': sorted(dict.fromkeys(evidence.acoustic_flags)),
        'dominant_frequency_band': evidence.dominant_frequency_band,
    }


def _validate_payload(
    payload: dict,
    taxonomy: Dict[str, Dict[str, Dict[str, str]]],
) -> tuple[AnalysisResult, MappingDiagnostics]:
    category = str(payload.get('category', '')).strip()
    subcategory = str(payload.get('subcategory', '')).strip()

    if category not in taxonomy:
        raise ValueError(f'Invalid category: {category!r}')
    if subcategory not in taxonomy[category]:
        raise ValueError(f'Invalid subcategory: {category}/{subcategory!r}')

    resolved = taxonomy[category][subcategory]

    result = AnalysisResult(
        description=str(payload.get('description', '')).strip(),
        category=category,
        subcategory=subcategory,
        cat_id=resolved['cat_id'],
        category_full=resolved['category_full'],
        fx_name=str(payload.get('fx_name', '')).strip()[:50] or subcategory.title(),
        keywords=_clean_list(payload.get('keywords', [])),
        sound_events=_clean_list(payload.get('sound_events', [])),
        uncertainty_notes=_clean_list(payload.get('uncertainty_notes', [])),
        conflict_flags=_clean_list(payload.get('conflict_flags', [])),
        mapper_notes=str(payload.get('mapper_notes', '')).strip(),
    )

    diagnostics = MappingDiagnostics(
        ranked_alternatives=_build_alternatives(payload.get('alternatives', []), taxonomy),
        conflict_flags=result.conflict_flags,
        mapper_notes=result.mapper_notes,
    )

    return result, diagnostics


def _build_alternatives(
    alternatives: List[dict],
    taxonomy: Dict[str, Dict[str, Dict[str, str]]],
) -> List[RankedAlternative]:
    items: list[RankedAlternative] = []
    for raw in alternatives[:5]:
        if not isinstance(raw, dict):
            continue
        category = str(raw.get('category', '')).strip()
        subcategory = str(raw.get('subcategory', '')).strip()
        if category not in taxonomy or subcategory not in taxonomy[category]:
            continue
        resolved = taxonomy[category][subcategory]
        items.append(RankedAlternative(
            category=category,
            subcategory=subcategory,
            cat_id=resolved['cat_id'],
            category_full=resolved['category_full'],
            reason=str(raw.get('reason', '')).strip(),
        ))
    return items


def _clean_list(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for v in values:
        text = str(v).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
