"""
metadata_mapper.py
------------------
LLM-backed constrained mapping from structured evidence into UCS metadata.
"""

from __future__ import annotations

import json
from typing import Dict, List

from ..llm.client import complete_json
from ..output.schema import (EvidenceBundle, RankedAlternative,
                             DescriptionDetails, MappingDiagnostics)

SYSTEM_PROMPT = """\
You map structured audio evidence into UCS metadata.

Rules:
- Use the evidence as the source of truth.
- The natural-language description is supporting context only.
- Return only valid JSON.
- Choose one best category/subcategory pair from the provided taxonomy.
- Keep fx_name short and catalog-friendly.
- Keywords should be specific, deduplicated, and search-oriented.
"""


def map_metadata(
    evidence: EvidenceBundle,
    description: DescriptionDetails,
    *,
    taxonomy: Dict[str, Dict[str, Dict[str, str]]],
    backend: str,
    model: str,
    temperature: float,
    retries: int,
) -> tuple[dict, MappingDiagnostics, dict]:
    """Map evidence + description into validated UCS metadata."""
    user_prompt = (
        'Map this clip into UCS metadata using the provided taxonomy.\n\n'
        f"Taxonomy:\n{json.dumps(taxonomy, indent=2)}\n\n"
        f"Evidence:\n{json.dumps(_evidence_payload(evidence, description), indent=2)}\n\n"
        'Return JSON with keys: category, subcategory, cat_id, category_full, '
        'fx_name, keywords, sound_events, alternatives, conflict_flags, mapper_notes.'
    )

    def repair_callback(raw: str) -> tuple[str, str]:
        repair_system = SYSTEM_PROMPT + '\nFix invalid or out-of-taxonomy JSON.'
        repair_user = (
            'Repair this mapping response. It must use only values from the taxonomy.\n\n'
            f"Taxonomy:\n{json.dumps(taxonomy, indent=2)}\n\n"
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
        validated_payload, diagnostics = _validate_mapping_payload(payload, taxonomy)
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
        validated_payload, diagnostics = _validate_mapping_payload(payload, taxonomy)
        provenance['attempts'] += repaired_provenance.get('attempts', 1)
        provenance['repaired'] = True

    diagnostics.repair_attempted = bool(provenance.get('repaired') or repaired_for_taxonomy)
    diagnostics.repair_succeeded = diagnostics.repair_attempted
    return validated_payload, diagnostics, provenance


def _evidence_payload(evidence: EvidenceBundle, description: DescriptionDetails) -> dict:
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
                'confidence': event.confidence,
            }
            for event in evidence.sound_events[:8]
        ],
        'description': description.model_dump(),
    }


def _validate_mapping_payload(
    payload: dict,
    taxonomy: Dict[str, Dict[str, Dict[str, str]]],
) -> tuple[dict, MappingDiagnostics]:
    category = str(payload.get('category', '')).strip()
    subcategory = str(payload.get('subcategory', '')).strip()

    if category not in taxonomy:
        raise ValueError(f'Invalid category from metadata mapper: {category}')
    if subcategory not in taxonomy[category]:
        raise ValueError(
            f'Invalid subcategory from metadata mapper: {category}/{subcategory}'
        )

    resolved = taxonomy[category][subcategory]
    cleaned = {
        'category': category,
        'subcategory': subcategory,
        'cat_id': resolved['cat_id'],
        'category_full': resolved['category_full'],
        'fx_name': str(payload.get('fx_name', '')).strip()[:50] or subcategory.title(),
        'keywords': _clean_list(payload.get('keywords', [])),
        'sound_events': _clean_list(payload.get('sound_events', [])),
    }

    diagnostics = MappingDiagnostics(
        ranked_alternatives=_build_alternatives(payload.get('alternatives', []), taxonomy),
        conflict_flags=_clean_list(payload.get('conflict_flags', [])),
        mapper_notes=str(payload.get('mapper_notes', '')).strip(),
    )
    return cleaned, diagnostics


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
        items.append(
            RankedAlternative(
                category=category,
                subcategory=subcategory,
                cat_id=resolved['cat_id'],
                category_full=resolved['category_full'],
                reason=str(raw.get('reason', '')).strip(),
            )
        )
    return items


def _clean_list(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
