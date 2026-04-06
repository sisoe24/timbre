"""
description_generator.py
------------------------
LLM-backed evidence-constrained audio description generation.
"""

from __future__ import annotations

import json
from typing import Iterable

from ..llm.client import complete_json
from ..output.schema import EvidenceBundle, DescriptionDetails

SYSTEM_PROMPT = """\
You generate grounded audio descriptions from structured evidence.

Rules:
- Use only the provided evidence.
- Do not invent unseen sources, materials, actions, or environments.
- Do not assign UCS categories or metadata.
- Keep the description concise and concrete.
- Express uncertainty explicitly when the evidence is mixed.
- Return only valid JSON.
"""


def generate_description(
    evidence: EvidenceBundle,
    *,
    backend: str,
    model: str,
    temperature: float,
    retries: int,
) -> tuple[DescriptionDetails, dict]:
    """Generate an evidence-constrained description."""
    user_prompt = (
        'Generate a grounded description for this clip.\n\n'
        f"{json.dumps(_evidence_payload(evidence), indent=2)}\n\n"
        'Return JSON with keys: description, salient_attributes, '
        'uncertain_attributes, negative_claims, keyword_candidates.'
    )

    payload, provenance = complete_json(
        backend=backend,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        retries=retries,
    )
    return DescriptionDetails.model_validate(payload), provenance


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
        'acoustic_flags': list(_sorted_unique(evidence.acoustic_flags)),
        'dominant_frequency_band': evidence.dominant_frequency_band,
    }


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(item for item in items if item))
