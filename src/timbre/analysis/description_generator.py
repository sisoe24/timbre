"""
description_generator.py
------------------------
LLM-backed generation of a normalized semantic description object.
"""

from __future__ import annotations

import json
from typing import Iterable

from ..llm.client import complete_json
from ..output.schema import EvidenceBundle, DescriptionDetails

SYSTEM_PROMPT = """\
You convert raw audio evidence into a normalized semantic description object.

Rules:
- Use only the provided evidence.
- Do not invent unseen sources, materials, actions, or environments.
- Do not assign UCS categories or metadata.
- Prefer compact canonical terms over prose.
- Express uncertainty explicitly in uncertainty_notes.
- normalized_events must be short event labels suitable for final output.
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
    """Generate a normalized semantic description object."""
    user_prompt = (
        'Generate a normalized audio description object for this clip.\n\n'
        f"{json.dumps(_evidence_payload(evidence), indent=2)}\n\n"
        'Return JSON with keys: primary_action, secondary_actions, primary_source, '
        'secondary_sources, texture_traits, temporal_traits, environment_traits, '
        'uncertainty_notes, negative_claims, keyword_candidates, normalized_events.'
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


def render_description(description: DescriptionDetails) -> str:
    """Render a concise human-readable description from the normalized object."""
    clauses: list[str] = []

    action_bits = [description.primary_action] if description.primary_action else []
    action_bits.extend(description.secondary_actions[:2])
    source_bits = [description.primary_source] if description.primary_source else []
    source_bits.extend(description.secondary_sources[:2])

    if action_bits or source_bits:
        lead = ' '.join(bit for bit in [', '.join(action_bits),
                        _with_prefix(source_bits, 'from')] if bit)
        if lead:
            clauses.append(_sentence_case(lead))

    if description.texture_traits:
        clauses.append(f"Texture is {', '.join(description.texture_traits[:3])}")
    if description.temporal_traits:
        clauses.append(f"Timing feels {', '.join(description.temporal_traits[:3])}")
    if description.environment_traits:
        clauses.append(f"Environment suggests {', '.join(description.environment_traits[:3])}")
    if description.uncertainty_notes:
        clauses.append(f"Uncertain elements: {', '.join(description.uncertainty_notes[:2])}")

    rendered = '. '.join(clause.rstrip('. ') for clause in clauses if clause).strip()
    if not rendered:
        return 'Ambiguous audio event with limited descriptive evidence.'
    return rendered + '.'


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


def _with_prefix(values: list[str], prefix: str) -> str:
    filtered = [value for value in values if value]
    if not filtered:
        return ''
    return f"{prefix} {', '.join(filtered)}"


def _sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(item for item in items if item))
