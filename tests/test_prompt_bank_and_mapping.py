from __future__ import annotations

from timbre.output.schema import (EvidenceEvent, EvidenceBundle,
                                  PromptEvidence, DescriptionDetails)
from timbre.analysis.prompt_bank import build_descriptive_prompt_bank
from timbre.analysis.metadata_mapper import map_metadata


def _taxonomy() -> dict:
    return {
        'IMPACTS': {
            'METAL': {
                'cat_id': 'IMPMtl',
                'category_full': 'IMPACTS-METAL',
            }
        }
    }


def _evidence() -> EvidenceBundle:
    return EvidenceBundle(
        base_label_scores={'metal impact': 0.72, 'clang': 0.18},
        descriptive_prompt_matches=[
            PromptEvidence(
                prompt='sharp metal impact',
                base_label='metal impact',
                category='IMPACTS',
                subcategory='METAL',
                cat_id='IMPMtl',
                category_full='IMPACTS-METAL',
                modifiers=['sharp'],
                score=0.44,
            )
        ],
        sound_events=[
            EvidenceEvent(
                label='metal impact',
                category='IMPACTS',
                start_time=0.0,
                end_time=0.5,
                confidence=0.81,
            )
        ],
        acoustic_flags=['percussive'],
        dominant_frequency_band='mid',
    )


def test_prompt_bank_generation_is_deterministic() -> None:
    kwargs = {
        'candidate_labels': ['metal impact'],
        'label_to_category': {'metal impact': 'IMPACTS'},
        'label_to_subcategory': {'metal impact': 'METAL'},
        'label_to_cat_id': {'metal impact': 'IMPMtl'},
        'label_to_category_full': {'metal impact': 'IMPACTS-METAL'},
    }
    first = build_descriptive_prompt_bank(**kwargs)
    second = build_descriptive_prompt_bank(**kwargs)

    assert [entry.prompt for entry in first] == [entry.prompt for entry in second]
    assert len(first) == 15
    assert first[0].prompt == 'metal impact'
    assert 'reverberant metal impact' in [entry.prompt for entry in first]


def test_metadata_mapper_repairs_out_of_taxonomy_response(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_complete_json(**kwargs):
        calls.append((kwargs['system_prompt'], kwargs['user_prompt']))
        if len(calls) == 1:
            return ({
                'category': 'OBJECTS',
                'subcategory': 'METAL',
                'cat_id': 'OBJMtl',
                'category_full': 'OBJECTS-METAL',
                'fx_name': 'Wrong',
                'keywords': ['wrong'],
                'sound_events': ['wrong'],
                'alternatives': [],
                'conflict_flags': ['taxonomy_mismatch'],
                'mapper_notes': 'needs repair',
            }, {'backend': 'openai', 'model': 'gpt', 'attempts': 1, 'repaired': False})
        return ({
            'category': 'IMPACTS',
            'subcategory': 'METAL',
            'cat_id': 'bad',
            'category_full': 'bad',
            'fx_name': 'Metal Hit',
            'keywords': ['metal', 'impact'],
            'sound_events': ['metal impact'],
            'alternatives': [{'category': 'IMPACTS', 'subcategory': 'METAL', 'reason': 'match'}],
            'conflict_flags': [],
            'mapper_notes': 'repaired',
        }, {'backend': 'openai', 'model': 'gpt', 'attempts': 1, 'repaired': False})

    monkeypatch.setattr('timbre.analysis.metadata_mapper.complete_json', fake_complete_json)

    mapped, diagnostics, provenance = map_metadata(
        _evidence(),
        DescriptionDetails(
            description='A sharp metallic hit.',
            salient_attributes=['sharp', 'metallic'],
            uncertain_attributes=[],
            negative_claims=[],
            keyword_candidates=['metal', 'impact'],
        ),
        taxonomy=_taxonomy(),
        backend='openai',
        model='gpt',
        temperature=0.1,
        retries=1,
    )

    assert mapped['category'] == 'IMPACTS'
    assert mapped['cat_id'] == 'IMPMtl'
    assert diagnostics.repair_attempted is True
    assert provenance['attempts'] == 2
