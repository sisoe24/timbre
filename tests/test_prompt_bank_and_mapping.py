from __future__ import annotations

from timbre.output.schema import EvidenceEvent, EvidenceBundle, PromptEvidence
from timbre.analysis.analyzer import analyze
from timbre.analysis.prompt_bank import build_descriptive_prompt_bank


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


def test_analyzer_repairs_out_of_taxonomy_response(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_complete_json(**kwargs):
        calls.append((kwargs['system_prompt'], kwargs['user_prompt']))
        if len(calls) == 1:
            return ({
                'description': 'A metallic impact sound.',
                'category': 'OBJECTS',       # invalid — not in taxonomy
                'subcategory': 'METAL',
                'fx_name': 'Wrong',
                'keywords': ['wrong'],
                'sound_events': ['wrong'],
                'alternatives': [],
                'conflict_flags': ['taxonomy_mismatch'],
                'mapper_notes': 'needs repair',
            }, {'backend': 'openai', 'model': 'gpt', 'attempts': 1, 'repaired': False})
        return ({
            'description': 'A short sharp metal impact.',
            'category': 'IMPACTS',
            'subcategory': 'METAL',
            'fx_name': 'Metal Hit',
            'keywords': ['metal', 'impact'],
            'sound_events': ['metal impact'],
            'alternatives': [{'category': 'IMPACTS', 'subcategory': 'METAL', 'reason': 'match'}],
            'conflict_flags': [],
            'mapper_notes': 'repaired',
        }, {'backend': 'openai', 'model': 'gpt', 'attempts': 1, 'repaired': False})

    monkeypatch.setattr('timbre.analysis.analyzer.complete_json', fake_complete_json)

    result, diagnostics, provenance = analyze(
        _evidence(),
        _taxonomy(),
        backend='openai',
        model='gpt',
        temperature=0.1,
        retries=1,
    )

    assert result.category == 'IMPACTS'
    assert result.cat_id == 'IMPMtl'
    assert diagnostics.repair_attempted is True
    assert provenance['attempts'] == 2


def test_analyzer_prompt_includes_evidence(monkeypatch) -> None:
    prompts: list[str] = []

    def fake_complete_json(**kwargs):
        prompts.append(kwargs['user_prompt'])
        return ({
            'description': 'A sharp metallic impact.',
            'category': 'IMPACTS',
            'subcategory': 'METAL',
            'fx_name': 'Metal Impact',
            'keywords': ['metal', 'impact'],
            'sound_events': ['metal impact'],
            'alternatives': [],
            'conflict_flags': [],
            'mapper_notes': 'clean mapping',
        }, {'backend': 'openai', 'model': 'gpt', 'attempts': 1, 'repaired': False})

    monkeypatch.setattr('timbre.analysis.analyzer.complete_json', fake_complete_json)

    result, _, _ = analyze(
        _evidence(),
        _taxonomy(),
        backend='openai',
        model='gpt',
        temperature=0.1,
        retries=0,
    )

    assert result.category == 'IMPACTS'
    assert prompts
    # Evidence and taxonomy are both in the single prompt
    assert 'top_labels' in prompts[0]
    assert 'Taxonomy' in prompts[0]
    assert 'Evidence' in prompts[0]


def test_analyzer_validates_and_resolves_taxonomy(monkeypatch) -> None:
    """LLM-returned cat_id/category_full are ignored; resolved from taxonomy."""

    def fake_complete_json(**kwargs):
        return ({
            'description': 'Metal impact.',
            'category': 'IMPACTS',
            'subcategory': 'METAL',
            'cat_id': 'IGNORED',          # should be overwritten by taxonomy lookup
            'category_full': 'IGNORED',   # same
            'fx_name': 'Metal Hit',
            'keywords': ['metal'],
            'sound_events': ['metal impact'],
            'alternatives': [],
            'conflict_flags': [],
            'mapper_notes': '',
        }, {'backend': 'openai', 'model': 'gpt', 'attempts': 1, 'repaired': False})

    monkeypatch.setattr('timbre.analysis.analyzer.complete_json', fake_complete_json)

    result, _, _ = analyze(
        _evidence(),
        _taxonomy(),
        backend='openai',
        model='gpt',
        temperature=0.1,
        retries=0,
    )

    assert result.cat_id == 'IMPMtl'
    assert result.category_full == 'IMPACTS-METAL'
