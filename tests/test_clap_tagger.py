from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from timbre.models.clap_tagger import CLAPTagger


@pytest.mark.parametrize('failure_stage', ['processor', 'model'])
def test_model_loading_failure_never_substitutes_another_model(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    """A failed requested model cannot silently invalidate cache identity."""
    calls = []

    def fail_load(model_id: str) -> None:
        calls.append(model_id)
        raise OSError('model unavailable')

    transformers = ModuleType('transformers')
    transformers.ClapProcessor = SimpleNamespace(
        from_pretrained=fail_load if failure_stage == 'processor' else lambda model_id: object(),
    )
    transformers.ClapModel = SimpleNamespace(from_pretrained=fail_load)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    tagger = CLAPTagger(model_id='requested/model', device='cpu')

    with pytest.raises(OSError, match='model unavailable'):
        tagger.load()

    assert calls == ['requested/model']
    assert not tagger.is_loaded
