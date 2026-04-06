"""
prompt_bank.py
--------------
Builds a descriptive prompt bank on top of the UCS label set.

The prompts are still controlled text candidates for CLAP scoring; they are
not generated at runtime by an LLM.
"""

from __future__ import annotations

from typing import Dict, List
from dataclasses import dataclass

PROMPT_BANK_VERSION = 'v1'

MODIFIER_GROUPS = {
    'temporal': ('single', 'repeated', 'sustained'),
    'intensity': ('soft', 'hard', 'heavy'),
    'space': ('dry', 'reverberant', 'close', 'distant'),
    'texture': ('sharp', 'boomy', 'tonal', 'noisy'),
}


@dataclass(frozen=True)
class PromptBankEntry:
    prompt: str
    base_label: str
    modifiers: List[str]
    category: str
    subcategory: str
    cat_id: str
    category_full: str


def build_descriptive_prompt_bank(
    candidate_labels: List[str],
    label_to_category: Dict[str, str],
    label_to_subcategory: Dict[str, str],
    label_to_cat_id: Dict[str, str],
    label_to_category_full: Dict[str, str],
) -> List[PromptBankEntry]:
    """Generate a compact descriptive prompt bank from the UCS label set."""
    prompts: list[PromptBankEntry] = []

    for label in candidate_labels:
        category = label_to_category.get(label, 'UNKNOWN')
        subcategory = label_to_subcategory.get(label, 'UNKNOWN')
        cat_id = label_to_cat_id.get(label, 'UNKNOWN')
        category_full = label_to_category_full.get(label, f'{category}-{subcategory}')

        prompts.append(
            PromptBankEntry(
                prompt=label,
                base_label=label,
                modifiers=[],
                category=category,
                subcategory=subcategory,
                cat_id=cat_id,
                category_full=category_full,
            )
        )

        for values in MODIFIER_GROUPS.values():
            for modifier in values:
                prompts.append(
                    PromptBankEntry(
                        prompt=f'{modifier} {label}',
                        base_label=label,
                        modifiers=[modifier],
                        category=category,
                        subcategory=subcategory,
                        cat_id=cat_id,
                        category_full=category_full,
                    )
                )

    return prompts
