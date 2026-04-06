"""
client.py
---------
Shared JSON-oriented LLM client used by generation and validation flows.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Callable


class LLMError(RuntimeError):
    """Base LLM runtime error."""


class LLMUnavailableError(LLMError):
    """Raised when a provider cannot be used."""


class LLMJSONError(LLMError):
    """Raised when the provider repeatedly fails to return valid JSON."""


def complete_json(
    *,
    backend: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    retries: int = 1,
    repair_callback: Callable[[str], Tuple[str, str]] | None = None,
) -> tuple[dict, dict]:
    """Run a JSON-only completion with one optional repair attempt."""
    attempts = 0
    repaired = False
    current_system = system_prompt
    current_user = user_prompt
    last_raw = ''

    while True:
        attempts += 1
        raw = _dispatch_backend_call(
            backend=backend,
            model=model,
            system_prompt=current_system,
            user_prompt=current_user,
            temperature=temperature,
        )
        last_raw = raw
        try:
            parsed = _parse_json(raw)
            return parsed, {
                'backend': backend,
                'model': model,
                'attempts': attempts,
                'repaired': repaired,
            }
        except json.JSONDecodeError as exc:
            if retries <= 0:
                raise LLMJSONError(f'Invalid JSON from {backend}/{model}: {exc}') from exc
            retries -= 1
            repaired = True
            if repair_callback is not None:
                current_system, current_user = repair_callback(last_raw)
            else:
                current_system = (
                    'Return only valid JSON. Do not use markdown fences or commentary.'
                )
                current_user = (
                    'Repair this invalid JSON response and return only valid JSON:\n\n'
                    f'{last_raw}'
                )


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith('```'):
        lines = raw.split('\n')
        if len(lines) >= 3:
            raw = '\n'.join(lines[1:-1])
    return json.loads(raw)


def _dispatch_backend_call(
    *,
    backend: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    backend = backend.lower()
    if backend == 'openai':
        return _call_openai(model, system_prompt, user_prompt, temperature)
    if backend == 'anthropic':
        return _call_anthropic(model, system_prompt, user_prompt, temperature)
    if backend == 'ollama':
        return _call_ollama(model, system_prompt, user_prompt, temperature)
    raise LLMUnavailableError(f'Unsupported backend: {backend}')


def _call_openai(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailableError('openai package is not installed.') from exc

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise LLMUnavailableError('OPENAI_API_KEY environment variable not set.')

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=temperature,
        response_format={'type': 'json_object'},
    )
    return response.choices[0].message.content or '{}'


def _call_anthropic(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailableError('anthropic package is not installed.') from exc

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise LLMUnavailableError('ANTHROPIC_API_KEY environment variable not set.')

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=temperature,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    return response.content[0].text


def _call_ollama(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise LLMUnavailableError('ollama package is not installed.') from exc

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        options={'temperature': temperature},
        format='json',
    )
    return response['message']['content']
