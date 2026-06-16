"""Inference-client tests: tolerant JSON parsing, error wrapping, and backend
dispatch. No network — the real backends are only constructed, never called."""

from __future__ import annotations

import pytest

from config import Settings
from inference import (
    InferenceError,
    MockInferenceClient,
    OllamaInferenceClient,
    VLLMInferenceClient,
    _extract_json,
    build_inference_client,
)


def test_extract_json_plain_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence():
    fenced = '```json\n{"a": 1}\n```'
    assert _extract_json(fenced) == {"a": 1}


def test_extract_json_recovers_embedded_object():
    # Model added prose around the JSON despite being told not to.
    assert _extract_json('here you go: {"a": 1} done') == {"a": 1}


def test_extract_json_empty_raises():
    with pytest.raises(InferenceError):
        _extract_json("")
    with pytest.raises(InferenceError):
        _extract_json(None)


def test_extract_json_garbage_raises():
    with pytest.raises(InferenceError):
        _extract_json("not json at all")


def test_extract_json_non_object_raises():
    with pytest.raises(InferenceError):
        _extract_json("[1, 2, 3]")


def test_build_inference_client_dispatch():
    assert isinstance(
        build_inference_client(Settings(inference_backend="mock")),
        MockInferenceClient,
    )
    assert isinstance(
        build_inference_client(Settings(inference_backend="ollama")),
        OllamaInferenceClient,
    )
    assert isinstance(
        build_inference_client(Settings(inference_backend="vllm")),
        VLLMInferenceClient,
    )


def test_vllm_structured_output_uses_guided_json():
    payload: dict = {}
    VLLMInferenceClient(Settings(inference_backend="vllm"))._structured_output(payload)
    assert "guided_json" in payload


def test_ollama_targets_native_chat_endpoint():
    # Ollama uses the native /api/chat (think:false) — derived from the ".../v1" base.
    c = OllamaInferenceClient(Settings(inference_backend="ollama",
                                       ollama_base_url="http://ollama:11434/v1"))
    assert c._chat_url == "http://ollama:11434/api/chat"
