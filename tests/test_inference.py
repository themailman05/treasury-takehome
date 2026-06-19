"""Inference-client tests: tolerant JSON parsing, error wrapping, and backend
dispatch. No network — the real backends are only constructed, never called."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from config import Settings
from inference import (
    InferenceError,
    MockInferenceClient,
    OllamaInferenceClient,
    VLLMInferenceClient,
    _MAX_IMAGE_DIM,
    _encode_for_vlm,
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


# --------------------------------------------------------------------------- #
# _encode_for_vlm: the bytes we hand the vision backend must always be a PNG it
# can decode. The backend (Ollama/llama.cpp) decodes with stb_image, which can't
# read every container Pillow can open — a small WebP/HEIC/GIF/TIFF used to be
# passed through verbatim and 400'd ("failed to decode image bytes"), wrongly
# degrading the label to needs_review / image_quality "unreadable". The sample
# suite never caught this: every fixture is a 1408px PNG (always re-encoded).
# --------------------------------------------------------------------------- #


def _img_bytes(fmt: str, size=(800, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (123, 45, 67)).save(buf, format=fmt)
    return buf.getvalue()


def _png_open(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    assert im.format == "PNG"
    return im


@pytest.mark.parametrize("fmt", ["WEBP", "GIF", "BMP", "TIFF", "JPEG", "PNG"])
def test_encode_for_vlm_always_emits_png(fmt):
    try:
        src = _img_bytes(fmt)
    except Exception:  # pragma: no cover - Pillow missing a codec in this env
        pytest.skip(f"Pillow lacks {fmt} encode support here")
    _png_open(_encode_for_vlm(src))  # decodable PNG regardless of input container


def test_encode_for_vlm_small_nonpng_normalized_same_size():
    # The exact regression: a <= max_dim WebP must come back as PNG, not verbatim.
    out = _encode_for_vlm(_img_bytes("WEBP", size=(900, 491)))
    assert _png_open(out).size == (900, 491)


def test_encode_for_vlm_downscales_oversized_to_png():
    out = _encode_for_vlm(_img_bytes("PNG", size=(4000, 2000)))
    assert max(_png_open(out).size) == _MAX_IMAGE_DIM


def test_encode_for_vlm_non_image_returned_unchanged():
    junk = b"this is definitely not an image"
    assert _encode_for_vlm(junk) == junk
