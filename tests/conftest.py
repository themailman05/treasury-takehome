"""Shared pytest fixtures + import-time environment for the suite.

The whole suite runs with the GPU-free, Redis-free backends so it passes on a
laptop and in CI: mock inference, mock OCR, no Redis. We set those env vars and
clear the ``get_settings`` cache *before* any module that reads settings is
imported, so every test sees a consistent configuration.
"""

from __future__ import annotations

import os
from io import BytesIO

import pytest

# Pin the laptop/CI backends before config/Settings are first instantiated.
os.environ["INFERENCE_BACKEND"] = "mock"
os.environ["OCR_BACKEND"] = "mock"
os.environ["REDIS_URL"] = ""

from config import get_settings  # noqa: E402  (after env is set)

# Force a fresh Settings read against the env above.
get_settings.cache_clear()


@pytest.fixture
def settings():
    """The process-wide cached Settings (mock backends, no Redis)."""
    return get_settings()


@pytest.fixture
def tiny_png() -> bytes:
    """A minimal valid PNG. Decodes by PIL but is *not* UTF-8 JSON, so the mock
    inference client takes its echo path (a clean label) rather than the
    fixture path."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()
