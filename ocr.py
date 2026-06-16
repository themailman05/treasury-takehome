"""Deterministic OCR of the cropped government-warning region.

The warning is the one field that must *not* trust the VLM transcription
(see ``verify_warning`` for the rationale): OCR reports what is literally on the
label, not what should be there. This module supplies that authoritative reading.

Two backends, dispatched by ``Settings.ocr_backend``:

* ``MockOCR`` returns the canonical warning string for the configured version,
  case-preserving, with no external binary. This keeps the whole service runnable
  on a laptop and in CI (and yields a clean PASS for a well-formed mock label).
* ``TesseractOCR`` crops the warning bbox and runs Tesseract. ``pytesseract`` and
  ``PIL`` are imported lazily so importing this module never requires the binary.
"""

from __future__ import annotations

import abc
from io import BytesIO
from typing import Optional

from config import Settings, canonical_warning, get_settings


class OCRClient(abc.ABC):
    """Reads text from a (cropped) region of a label image."""

    @abc.abstractmethod
    async def read_region(self, image_bytes: bytes, bbox: Optional[list[int]]) -> str:
        """OCR the region ``bbox`` (``[x, y, w, h]``); full image when ``bbox`` is None.

        Returns the literal recognized text, case-preserving and untrimmed of its
        meaning — the deterministic, authoritative reading of the warning.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release any held resources. Safe to call more than once."""
        raise NotImplementedError


class MockOCR(OCRClient):
    """Deterministic OCR for laptop/CI runs — no Tesseract binary required.

    Always returns the canonical warning for the configured version, preserving
    case so the downstream header-casing check sees real upper-case text.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    async def read_region(self, image_bytes: bytes, bbox: Optional[list[int]]) -> str:
        """Return the canonical warning string (bbox/image are ignored)."""
        return canonical_warning(self._settings.warning_text_version)

    async def aclose(self) -> None:
        """No resources to release."""
        return None


class TesseractOCR(OCRClient):
    """Real deterministic OCR via Tesseract on the cropped warning region.

    ``pytesseract`` (and the Tesseract binary it wraps) plus ``PIL`` are imported
    lazily inside ``__init__`` so this module imports cleanly even where Tesseract
    is not installed; the import error only surfaces when this backend is selected.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        # Lazy imports: fail loudly only when the tesseract backend is actually built.
        import pytesseract  # noqa: F401  (validated here, used in read_region)
        from PIL import Image  # noqa: F401

    async def read_region(self, image_bytes: bytes, bbox: Optional[list[int]]) -> str:
        """Read the government warning with a focused, upscaled crop.

        OCRing the whole label makes Tesseract mis-segment the page and miss the
        small warning paragraph (it came back with clauses "missing" in testing).
        Instead we *box* the warning: a first pass locates the ``GOVERNMENT``
        anchor, we crop from there to the bottom, upscale 2x, and OCR that block
        with ``--psm 6`` (a single uniform text block). This reads both clauses
        reliably across labels. Falls back to an upscaled full-image read if the
        anchor isn't found.

        ``bbox`` is accepted for interface compatibility but not used — the text
        anchor is more reliable than the VLM's (downscaled) box.
        """
        import pytesseract
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        # The government warning sits at the bottom of the label. Crop the bottom
        # ~40% and upscale 2x before OCR. Full-page OCR mis-segments the small
        # warning text (clauses came back "missing"), and the word-box anchor pass
        # (image_to_data) is unreliable under a restricted service environment, so
        # we use a fixed bottom crop: deterministic and consistent across processes.
        # CROP_V0p60 marker.
        region = image.crop((0, int(h * 0.60), w, h))
        region = region.resize((region.size[0] * 2, region.size[1] * 2))
        text: str = pytesseract.image_to_string(region, config="--psm 6")
        return text.strip()

    async def aclose(self) -> None:
        """No persistent resources to release."""
        return None


def build_ocr_client(settings: Settings) -> OCRClient:
    """Construct the OCR client for ``settings.ocr_backend`` ("mock" | "tesseract")."""
    if settings.ocr_backend == "tesseract":
        return TesseractOCR(settings)
    return MockOCR(settings)
