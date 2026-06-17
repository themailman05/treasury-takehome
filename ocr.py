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
        """Read the government warning, located by its text anchor.

        The warning can sit anywhere — the bottom of a portrait label, or the
        middle of the back panel on a front+back layout — so a fixed bottom crop
        misses it (it read the wrong region in testing). Instead we **find the
        warning**: upscale the label so the small warning text is legible, locate
        the ``GOVERNMENT`` anchor with a word-box pass (``image_to_data``), and
        crop from there to the bottom of the label, then OCR that block
        (``--psm 6``). Falls back to the bottom of the label if no anchor is found
        (e.g. the header is too degraded to read).

        ``bbox`` is accepted for interface compatibility but not used — the text
        anchor is more reliable than the VLM's (downscaled, often absent) box.
        """
        import pytesseract
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Upscale small/low-res labels so both the anchor pass and the final read
        # have legible text (warning paragraphs are tiny on real captures).
        longest = max(image.size)
        if longest < 1800:
            f = 1800 / longest
            image = image.resize((round(image.size[0] * f), round(image.size[1] * f)))
        w, h = image.size

        region = None
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            tops = [
                data["top"][i]
                for i, t in enumerate(data["text"])
                if t.strip().lower().startswith("government")
            ]
            if tops:
                region = image.crop((0, max(0, min(tops) - 15), w, h))
        except Exception:
            region = None
        if region is None:
            region = image.crop((0, int(h * 0.55), w, h))  # fallback: bottom of the label

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
