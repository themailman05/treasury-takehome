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
    async def read_region(
        self, image_bytes: bytes, bbox: Optional[list[int]]
    ) -> tuple[str, Optional[list[int]]]:
        """Locate and OCR the government warning.

        Returns ``(text, box)``: the literal recognized warning text
        (case-preserving), and the warning's bounding box as
        ``[ymin, xmin, ymax, xmax]`` normalized 0-1000 (for the audit overlay),
        or ``None`` if it couldn't be localized. ``bbox`` is a hint, unused.
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

    async def read_region(
        self, image_bytes: bytes, bbox: Optional[list[int]]
    ) -> tuple[str, Optional[list[int]]]:
        """Return the canonical warning string and no box (bbox/image ignored)."""
        return canonical_warning(self._settings.warning_text_version), None

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

    async def read_region(
        self, image_bytes: bytes, bbox: Optional[list[int]]
    ) -> tuple[str, Optional[list[int]]]:
        """Locate and read the government warning by its text anchor.

        The warning can sit anywhere — bottom of a portrait label, or mid-panel on
        a front+back layout — so a fixed bottom crop misses it. Instead we **find**
        it: upscale the label, locate the ``GOVERNMENT`` anchor with a word-box pass
        (``image_to_data``), crop from there to the bottom, and OCR that block
        (``--psm 6``). We also return the warning's bounding box (the union of the
        anchored words, in the same column) so the audit overlay points at the real
        warning rather than a guessed region. Falls back to the bottom of the label
        (and no box) if the header is too degraded to anchor.
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
        box: Optional[list[int]] = None
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            gov = [i for i in range(n) if data["text"][i].strip().lower().startswith("government")]
            if gov:
                ai = min(gov, key=lambda i: data["top"][i])
                ay, ax = data["top"][ai], data["left"][ai]
                # Warning box = the anchor word + the words below it in the same
                # column (excludes other panels/columns), so it tracks the real
                # warning region across layouts.
                x0s, y0s, x1s, y1s = [], [], [], []
                for i in range(n):
                    if not data["text"][i].strip():
                        continue
                    try:
                        conf = float(data["conf"][i])
                    except (TypeError, ValueError):
                        conf = -1.0
                    top, left = data["top"][i], data["left"][i]
                    if conf >= 30 and ay - 5 <= top <= ay + 0.45 * h and left >= ax - 0.05 * w:
                        x0s.append(left); y0s.append(top)
                        x1s.append(left + data["width"][i]); y1s.append(top + data["height"][i])
                if x0s:
                    px, py = 0.012 * w, 0.012 * h
                    x0 = max(0, min(x0s) - px); y0 = max(0, min(y0s) - py)
                    x1 = min(w, max(x1s) + px); y1 = min(h, max(y1s) + py)
                    box = [round(y0 / h * 1000), round(x0 / w * 1000),
                           round(y1 / h * 1000), round(x1 / w * 1000)]
                    # Crop TIGHTLY to the warning block (not the full width below the
                    # anchor) and upscale it, so Tesseract reads big, clean text —
                    # this is what lifts noisy back-panel clauses out of the
                    # ambiguous band.
                    region = image.crop((round(x0), round(y0), round(x1), round(y1)))
                    if region.size[0] and max(region.size) < 1500:
                        z = 1500 / max(region.size)
                        region = region.resize((round(region.size[0] * z), round(region.size[1] * z)))
                else:
                    region = image.crop((0, max(0, ay - 15), w, h))
        except Exception:
            region = None
        if region is None:
            region = image.crop((0, int(h * 0.55), w, h))  # fallback: bottom of the label

        text: str = pytesseract.image_to_string(region, config="--psm 6")
        return text.strip(), box

    async def aclose(self) -> None:
        """No persistent resources to release."""
        return None


def build_ocr_client(settings: Settings) -> OCRClient:
    """Construct the OCR client for ``settings.ocr_backend`` ("mock" | "tesseract")."""
    if settings.ocr_backend == "tesseract":
        return TesseractOCR(settings)
    return MockOCR(settings)
