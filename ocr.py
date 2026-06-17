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
import re
from io import BytesIO
from typing import Optional

from rapidfuzz import fuzz

from config import Settings, canonical_warning, get_settings


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip().lower()


def _locate(words: list, query: str, W: int, H: int) -> Optional[list[int]]:
    """Find the contiguous OCR word-span that best matches ``query`` and return its
    normalized ``[ymin, xmin, ymax, xmax]`` box (0-1000), or ``None`` if no
    confident match. ``words`` is a list of ``(text, left, top, w, h)``.

    This grounds a field's overlay box in where its text *actually appears* on the
    label (via OCR), instead of trusting the VLM's bounding box — which is wildly
    off on complex layouts.
    """
    q = _norm(query)
    if not q or not words:
        return None
    nq = max(1, len(q.split()))
    best_r, best_span = 0.0, None
    n = len(words)
    for length in range(max(1, nq - 1), nq + 3):
        for i in range(0, n - length + 1):
            span = words[i:i + length]
            joined = _norm(" ".join(w[0] for w in span))
            r = fuzz.ratio(q, joined)
            if r > best_r:
                best_r, best_span = r, span
    if best_r >= 80 and best_span:
        x0 = min(w[1] for w in best_span); y0 = min(w[2] for w in best_span)
        x1 = max(w[1] + w[3] for w in best_span); y1 = max(w[2] + w[4] for w in best_span)
        return [round(y0 / H * 1000), round(x0 / W * 1000),
                round(y1 / H * 1000), round(x1 / W * 1000)]
    return None


class OCRClient(abc.ABC):
    """Reads text from a (cropped) region of a label image."""

    @abc.abstractmethod
    async def read_region(
        self,
        image_bytes: bytes,
        bbox: Optional[list[int]],
        field_texts: Optional[dict[str, str]] = None,
    ) -> tuple[str, Optional[list[int]], dict[str, list[int]]]:
        """Locate + OCR the warning, and OCR-ground the field boxes.

        Returns ``(warning_text, warning_box, field_boxes)``:
          * ``warning_text`` — the recognized warning text (case-preserving);
          * ``warning_box`` — its ``[ymin, xmin, ymax, xmax]`` box normalized
            0-1000, or ``None``;
          * ``field_boxes`` — ``{field_name: box}`` for each ``field_texts`` entry
            whose text could be confidently located in the OCR (others omitted).
        ``bbox`` is a hint, unused.
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
        self,
        image_bytes: bytes,
        bbox: Optional[list[int]] = None,
        field_texts: Optional[dict[str, str]] = None,
    ) -> tuple[str, Optional[list[int]], dict[str, list[int]]]:
        """Return the canonical warning string, no box, no field boxes."""
        return canonical_warning(self._settings.warning_text_version), None, {}

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
        self,
        image_bytes: bytes,
        bbox: Optional[list[int]] = None,
        field_texts: Optional[dict[str, str]] = None,
    ) -> tuple[str, Optional[list[int]], dict[str, list[int]]]:
        """Locate + read the warning, and OCR-ground the field boxes — one pass.

        Upscale the label and run a single word-box pass (``image_to_data``). From
        those words we (1) find the ``GOVERNMENT`` anchor, tight-crop the warning
        block and OCR it (``--psm 6``), returning its box; and (2) locate each
        extracted field's text to box it where it actually appears (the VLM's own
        boxes are unreliable on complex layouts). A field whose text can't be
        confidently located gets no box (better than a wrong one). Falls back to
        the bottom of the label if the warning header can't be anchored.
        """
        import pytesseract
        from PIL import Image

        field_texts = field_texts or {}
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Upscale small/low-res labels so the word-box pass and final read are legible.
        longest = max(image.size)
        if longest < 1800:
            f = 1800 / longest
            image = image.resize((round(image.size[0] * f), round(image.size[1] * f)))
        w, h = image.size

        region = None
        box: Optional[list[int]] = None
        field_boxes: dict[str, list[int]] = {}
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            words = []  # (text, left, top, width, height) for confident words
            for i in range(len(data["text"])):
                t = data["text"][i].strip()
                if not t:
                    continue
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    conf = -1.0
                if conf >= 30:
                    words.append((t, data["left"][i], data["top"][i],
                                  data["width"][i], data["height"][i]))

            # (1) Ground each field's overlay box in where its text actually appears.
            for label, txt in field_texts.items():
                located = _locate(words, txt, w, h)
                if located:
                    field_boxes[label] = located

            # (2) Locate the warning by its GOVERNMENT anchor, box + tight-crop it.
            gov = [g for g in words if g[0].lower().startswith("government")]
            if gov:
                ay = min(g[2] for g in gov)
                ax = min(g[1] for g in gov)
                block = [g for g in words if ay - 5 <= g[2] <= ay + 0.45 * h and g[1] >= ax - 0.05 * w]
                if block:
                    px, py = 0.012 * w, 0.012 * h
                    x0 = max(0, min(g[1] for g in block) - px)
                    y0 = max(0, min(g[2] for g in block) - py)
                    x1 = min(w, max(g[1] + g[3] for g in block) + px)
                    y1 = min(h, max(g[2] + g[4] for g in block) + py)
                    box = [round(y0 / h * 1000), round(x0 / w * 1000),
                           round(y1 / h * 1000), round(x1 / w * 1000)]
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
        return text.strip(), box, field_boxes

    async def aclose(self) -> None:
        """No persistent resources to release."""
        return None


def build_ocr_client(settings: Settings) -> OCRClient:
    """Construct the OCR client for ``settings.ocr_backend`` ("mock" | "tesseract")."""
    if settings.ocr_backend == "tesseract":
        return TesseractOCR(settings)
    return MockOCR(settings)
