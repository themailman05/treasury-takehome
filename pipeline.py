"""The verification pipeline — where extraction meets judgement.

This is the heart of the service: it wires the three independent pieces (the VLM
inference client, the deterministic OCR client, and the pure-Python verifiers)
into one ``extract -> OCR-fill -> verify -> aggregate`` flow that turns an image
plus an application into a single ``VerificationResult``.

The design rule the whole tool rests on (README §5) lives here in the ordering:
**the model only extracts; deterministic code judges.** Concretely:

  1. The VLM returns verbatim text (``ModelExtraction``) — no verdicts.
  2. If the model located the warning but produced no deterministic OCR reading,
     we fill it here from the authoritative OCR client (the warning must never be
     judged on the VLM transcription alone — see ``verify_warning``).
  3. ``verify_fields`` and ``verify_warning`` decide pass / fail / needs_review.
  4. The overall verdict is the *worst* severity across every field and the
     warning, escalated to at least ``needs_review`` when the image isn't clean.

An ``UNREADABLE`` image short-circuits to ``needs_review`` (a bad photo must
never *fail* a label, and never fail a whole batch — README §9), but we still run
the field and warning checks best-effort so the agent sees whatever could be read.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from typing import Optional

from config import Settings, get_settings
from enums import ImageQuality, Verdict
from inference import InferenceClient, InferenceError
from ocr import OCRClient
from schemas import (
    ApplicationData,
    FieldVerdict,
    LabelRegion,
    ModelExtraction,
    VerificationResult,
    WarningVerdict,
)
from verify_fields import verify_fields
from verify_warning import WarningResult, verify_warning

logger = logging.getLogger("label_verification.pipeline")

# Severity ranking for the overall roll-up: a single FAIL dominates, a
# NEEDS_REVIEW beats a clean PASS, and the worst wins.
_SEVERITY: dict[Verdict, int] = {
    Verdict.PASS: 0,
    Verdict.NEEDS_REVIEW: 1,
    Verdict.FAIL: 2,
}


def _unreadable_result(settings: Settings, start: float, flag: str) -> VerificationResult:
    """A graceful ``needs_review`` result when the model can't be read/parsed.

    A hard real-world photo (or a truncated/invalid model response) must never
    surface as a 500/502 — per the brief, a bad image resolves to
    ``needs_review: unreadable`` so one bad photo never fails the request (or a
    whole batch). No fields/regions; the warning is left for a human."""
    cv = settings.warning_text_version
    return VerificationResult(
        fields={},
        warning=WarningVerdict(
            verdict=Verdict.NEEDS_REVIEW, exact_match=False, caps_ok=None,
            readings_agree=None, agreement_ratio=None, canonical_version=cv,
            components=[], review_flags=[flag],
        ),
        overall=Verdict.NEEDS_REVIEW,
        image_quality=ImageQuality.UNREADABLE,
        review_flags=[flag, "unreadable"],
        canonical_version=cv,
        timing_ms=round((time.perf_counter() - start) * 1000.0, 2),
        regions=[],
    )


def _coerce_quality(value: object) -> ImageQuality:
    """Normalize a ``ModelExtraction.image_quality`` to an ``ImageQuality`` member.

    ``ModelExtraction`` uses ``use_enum_values=True``, so the field may arrive as
    either an enum member (the default) or a bare string (when set from JSON).
    """
    if isinstance(value, ImageQuality):
        return value
    try:
        return ImageQuality(value)
    except ValueError:  # pragma: no cover - schema constrains the input
        return ImageQuality.OK


def _worst(verdicts: list[Verdict]) -> Verdict:
    """Return the highest-severity verdict (FAIL > NEEDS_REVIEW > PASS)."""
    if not verdicts:
        return Verdict.PASS
    return max(verdicts, key=lambda v: _SEVERITY[v])


def _build_warning_verdict(wr: WarningResult) -> WarningVerdict:
    """Map the standalone ``WarningResult`` dataclass onto the client schema."""
    return WarningVerdict(
        verdict=wr.verdict,
        exact_match=wr.exact_match,
        caps_ok=wr.caps_ok,
        readings_agree=wr.readings_agree,
        agreement_ratio=wr.agreement_ratio,
        canonical_version=wr.canonical_version,
        components=[asdict(c) for c in wr.components],
        review_flags=list(wr.review_flags),
    )


def _field_flags(fields: dict[str, FieldVerdict]) -> list[str]:
    """Surface per-field review reasons (``low_confidence`` / ``not_extracted``)
    as namespaced flags on the overall result, in field order."""
    flags: list[str] = []
    for name, fv in fields.items():
        if fv.note in ("low_confidence", "not_extracted"):
            flags.append(f"{name}:{fv.note}")
    return flags


def _warning_span(full_ocr_text: str) -> str:
    """Narrow a full-page OCR reading to the government-warning region.

    The OCR client reads the whole label (the model's bbox is unreliable), but
    ``verify_warning`` compares the OCR reading against the VLM's warning-only
    transcription — so we slice from the ``GOVERNMENT WARNING`` header to the end
    (the warning sits at the bottom of a label). Case is preserved for the header
    caps check. If the header can't be located, return the full text unchanged
    (``verify_warning`` then routes the unreadable warning to review)."""
    m = re.search(r"government\s+warning", full_ocr_text, flags=re.IGNORECASE)
    return full_ocr_text[m.start():] if m else full_ocr_text


def _build_regions(
    ext: ModelExtraction,
    fields: dict[str, FieldVerdict],
    warning_verdict: WarningVerdict,
) -> list[LabelRegion]:
    """Assemble the audit overlay: one box per located region with its verdict.

    Field boxes carry their comparison verdict (or None when the applicant didn't
    submit that field); the warning box carries the warning verdict. Boxes are the
    model's normalized [ymin, xmin, ymax, xmax] (0-1000)."""
    regions: list[LabelRegion] = []
    for name in ("brand_name", "class_type", "abv", "net_contents"):
        reading = getattr(ext, name)
        if reading.box:
            fv = fields.get(name)
            regions.append(LabelRegion(
                label=name, box=reading.box,
                verdict=fv.verdict if fv else None, text=reading.text,
            ))
    # The warning is mandatory and the headline compliance check, so always show
    # it. Fall back to the bottom band (where the warning sits, and where OCR reads
    # it) when the model didn't return a box.
    warning_box = ext.warning.box or [600, 20, 990, 980]
    regions.append(LabelRegion(
        label="government_warning", box=warning_box,
        verdict=warning_verdict.verdict, text=ext.warning.vlm_text or "",
    ))
    return regions


def _dedupe(flags: list[str]) -> list[str]:
    """Stable de-duplication preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


async def verify_label(
    image_bytes: bytes,
    application: ApplicationData,
    *,
    inference_client: InferenceClient,
    ocr_client: OCRClient,
    settings: Optional[Settings] = None,
) -> VerificationResult:
    """Run one label end-to-end: extract, OCR-fill the warning, verify, aggregate.

    Parameters
    ----------
    image_bytes : raw label artwork (the inference and OCR clients decode it).
    application : the applicant's expected field values; only submitted fields
        are judged.
    inference_client : VLM client returning verbatim text (never a verdict).
    ocr_client : authoritative deterministic OCR of the cropped warning region.

    Returns the full ``VerificationResult`` the API serves and the UI renders.
    An ``UNREADABLE`` image is reported as ``needs_review`` (never ``fail``) with
    an ``unreadable`` flag, while still verifying whatever text could be read.
    """
    settings = settings or get_settings()
    start = time.perf_counter()

    # 1. Extract — the model reads text only; it makes no judgements. If the
    #    backend errors or returns unparseable/truncated output (a hard real-world
    #    photo can make the model run away past the token cap), degrade to
    #    needs_review:unreadable rather than erroring the request (README §9).
    try:
        ext: ModelExtraction = await inference_client.extract(
            image_bytes, application=application
        )
    except InferenceError as exc:
        logger.warning("extraction failed, returning needs_review: %s", exc)
        return _unreadable_result(settings, start, "extraction_failed")
    quality = _coerce_quality(ext.image_quality)

    # 2. OCR-fill rule: the warning is the one field that must not be judged on the
    #    VLM transcription. If the model located the region but supplied no
    #    deterministic reading, read it now from the authoritative OCR client.
    if not (ext.warning.ocr_text or "").strip():
        # The warning is mandatory, so always read it deterministically. In one OCR
        # pass we read + box the warning AND OCR-ground each field's overlay box
        # (the VLM's own boxes are unreliable on complex layouts).
        field_texts = {
            name: getattr(ext, name).text
            for name in ("brand_name", "class_type", "abv", "net_contents")
            if getattr(ext, name).text.strip()
        }
        full_ocr, ocr_box, field_boxes = await ocr_client.read_region(
            image_bytes, ext.warning.bbox, field_texts
        )
        ext.warning.ocr_text = _warning_span(full_ocr)
        # Prefer the OCR-located box for the overlay — grounded in the actual text,
        # unlike the model's (often absent/imprecise) box.
        if ocr_box:
            ext.warning.box = ocr_box
        # Replace the VLM field boxes with OCR-grounded ones; a field we can't
        # confidently locate gets no box (omitted) rather than a wrong one.
        for name in ("brand_name", "class_type", "abv", "net_contents"):
            getattr(ext, name).box = field_boxes.get(name)

    # 3. Judge — deterministic field matching + the strict dual-path warning check.
    #    These run even for a POOR/UNREADABLE image so the agent sees a best-effort
    #    reading rather than an empty result.
    fields: dict[str, FieldVerdict] = verify_fields(
        ext, application, settings=settings
    )
    # Pass None (not "") for an absent VLM reading so verify_warning takes its
    # conservative single-path branch instead of comparing against empty text.
    vlm_text = ext.warning.vlm_text or None
    wr: WarningResult = verify_warning(ext.warning.ocr_text or "", vlm_text)
    warning_verdict = _build_warning_verdict(wr)

    # 4. Aggregate — overall is the worst severity across all fields + the warning.
    component_verdicts = [fv.verdict for fv in fields.values()]
    component_verdicts.append(warning_verdict.verdict)
    overall = _worst(component_verdicts)

    review_flags: list[str] = list(warning_verdict.review_flags)
    review_flags.extend(_field_flags(fields))
    # A warning that couldn't be located routes to needs_review inside
    # verify_warning; surface the reason explicitly for the agent.
    if not ext.warning.located:
        review_flags.append("warning:not_located")

    # Image-quality policy (README §9): a photo we cannot read must never *fail*
    # (nor pass) a label — UNREADABLE always clamps to needs_review. A POOR but
    # readable photo can't auto-pass, but a clear defect on it may still fail.
    if quality is ImageQuality.UNREADABLE:
        review_flags.append(f"image_quality:{quality.value}")
        review_flags.append("unreadable")
        overall = Verdict.NEEDS_REVIEW
    elif quality is ImageQuality.POOR:
        review_flags.append(f"image_quality:{quality.value}")
        if _SEVERITY[overall] < _SEVERITY[Verdict.NEEDS_REVIEW]:
            overall = Verdict.NEEDS_REVIEW

    timing_ms = (time.perf_counter() - start) * 1000.0

    return VerificationResult(
        fields=fields,
        warning=warning_verdict,
        overall=overall,
        image_quality=quality,
        review_flags=_dedupe(review_flags),
        canonical_version=wr.canonical_version,
        timing_ms=round(timing_ms, 2),
        regions=_build_regions(ext, fields, warning_verdict),
    )
