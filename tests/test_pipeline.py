"""End-to-end pipeline tests over the mock backends (no GPU, no Redis).

Uses ``build_inference_client`` / ``build_ocr_client`` so the wiring under test
is the same the service uses. Three flows:

  * a clean label (the mock echo path) -> overall PASS, every field PASS, warning
    PASS;
  * a JSON-fixture image crafting a defective warning -> overall fail / review;
  * an ``unreadable`` image-quality fixture -> overall needs_review.

Pipeline coroutines are driven with ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import json

from config import canonical_warning, get_settings
from enums import ImageQuality, Verdict
from inference import build_inference_client
from ocr import build_ocr_client
from pipeline import verify_label
from schemas import ApplicationData

SETTINGS = get_settings()
CANONICAL = canonical_warning(SETTINGS.warning_text_version)


def _run_label(image_bytes: bytes, application: ApplicationData):
    """Drive ``verify_label`` to completion against fresh mock clients."""
    inference_client = build_inference_client(SETTINGS)
    ocr_client = build_ocr_client(SETTINGS)

    async def _go():
        try:
            return await verify_label(
                image_bytes,
                application,
                inference_client=inference_client,
                ocr_client=ocr_client,
                settings=SETTINGS,
            )
        finally:
            await inference_client.aclose()
            await ocr_client.aclose()

    return asyncio.run(_go())


def test_clean_label_is_overall_pass():
    # Non-JSON bytes -> mock echo path: extraction mirrors the application and a
    # clean canonical warning, so everything should pass.
    app = ApplicationData(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net_contents="750 mL",
    )
    result = _run_label(b"\x89PNG\r\n\x1a\n not-json image bytes", app)

    assert result.overall is Verdict.PASS
    assert result.image_quality is ImageQuality.OK
    assert set(result.fields) == {"brand_name", "class_type", "abv", "net_contents"}
    assert all(fv.verdict is Verdict.PASS for fv in result.fields.values())
    assert result.warning.verdict is Verdict.PASS
    assert result.warning.exact_match is True
    assert result.timing_ms is not None


def test_defective_warning_fixture_is_not_pass():
    # A fixture image: the VLM reads brand/abv fine, but the warning region is
    # missing clause (2) and both readings agree on the defect -> warning fail,
    # so the overall verdict can never be PASS.
    defective = f"{CANONICAL.split('(2)')[0].rstrip()}"  # header + clause 1 only
    fixture = {
        "brand_name": {"text": "OLD TOM DISTILLERY", "confidence": 0.97},
        "abv": {"text": "45% Alc./Vol.", "confidence": 0.96},
        "warning": {
            "vlm_text": defective,
            "ocr_text": defective,  # supplied -> pipeline does NOT re-OCR via mock
            "located": True,
            "bbox": [10, 200, 400, 80],
            "confidence": 0.9,
        },
        "image_quality": "ok",
    }
    app = ApplicationData(brand_name="OLD TOM DISTILLERY", abv="45")
    result = _run_label(json.dumps(fixture).encode("utf-8"), app)

    assert result.warning.verdict is Verdict.FAIL
    assert result.overall in (Verdict.FAIL, Verdict.NEEDS_REVIEW)
    assert result.overall is not Verdict.PASS
    # The value fields the applicant submitted still pass on their own.
    assert result.fields["brand_name"].verdict is Verdict.PASS
    assert result.fields["abv"].verdict is Verdict.PASS


def test_unreadable_image_is_needs_review():
    # An unreadable photo must resolve to needs_review (never fail a label).
    fixture = {
        "brand_name": {"text": "", "confidence": 0.1},
        "warning": {"located": False, "confidence": 0.0},
        "image_quality": "unreadable",
    }
    app = ApplicationData(brand_name="OLD TOM DISTILLERY")
    result = _run_label(json.dumps(fixture).encode("utf-8"), app)

    assert result.image_quality is ImageQuality.UNREADABLE
    assert result.overall is Verdict.NEEDS_REVIEW
    assert "unreadable" in result.review_flags
    # The warning couldn't be located -> the reason is surfaced explicitly.
    assert "warning:not_located" in result.review_flags


def test_unreadable_image_clamps_a_field_fail_to_needs_review():
    # Even with a high-confidence, clearly-mismatched field, an UNREADABLE photo
    # must NOT produce overall=FAIL — it clamps to needs_review (README §9).
    fixture = {
        "brand_name": {"text": "ACME VODKA CORP", "confidence": 0.99},
        "warning": {"located": False, "confidence": 0.0},
        "image_quality": "unreadable",
    }
    app = ApplicationData(brand_name="OLD TOM DISTILLERY")
    result = _run_label(json.dumps(fixture).encode("utf-8"), app)

    # The field itself still fails on its own merits...
    assert result.fields["brand_name"].verdict is Verdict.FAIL
    # ...but the overall verdict is clamped to needs_review, never FAIL.
    assert result.overall is Verdict.NEEDS_REVIEW
    assert "unreadable" in result.review_flags
