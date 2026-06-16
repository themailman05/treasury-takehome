"""Pydantic models = the contract between the model, the verifier, and the API.

Two directions:

  * **Model -> verifier** (``ModelExtraction``): what the VLM is prompted to
    return, enforced by guided/structured JSON decoding. The verifier consumes
    only *extracted text* — it never trusts a model "verdict".
  * **Verifier -> client** (``VerificationResult`` and friends): the per-field
    checklist the UI renders.

Plus the application payload (``ApplicationData``) and the batch job models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from enums import ImageQuality, Verdict

# --------------------------------------------------------------------------- #
# Model -> verifier contract (guided-decoding target)
# --------------------------------------------------------------------------- #


class FieldReading(BaseModel):
    """A single extracted field value plus the model's confidence."""

    text: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    # [ymin, xmin, ymax, xmax] normalized to 0-1000 (Gemma's box convention),
    # used to draw the audit overlay on the label image.
    box: Optional[list[int]] = None


class WarningReading(BaseModel):
    """The government-warning region, read two independent ways.

    ``ocr_text`` (deterministic OCR of the cropped region) is authoritative;
    ``vlm_text`` is a cross-check only (compared against the OCR reading, never
    against the canonical text). See ``verify_warning``.
    """

    vlm_text: Optional[str] = None
    ocr_text: Optional[str] = None
    located: bool = False
    # [x, y, w, h] in pixels of the cropped warning region (legacy; OCR now reads
    # the bottom crop directly).
    bbox: Optional[list[int]] = None
    # [ymin, xmin, ymax, xmax] normalized to 0-1000 for the audit overlay.
    box: Optional[list[int]] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ModelExtraction(BaseModel):
    """Exactly what the VLM returns. No verdicts, no judgement — text only."""

    model_config = ConfigDict(use_enum_values=True)

    brand_name: FieldReading = Field(default_factory=FieldReading)
    class_type: FieldReading = Field(default_factory=FieldReading)
    abv: FieldReading = Field(default_factory=FieldReading)
    net_contents: FieldReading = Field(default_factory=FieldReading)
    warning: WarningReading = Field(default_factory=WarningReading)
    image_quality: ImageQuality = ImageQuality.OK


# --------------------------------------------------------------------------- #
# Application payload (what the applicant submitted on the COLA form)
# --------------------------------------------------------------------------- #


class ApplicationData(BaseModel):
    """Expected field values from the application. Only provided fields are
    checked; an omitted field is skipped (not failed)."""

    model_config = ConfigDict(extra="ignore")

    brand_name: Optional[str] = None
    class_type: Optional[str] = None
    abv: Optional[str] = None
    net_contents: Optional[str] = None


# --------------------------------------------------------------------------- #
# Verifier -> client output
# --------------------------------------------------------------------------- #


class FieldVerdict(BaseModel):
    """Verdict for one value field (brand, class/type, ABV, net contents)."""

    verdict: Verdict
    similarity: Optional[float] = None  # 0..100 rapidfuzz ratio on normalized text
    extracted: Optional[str] = None  # raw text the model read
    expected: Optional[str] = None  # value from the application
    normalized_extracted: Optional[str] = None
    normalized_expected: Optional[str] = None
    confidence: Optional[float] = None  # model confidence for the field
    note: Optional[str] = None  # e.g. "not_submitted", "low_confidence"


class WarningVerdict(BaseModel):
    """Verdict for the government warning (strict, dual-path)."""

    verdict: Verdict
    exact_match: bool  # all components within OCR-noise budget AND caps ok
    caps_ok: Optional[bool] = None  # None == header couldn't be located
    readings_agree: Optional[bool] = None  # None == single-path (no VLM reading)
    agreement_ratio: Optional[float] = None
    canonical_version: str
    components: list[dict] = Field(default_factory=list)  # per-component detail
    review_flags: list[str] = Field(default_factory=list)


class LabelRegion(BaseModel):
    """A located region on the label, for the visual audit overlay."""

    label: str  # e.g. "brand_name", "government_warning"
    box: list[int]  # [ymin, xmin, ymax, xmax] normalized 0-1000
    verdict: Optional[Verdict] = None  # None when the field wasn't compared
    text: str = ""


class VerificationResult(BaseModel):
    """The full per-label checklist the UI renders and the API returns."""

    fields: dict[str, FieldVerdict] = Field(default_factory=dict)
    warning: WarningVerdict
    overall: Verdict
    image_quality: ImageQuality = ImageQuality.OK
    review_flags: list[str] = Field(default_factory=list)
    canonical_version: str
    timing_ms: Optional[float] = None
    # Audit overlay: one entry per located region, drawn on the label image.
    # box is [ymin, xmin, ymax, xmax] normalized 0-1000; verdict colours the box.
    regions: list[LabelRegion] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Batch job models
# --------------------------------------------------------------------------- #


class ItemStatus(BaseModel):
    """One label inside a batch job."""

    index: int
    filename: str
    status: str = "pending"  # pending | processing | done | error | duplicate
    result: Optional[VerificationResult] = None
    error: Optional[str] = None


class JobStatus(BaseModel):
    """Batch job progress + per-item results, polled by the UI."""

    job_id: str
    status: str = "queued"  # queued | processing | complete
    total: int = 0
    completed: int = 0
    failed: int = 0
    created_at: Optional[str] = None
    items: list[ItemStatus] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return (self.completed + self.failed) >= self.total and self.total > 0
