"""Deterministic field matching for the value fields (brand, class/type, ABV,
net contents).

Design rule (README §5): **the model extracts, deterministic code judges.** The
VLM gives us text; this module decides ``pass`` / ``fail`` / ``needs_review`` by
normalizing both the extracted and the expected value and comparing them — fuzzy
for free-text fields (so Dave's ``STONE'S THROW`` vs ``Stone's Throw`` is a pass,
not a reject), and numeric for ABV (so ``45% Alc./Vol. (90 Proof)`` and ``45``
agree). Net contents normalize their unit formatting before fuzzy comparison.

Conservative thresholds (from ``Settings``): a borderline similarity becomes
``needs_review`` rather than an auto-reject, and low model confidence or a field
the model failed to read also routes to review — never a fabricated ``fail``.

Only fields the applicant actually submitted are judged; an omitted application
field is skipped entirely (it never appears in the returned dict).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from rapidfuzz import fuzz

from config import Settings, get_settings
from enums import Verdict
from schemas import ApplicationData, FieldReading, FieldVerdict, ModelExtraction

# ABV equality tolerance: two parsed percentages within this many points are the
# same value (covers rounding like 45 vs 45.0 vs 45.00).
_ABV_TOLERANCE = 0.05

# Edge punctuation/whitespace stripped from normalized free-text values.
_EDGE_PUNCT = " \t\r\n.,;:!?\"'`()[]{}<>-_/\\|*&^%$#@~"

# Net-contents unit aliases -> canonical unit token. Keyed on the lowercased,
# trailing-dot-stripped unit string read off the label.
_VOLUME_UNITS: dict[str, str] = {
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "cl": "cl",
    "centiliter": "cl",
    "centiliters": "cl",
    "centilitre": "cl",
    "centilitres": "cl",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "floz": "fl oz",
    "floz.": "fl oz",
    "fl oz": "fl oz",
    "fluidounce": "fl oz",
    "fluidounces": "fl oz",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "gal": "gal",
    "gallon": "gal",
    "gallons": "gal",
    "pt": "pt",
    "pint": "pt",
    "pints": "pt",
    "qt": "qt",
    "quart": "qt",
    "quarts": "qt",
}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize_field(text: str) -> str:
    """Normalize a free-text field for fuzzy comparison.

    NFKC-normalize, case-fold, collapse internal whitespace to single spaces,
    and strip leading/trailing punctuation and whitespace. ``None`` -> ``""``.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.casefold()
    return text.strip(_EDGE_PUNCT)


_NUM = r"\d+(?:[.,]\d+)?"  # integer or decimal, dot OR comma (European "12,5")


def _to_float(token: str) -> float:
    """Parse a numeric token, treating a comma as the decimal separator."""
    return float(token.replace(",", "."))


def normalize_abv(text: str) -> Optional[float]:
    """Parse an alcohol-content statement to its ABV percentage, or ``None``.

    The label text often contains *more than one* number, so we don't just grab
    the first — we resolve to the value that is actually the alcohol-by-volume
    percentage, in priority order:

      1. a number tied to a ``%`` sign (``"45% Alc./Vol. (90 Proof)"`` -> 45.0,
         ``"12,5%"`` -> 12.5);
      2. a number adjacent to an ABV cue word (alc / vol / abv / alcohol)
         (``"Alc 13.5 by Vol"`` -> 13.5, ``"ABV: 45"`` -> 45.0);
      3. a number labelled as *proof*, converted to ABV = proof / 2
         (``"90 Proof"`` -> 45.0);
      4. otherwise the first bare number (``"45"`` -> 45.0).

    Comma decimals (the European format common on imported wine/spirits) are
    accepted everywhere. Returns ``None`` when no number is present, so the caller
    can fall back to fuzzy comparison rather than guess.
    """
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)
    low = text.lower()

    # 1. A number bound to a percent sign is unambiguously the ABV.
    m = re.search(rf"({_NUM})\s*%", low)
    if m:
        return _to_float(m.group(1))

    # 2. A number adjacent to an ABV cue word (cue may precede or follow).
    m = re.search(rf"({_NUM})\s*(?:alc|abv|alcohol|vol)", low) or re.search(
        rf"(?:alc|abv|alcohol|vol)[^0-9]{{0,12}}({_NUM})", low
    )
    if m:
        return _to_float(m.group(1))

    # 3. A proof value -> ABV is half of proof (US definition).
    m = re.search(rf"({_NUM})\s*proof", low)
    if m:
        return _to_float(m.group(1)) / 2.0

    # 4. Fall back to the first bare number (e.g. the application's plain "45").
    m = re.search(rf"({_NUM})", low)
    if m:
        return _to_float(m.group(1))
    return None


def normalize_net_contents(text: str) -> str:
    """Normalize a net-contents value: ``"750 mL"`` / ``"750ml"`` / ``"750 ML"``
    all become ``"750 ml"``.

    NFKC-normalizes, lowercases, collapses whitespace, then splits the numeric
    quantity from its unit and maps the unit through ``_VOLUME_UNITS`` to a
    canonical token. Unrecognized formats fall back to the case-folded,
    whitespace-collapsed string so they can still be fuzzy-matched.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*([a-z. ]+?)\.?$", text)
    if not m:
        return text
    qty_raw, unit_raw = m.group(1), m.group(2)
    quantity = qty_raw.replace(",", ".")
    # Drop a redundant trailing ".0" so "750.0" and "750" agree.
    if quantity.endswith(".0"):
        quantity = quantity[:-2]

    unit_key = re.sub(r"\s+", " ", unit_raw).strip(" .")
    canonical_unit = _VOLUME_UNITS.get(unit_key)
    if canonical_unit is None:
        # Also try the spaceless form (e.g. "fl oz" written as "floz").
        canonical_unit = _VOLUME_UNITS.get(unit_key.replace(" ", ""))
    if canonical_unit is None:
        return f"{quantity} {unit_key}".strip()
    return f"{quantity} {canonical_unit}"


# --------------------------------------------------------------------------- #
# Per-field verdict
# --------------------------------------------------------------------------- #


def _classify(similarity: float, settings: Settings) -> Verdict:
    """Map a 0-100 similarity to a verdict using the configured thresholds."""
    if similarity >= settings.fuzzy_pass_threshold:
        return Verdict.PASS
    if similarity >= settings.fuzzy_review_threshold:
        return Verdict.NEEDS_REVIEW
    return Verdict.FAIL


def verify_field(
    name: str,
    extracted: str,
    expected: Optional[str],
    *,
    confidence: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> FieldVerdict:
    """Verify one extracted value against the applicant's expected value.

    ``name`` selects the comparison strategy: ``abv`` is compared numerically
    (falling back to fuzzy if either side is unparseable), ``net_contents`` has
    its units normalized before fuzzy matching, everything else is fuzzy on the
    normalized free text. The similarity is mapped to a verdict via the
    configured thresholds, then two conservative overrides apply: an empty
    extraction (the model couldn't read a submitted field) -> ``needs_review``
    (``not_extracted``); a model confidence below ``min_field_confidence`` ->
    ``needs_review`` (``low_confidence``).
    """
    settings = settings or get_settings()
    extracted = extracted or ""
    expected = expected or ""

    note: Optional[str] = None

    # The applicant submitted a value but the model read nothing -> can't judge.
    if not extracted.strip():
        return FieldVerdict(
            verdict=Verdict.NEEDS_REVIEW,
            similarity=None,
            extracted=extracted,
            expected=expected,
            normalized_extracted="",
            normalized_expected=(
                normalize_net_contents(expected)
                if name == "net_contents"
                else normalize_field(expected)
            ),
            confidence=confidence,
            note="not_extracted",
        )

    if name == "abv":
        abv_extracted = normalize_abv(extracted)
        abv_expected = normalize_abv(expected)
        norm_extracted = "" if abv_extracted is None else f"{abv_extracted:g}"
        norm_expected = "" if abv_expected is None else f"{abv_expected:g}"
        if abv_extracted is not None and abv_expected is not None:
            # Pure numeric comparison: equal within tolerance is a clean pass.
            if abs(abv_extracted - abv_expected) <= _ABV_TOLERANCE:
                similarity = 100.0
            else:
                similarity = 0.0
        else:
            # One side wasn't a number — fall back to fuzzy on the raw text.
            norm_extracted = normalize_field(extracted)
            norm_expected = normalize_field(expected)
            similarity = fuzz.ratio(norm_extracted, norm_expected)
    elif name == "net_contents":
        norm_extracted = normalize_net_contents(extracted)
        norm_expected = normalize_net_contents(expected)
        similarity = fuzz.ratio(norm_extracted, norm_expected)
    else:
        norm_extracted = normalize_field(extracted)
        norm_expected = normalize_field(expected)
        similarity = fuzz.ratio(norm_extracted, norm_expected)

    verdict = _classify(similarity, settings)

    # Low model confidence overrides toward review — a bad photo shouldn't fail.
    if confidence is not None and confidence < settings.min_field_confidence:
        verdict = Verdict.NEEDS_REVIEW
        note = "low_confidence"

    return FieldVerdict(
        verdict=verdict,
        similarity=round(similarity, 1),
        extracted=extracted,
        expected=expected,
        normalized_extracted=norm_extracted,
        normalized_expected=norm_expected,
        confidence=confidence,
        note=note,
    )


def verify_fields(
    extraction: ModelExtraction,
    application: ApplicationData,
    *,
    settings: Optional[Settings] = None,
) -> dict[str, FieldVerdict]:
    """Verify every *submitted* application field against the model extraction.

    Only fields the applicant actually provided (non-empty) are judged; an
    omitted application field is skipped and never appears in the returned dict
    (an absent field is not a defect). Returns a mapping of field name ->
    ``FieldVerdict`` for ``brand_name``, ``class_type``, ``abv`` and
    ``net_contents``.
    """
    settings = settings or get_settings()

    readings: dict[str, FieldReading] = {
        "brand_name": extraction.brand_name,
        "class_type": extraction.class_type,
        "abv": extraction.abv,
        "net_contents": extraction.net_contents,
    }
    expected_values: dict[str, Optional[str]] = {
        "brand_name": application.brand_name,
        "class_type": application.class_type,
        "abv": application.abv,
        "net_contents": application.net_contents,
    }

    results: dict[str, FieldVerdict] = {}
    for name, reading in readings.items():
        expected = expected_values[name]
        # Skip fields the applicant didn't submit (None or blank).
        if expected is None or not expected.strip():
            continue
        results[name] = verify_field(
            name,
            reading.text,
            expected,
            confidence=reading.confidence,
            settings=settings,
        )
    return results
