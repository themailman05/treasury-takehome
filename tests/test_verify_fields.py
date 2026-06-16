"""Field-matching tests (README §5 field rules + Dave's STONE'S THROW case).

These exercise ``verify_fields`` / ``verify_field`` directly: brand fuzzy
matching across the pass / needs_review / fail bands, numeric ABV comparison,
net-contents unit normalization, the omit-an-unsubmitted-field rule, and the
low-confidence override.
"""

from __future__ import annotations

from config import get_settings
from enums import Verdict
from schemas import (
    ApplicationData,
    FieldReading,
    ModelExtraction,
    WarningReading,
)
from verify_fields import (
    normalize_abv,
    normalize_field,
    normalize_net_contents,
    verify_field,
    verify_fields,
)

SETTINGS = get_settings()


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #


def test_normalize_field_casefolds_collapses_and_strips():
    assert normalize_field("  STONE'S   THROW.  ") == "stone's throw"


def test_normalize_abv_parses_percent_proof_and_bare_number():
    assert normalize_abv("45% Alc./Vol. (90 Proof)") == 45.0
    assert normalize_abv("45") == 45.0
    assert normalize_abv("not a number") is None


def test_normalize_abv_prefers_percent_over_proof():
    # The percentage is the ABV — not the first number, which is often the proof.
    assert normalize_abv("80 proof, 40% Alc./Vol.") == 40.0
    # A cue word (alc/vol/abv) also resolves the right number.
    assert normalize_abv("Alc 13.5 by Vol") == 13.5
    assert normalize_abv("ABV: 45") == 45.0


def test_normalize_abv_accepts_comma_decimals():
    # European format common on imported wine/spirits TTB reviews heavily.
    assert normalize_abv("12,5%") == 12.5
    assert normalize_abv("12,5% Alc./Vol.") == 12.5


def test_normalize_abv_converts_bare_proof_to_abv():
    # No percentage present -> a bare proof value is half its ABV.
    assert normalize_abv("90 Proof") == 45.0


def test_abv_comma_decimal_matches_dot_decimal():
    fv = verify_field("abv", "12,5% Alc./Vol.", "12.5", settings=SETTINGS)
    assert fv.verdict is Verdict.PASS
    assert fv.similarity == 100.0


def test_abv_bare_proof_matches_declared_abv():
    fv = verify_field("abv", "90 Proof", "45", settings=SETTINGS)
    assert fv.verdict is Verdict.PASS


def test_normalize_net_contents_unifies_units():
    assert (
        normalize_net_contents("750 mL")
        == normalize_net_contents("750ml")
        == normalize_net_contents("750 ML")
        == "750 ml"
    )


# --------------------------------------------------------------------------- #
# Brand name: pass / needs_review / fail bands
# --------------------------------------------------------------------------- #


def test_brand_exact_match_passes():
    fv = verify_field("brand_name", "OLD TOM DISTILLERY", "OLD TOM DISTILLERY",
                      settings=SETTINGS)
    assert fv.verdict is Verdict.PASS
    assert fv.similarity == 100.0


def test_brand_caps_vs_titlecase_passes_daves_case():
    # Dave's interview case: "STONE'S THROW" and "Stone's Throw" are one brand.
    fv = verify_field("brand_name", "STONE'S THROW", "Stone's Throw",
                      settings=SETTINGS)
    assert fv.verdict is Verdict.PASS
    assert fv.similarity == 100.0


def test_brand_clearly_different_fails():
    fv = verify_field("brand_name", "OLD TOM DISTILLERY", "Acme Vodka Corp",
                      settings=SETTINGS)
    assert fv.verdict is Verdict.FAIL
    assert fv.similarity < SETTINGS.fuzzy_review_threshold


def test_brand_borderline_needs_review():
    # ~80.9 similarity: inside [review_threshold, pass_threshold) -> needs_review.
    fv = verify_field(
        "brand_name", "Mountain Ridge Brewery", "Mountain Ridge Brewing Co",
        settings=SETTINGS,
    )
    assert fv.verdict is Verdict.NEEDS_REVIEW
    assert SETTINGS.fuzzy_review_threshold <= fv.similarity < SETTINGS.fuzzy_pass_threshold


# --------------------------------------------------------------------------- #
# ABV: numeric comparison
# --------------------------------------------------------------------------- #


def test_abv_percent_proof_matches_bare_number():
    fv = verify_field("abv", "45% Alc./Vol. (90 Proof)", "45", settings=SETTINGS)
    assert fv.verdict is Verdict.PASS
    assert fv.similarity == 100.0


def test_abv_mismatch_fails():
    fv = verify_field("abv", "45", "40", settings=SETTINGS)
    assert fv.verdict is Verdict.FAIL
    assert fv.similarity == 0.0


# --------------------------------------------------------------------------- #
# Net contents: unit-aware
# --------------------------------------------------------------------------- #


def test_net_contents_unit_variants_pass():
    fv = verify_field("net_contents", "750 mL", "750ml", settings=SETTINGS)
    assert fv.verdict is Verdict.PASS
    assert fv.normalized_extracted == "750 ml"
    assert fv.normalized_expected == "750 ml"


# --------------------------------------------------------------------------- #
# Confidence + not-extracted overrides
# --------------------------------------------------------------------------- #


def test_low_confidence_routes_to_needs_review():
    # Otherwise a clean exact match, but confidence below min_field_confidence.
    fv = verify_field(
        "brand_name", "OLD TOM DISTILLERY", "OLD TOM DISTILLERY",
        confidence=0.10, settings=SETTINGS,
    )
    assert fv.verdict is Verdict.NEEDS_REVIEW
    assert fv.note == "low_confidence"


def test_submitted_but_not_extracted_needs_review():
    fv = verify_field("brand_name", "", "OLD TOM DISTILLERY", settings=SETTINGS)
    assert fv.verdict is Verdict.NEEDS_REVIEW
    assert fv.note == "not_extracted"


# --------------------------------------------------------------------------- #
# verify_fields: only submitted fields appear
# --------------------------------------------------------------------------- #


def _extraction(**fields) -> ModelExtraction:
    """Build a ModelExtraction with the given value-field texts (conf 0.97)."""
    return ModelExtraction(
        brand_name=FieldReading(text=fields.get("brand_name", ""), confidence=0.97),
        class_type=FieldReading(text=fields.get("class_type", ""), confidence=0.97),
        abv=FieldReading(text=fields.get("abv", ""), confidence=0.97),
        net_contents=FieldReading(text=fields.get("net_contents", ""), confidence=0.97),
        warning=WarningReading(),
    )


def test_omitted_application_field_is_absent_from_results():
    ext = _extraction(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        abv="45% Alc./Vol.",
        net_contents="750 mL",
    )
    # Application submits only brand_name and abv; class_type / net_contents omitted.
    app = ApplicationData(brand_name="OLD TOM DISTILLERY", abv="45")
    results = verify_fields(ext, app, settings=SETTINGS)

    assert set(results) == {"brand_name", "abv"}
    assert "class_type" not in results
    assert "net_contents" not in results
    assert results["brand_name"].verdict is Verdict.PASS
    assert results["abv"].verdict is Verdict.PASS


def test_blank_application_field_is_also_skipped():
    ext = _extraction(brand_name="OLD TOM DISTILLERY")
    app = ApplicationData(brand_name="OLD TOM DISTILLERY", class_type="   ")
    results = verify_fields(ext, app, settings=SETTINGS)
    assert set(results) == {"brand_name"}
