"""Government-warning comparator tests (README §5, the six documented scenarios).

The warning is the field that must *not* trust the VLM transcription. These
assert the documented verdict for each of the six scenarios the comparator's
``__main__`` smoke test covers:

    clean             -> pass
    OCR noise         -> pass   (tolerates rn->m, O->0 within the edit budget)
    title-case header -> fail   (Jenny's reject)
    missing clause    -> fail
    VLM hallucination -> needs_review (OCR read the real defect; VLM "fixed" it)
    reworded clause   -> fail

The canonical text comes from ``config.canonical_warning`` (a versioned config
value, never a literal in the test).
"""

from __future__ import annotations

from config import (
    CANONICAL_CLAUSE_1,
    CANONICAL_HEADER,
    canonical_warning,
)
from enums import Verdict
from verify_warning import verify_warning

CANONICAL = canonical_warning()


def test_clean_label_passes():
    wr = verify_warning(CANONICAL, CANONICAL)
    assert wr.verdict is Verdict.PASS
    assert wr.exact_match is True
    assert wr.caps_ok is True
    assert wr.readings_agree is True


def test_ocr_noise_still_passes():
    # Real OCR artifacts (O->0, rn->m) inside the per-component edit budget.
    noisy = CANONICAL.replace("operate", "0perate").replace("machinery", "machmery")
    wr = verify_warning(noisy, CANONICAL)
    assert wr.verdict is Verdict.PASS
    assert wr.exact_match is True
    assert wr.readings_agree is True


def test_title_case_header_fails():
    # Jenny's reject: header not in caps. Both readings agree it's title-case.
    titlecased = CANONICAL.replace(CANONICAL_HEADER, "Government Warning:")
    wr = verify_warning(titlecased, titlecased)
    assert wr.verdict is Verdict.FAIL
    assert wr.caps_ok is False
    assert wr.exact_match is False
    assert "header_not_caps" in wr.review_flags


def test_missing_clause_fails():
    # Clause (2) dropped entirely; both readings agree on the truncated label.
    missing = f"{CANONICAL_HEADER} {CANONICAL_CLAUSE_1}"
    wr = verify_warning(missing, missing)
    assert wr.verdict is Verdict.FAIL
    assert wr.exact_match is False
    assert any(f.startswith("component_deviates") for f in wr.review_flags)


def test_vlm_hallucination_does_not_autopass():
    # OCR reads the real defect (clause 2 missing); the VLM emits canonical from
    # memory. The OCR is authoritative, so the missing clause -> FAIL (never the
    # VLM's hallucinated pass). The VLM/OCR disagreement is also surfaced.
    ocr_defective = f"{CANONICAL_HEADER} {CANONICAL_CLAUSE_1}"
    wr = verify_warning(ocr_defective, CANONICAL)
    assert wr.verdict is Verdict.FAIL
    assert wr.readings_agree is False
    assert "ocr_vlm_disagree" in wr.review_flags


def test_reworded_clause_is_caught():
    # A substantive reword (the non-compliant softening Jenny flagged). It must not
    # PASS; depending on how much wording survives it lands on fail or needs_review.
    reworded = CANONICAL.replace(
        "women should not drink alcoholic beverages during pregnancy "
        "because of the risk of birth defects",
        "pregnant women may drink alcohol in moderation",
    )
    wr = verify_warning(reworded, reworded)
    assert wr.verdict in (Verdict.FAIL, Verdict.NEEDS_REVIEW)
    assert wr.exact_match is False
    assert any(f.startswith("component_deviates") for f in wr.review_flags)


def test_canonical_version_is_reported():
    wr = verify_warning(CANONICAL, CANONICAL)
    assert wr.canonical_version  # non-empty version string carried through
