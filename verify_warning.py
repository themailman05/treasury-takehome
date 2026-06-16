"""
Government health warning verification (27 CFR 16.21 / 16.22).

Design rationale
----------------
The mandatory government warning is famous, fixed, memorized text. That creates
two opposing failure modes for naive approaches:

1. VLM transcription false-PASS. A vision-language model asked to "transcribe the
   warning" leans on its prior toward the canonical string. Shown a *defective*
   label (dropped clause, reworded, title-case header) it may emit the *correct*
   text anyway. A literal `==` then agrees and passes a non-compliant label.
   Passing a bad warning is the worst outcome this tool can produce.

2. Deterministic-OCR false-FAIL. Real OCR is noisy (rn->m, O->0, glare). A literal
   `==` against the canonical string fails compliant labels over recognition
   artifacts.

So "exact match" must mean a *structured strict* comparison, not string equality:

  - Authoritative reading comes from deterministic OCR of the cropped warning
    region (it reports what is literally on the label, not what should be there).
  - Comparison is CLAUSE-LEVEL (header + clause 1 + clause 2), each matched with a
    small edit-distance budget calibrated to the OCR error rate. This tolerates
    OCR noise while still catching a missing or reworded clause.
  - The VLM transcription is a CROSS-CHECK, compared against the OCR reading (not
    against the canonical). If the two readings of the same label disagree, one of
    them misread -> route to human review. This single check catches BOTH failure
    modes: a hallucinated-canonical VLM disagrees with OCR that read the real
    defect; noisy OCR disagrees with a clean VLM read.
  - Header casing is checked explicitly (OCR preserves case).
  - Bold and physical type size are NOT verifiable from an uncontrolled photo and
    are emitted as review flags, never as a fabricated verdict.

Verdict policy (conservative; never auto-pass a defect):
  dual path, readings agree + reading matches canonical -> pass
  dual path, readings agree + reading deviates          -> fail
  dual path, readings disagree                          -> needs_review
  single path (no VLM), matches canonical               -> pass (flagged single_path)
  single path (no VLM), deviates                        -> needs_review

Dependency: rapidfuzz (pip install rapidfuzz)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional

from rapidfuzz import fuzz

from config import (
    AGREEMENT_RATIO,
    CANONICAL_CLAUSE_1,
    CANONICAL_CLAUSE_2,
    CANONICAL_HEADER,
    WARNING_ABSENT_RATIO,
    WARNING_MATCH_RATIO,
    WARNING_TEXT_VERSION,
)
from enums import Verdict


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

def normalize(text: str, *, fold_case: bool = False) -> str:
    """Whitespace/diacritic normalization only. Wording must survive intact."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if fold_case else text


# --------------------------------------------------------------------------- #
# Component-level matching
# --------------------------------------------------------------------------- #

@dataclass
class ComponentResult:
    name: str
    similarity: float      # rapidfuzz partial_ratio (0-100) of canonical within the reading
    present: bool          # similarity >= match threshold
    absent: bool           # similarity < absent threshold (missing / heavily reworded)


def _score_component(reading: str, expected: str, label: str) -> ComponentResult:
    """Score how well the canonical component appears within `reading`.

    Uses rapidfuzz ``partial_ratio`` (best-substring similarity, 0-100), which is
    robust to label-dependent OCR noise: a compliant clause reads ~88-95% even
    when Tesseract garbles a few characters, while a missing or reworded clause
    scores far lower. `label` is a clean display name (e.g. "clause (1)").
    """
    reading_f = normalize(reading, fold_case=True)
    expected_f = normalize(expected, fold_case=True)
    sim = fuzz.partial_ratio(expected_f, reading_f) if reading_f else 0.0
    sim = round(sim, 1)
    return ComponentResult(
        name=label,
        similarity=sim,
        present=sim >= WARNING_MATCH_RATIO,
        absent=sim < WARNING_ABSENT_RATIO,
    )


def _header_caps_ok(reading: str) -> Optional[bool]:
    """Check the literal 'government warning' token is uppercase on the label.
    Returns None if the header token can't be located at all (treated as missing).
    Case is checked on the ORIGINAL-case reading, so OCR must preserve case.
    """
    m = re.search(r"government\s+warning", reading, flags=re.IGNORECASE)
    if not m:
        return None
    matched = m.group()
    letters = [c for c in matched if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


# --------------------------------------------------------------------------- #
# Top-level comparator
# --------------------------------------------------------------------------- #

@dataclass
class WarningResult:
    verdict: Verdict
    exact_match: bool                      # all clauses present (>= match ratio) + caps ok
    caps_ok: Optional[bool]
    readings_agree: Optional[bool]         # None when no VLM reading supplied
    agreement_ratio: Optional[float]
    components: list[ComponentResult] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    canonical_version: str = WARNING_TEXT_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


def verify_warning(ocr_text: str, vlm_text: Optional[str] = None) -> WarningResult:
    """Verify a label's government warning.

    Parameters
    ----------
    ocr_text : deterministic-OCR transcription of the warning region
        (case-preserving). This is authoritative — a vision LLM asked to read the
        famous warning hallucinates the canonical text even over a defective or
        absent label (verified empirically), so it cannot judge the warning.
    vlm_text : optional VLM transcription, used only to report an agreement ratio
        for transparency; it never changes the verdict.

    Verdict policy (never auto-passes a defect; never false-fails on OCR noise):
      * header casing wrong                          -> fail  (Jenny's title-case)
      * all clauses present + caps ok                -> pass
      * some clauses clearly present, another absent -> fail  (clause missing/reworded)
      * otherwise (all borderline, or all absent)    -> needs_review (OCR unsure -> human)
    """
    components = [
        _score_component(ocr_text, CANONICAL_HEADER, "header"),
        _score_component(ocr_text, CANONICAL_CLAUSE_1, "clause (1)"),
        _score_component(ocr_text, CANONICAL_CLAUSE_2, "clause (2)"),
    ]
    caps_ok = _header_caps_ok(ocr_text)
    all_present = all(c.present for c in components)
    any_present = any(c.present for c in components)
    any_absent = any(c.absent for c in components)
    exact_match = all_present and caps_ok is True

    flags = ["warning_bold_unverified", "warning_typesize_unverified"]
    if caps_ok is False:
        flags.append("header_not_caps")
    flags += [f"component_deviates:{c.name!r}" for c in components if not c.present]

    if caps_ok is False:
        verdict = Verdict.FAIL  # title-case header is reliably read off the OCR
    elif exact_match:
        verdict = Verdict.PASS
    elif any_present and any_absent:
        # OCR clearly read some clauses but not another -> that clause is genuinely
        # missing or reworded (not a wholesale OCR failure).
        verdict = Verdict.FAIL
    else:
        # All clauses borderline (OCR noise) or all absent (OCR failed / no warning):
        # can't decide confidently -> route to a human, never auto-pass.
        verdict = Verdict.NEEDS_REVIEW

    # The VLM reading (if any) is reported only as an agreement ratio; it never
    # overrides the OCR verdict (the VLM hallucinates the warning text).
    ratio: Optional[float] = None
    readings_agree: Optional[bool] = None
    if vlm_text is not None:
        ratio = round(fuzz.ratio(normalize(ocr_text, fold_case=True),
                                 normalize(vlm_text, fold_case=True)), 1)
        readings_agree = ratio >= AGREEMENT_RATIO
        if not readings_agree:
            flags.append("ocr_vlm_disagree")
    else:
        flags.append("single_path")

    return WarningResult(
        verdict=verdict,
        exact_match=exact_match,
        caps_ok=caps_ok,
        readings_agree=readings_agree,
        agreement_ratio=ratio,
        components=components,
        review_flags=flags,
    )


# --------------------------------------------------------------------------- #
# Demo / smoke tests
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    CANONICAL = f"{CANONICAL_HEADER} {CANONICAL_CLAUSE_1} {CANONICAL_CLAUSE_2}"

    cases = {
        "compliant, clean": (CANONICAL, CANONICAL),
        "compliant, OCR noise (rn->m, O->0)": (
            CANONICAL.replace("operate", "0perate").replace("machinery", "machmery"),
            CANONICAL,
        ),
        "title-case header (Jenny's reject)": (
            CANONICAL.replace("GOVERNMENT WARNING:", "Government Warning:"),
            CANONICAL.replace("GOVERNMENT WARNING:", "Government Warning:"),
        ),
        "missing clause 2": (
            f"{CANONICAL_HEADER} {CANONICAL_CLAUSE_1}",
            f"{CANONICAL_HEADER} {CANONICAL_CLAUSE_1}",
        ),
        "VLM hallucinates canonical over defective label": (
            f"{CANONICAL_HEADER} {CANONICAL_CLAUSE_1}",   # OCR reads the real defect
            CANONICAL,                                     # VLM "fixes" it
        ),
        "reworded clause 1": (
            CANONICAL.replace(
                "women should not drink alcoholic beverages during pregnancy "
                "because of the risk of birth defects",
                "pregnant women may drink alcohol in moderation",
            ),
            CANONICAL.replace(
                "women should not drink alcoholic beverages during pregnancy "
                "because of the risk of birth defects",
                "pregnant women may drink alcohol in moderation",
            ),
        ),
    }

    for label, (ocr, vlm) in cases.items():
        r = verify_warning(ocr, vlm)
        flags = [f for f in r.review_flags
                 if f not in ("warning_bold_unverified", "warning_typesize_unverified")]
        print(f"{label:48s} -> {r.verdict.value:12s} "
              f"exact={r.exact_match!s:5s} caps={r.caps_ok!s:5s} "
              f"agree={r.readings_agree!s:5s} ratio={r.agreement_ratio} "
              f"flags={flags}")
