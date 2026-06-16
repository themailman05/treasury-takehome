"""Shared enums. Pure stdlib so every module (including the standalone
``verify_warning`` comparator) can import these without pulling in pydantic,
fastapi, or redis."""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """Per-field and overall verdict.

    Conservative by design: a borderline field is ``NEEDS_REVIEW`` rather than
    auto-rejected, and the warning pipeline never auto-passes a defect (see
    ``verify_warning``).
    """

    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"


class ImageQuality(str, Enum):
    """Model's self-reported read quality of the supplied image.

    ``UNREADABLE`` short-circuits the pipeline to ``needs_review`` so one bad
    photo never *fails* a label (and never fails a whole batch)."""

    OK = "ok"
    POOR = "poor"
    UNREADABLE = "unreadable"