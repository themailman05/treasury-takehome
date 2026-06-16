"""Central configuration.

Everything tunable lives here so a rule change (e.g. an amended government
warning) or a threshold recalibration is a config bump, not a code edit. Values
are read from the environment via pydantic-settings; sensible defaults make the
service runnable locally with no setup (``INFERENCE_BACKEND=mock``,
``OCR_BACKEND=mock``, no Redis).

The canonical warning text is kept here as a *versioned* config value
(``WARNING_TEXT_VERSION``) rather than a literal buried in the comparator — a
future cancer-warning amendment becomes a new version string + new clauses.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Canonical government health warning (27 CFR 16.21).
#
# Module-level constants so the standalone ``verify_warning`` comparator can
# import them without instantiating Settings. The active version is selected by
# ``WARNING_TEXT_VERSION``; add a new entry to bump the rule.
# --------------------------------------------------------------------------- #

WARNING_TEXT_VERSION = "1988-ABLA"

CANONICAL_HEADER = "GOVERNMENT WARNING:"
CANONICAL_CLAUSE_1 = (
    "(1) According to the Surgeon General, women should not drink alcoholic "
    "beverages during pregnancy because of the risk of birth defects."
)
CANONICAL_CLAUSE_2 = (
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car "
    "or operate machinery, and may cause health problems."
)

# Registry of warning versions. Keyed by version string -> (header, clause1, clause2).
WARNING_VERSIONS: dict[str, tuple[str, str, str]] = {
    "1988-ABLA": (CANONICAL_HEADER, CANONICAL_CLAUSE_1, CANONICAL_CLAUSE_2),
}

# Warning-comparator defaults, exposed as module constants so the standalone
# ``verify_warning`` comparator can import them. Settings (below) reuse these as
# field defaults, so env vars override them at the service layer.
#
# OCR_ERROR_RATE: per-character edit budget for OCR noise, as a fraction of the
# expected component length. Tolerates routine OCR noise (rn->m, O->0). The primary
# defense against label-dependent OCR noise lives in verify_warning: when the OCR
# and VLM readings agree, each clause is judged on whichever reading is closer to
# canonical, so this budget mainly governs the single-path (OCR-only) case.
OCR_ERROR_RATE = 0.10
# AGREEMENT_RATIO: rapidfuzz ratio (0-100) at/above which the two independent
# readings of the warning are treated as agreeing (reported for transparency).
AGREEMENT_RATIO = 90.0
# Warning clause similarity bands (rapidfuzz partial_ratio, 0-100), robust to
# label-dependent OCR noise where an absolute edit budget was not:
#   sim >= WARNING_MATCH_RATIO   -> the clause is present and matches canonical
#   sim <  WARNING_ABSENT_RATIO  -> the clause is missing or heavily reworded
#   in between                   -> ambiguous (OCR noise vs a real edit) -> review
WARNING_MATCH_RATIO = 85.0
WARNING_ABSENT_RATIO = 60.0


def canonical_components(version: str = WARNING_TEXT_VERSION) -> tuple[str, str, str]:
    """Return (header, clause_1, clause_2) for a warning-text version."""
    try:
        return WARNING_VERSIONS[version]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown WARNING_TEXT_VERSION {version!r}; "
            f"known: {sorted(WARNING_VERSIONS)}"
        ) from exc


def canonical_warning(version: str = WARNING_TEXT_VERSION) -> str:
    """The full canonical warning string for a version (header + both clauses)."""
    header, c1, c2 = canonical_components(version)
    return f"{header} {c1} {c2}"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Inference (the self-hosted Gemma 4 VLM) ---------------------------- #
    # "mock"   -> no model at all: deterministic fake extractions. This is what
    #             makes the repo runnable on a laptop and in CI (and the test
    #             default).
    # "ollama" -> a self-hosted Ollama endpoint serving Gemma 4 on CPU. This is
    #             the default `docker compose up` profile — no GPU required.
    # "vllm"   -> a self-hosted vLLM endpoint serving Gemma 4 on a GPU (the
    #             higher-throughput production target in README §6).
    # Both real backends are OpenAI-compatible and stay INSIDE the boundary —
    # no outbound ML traffic (Marcus's firewall constraint, README §4).
    inference_backend: Literal["mock", "vllm", "ollama"] = Field("mock")
    vllm_base_url: str = Field("http://localhost:8001/v1")      # GPU / vLLM
    ollama_base_url: str = Field("http://localhost:11434/v1")   # CPU / Ollama
    # Sent as `Authorization: Bearer ...`. vLLM/Ollama ignore the value but the
    # OpenAI-compatible protocol requires a non-empty string.
    vllm_api_key: str = Field("EMPTY")
    # Model id/tag named in the chat request. For Ollama use a VISION-capable tag
    # (gemma4:e4b / gemma4:e2b / gemma4:12b) — never a *-mlx tag, which is
    # text-only and cannot read a label. For vLLM use the HF id (google/gemma-4-12B).
    gemma_model: str = Field("gemma4:e4b")
    # Per-request timeout (seconds). Generous by default because CPU inference is
    # slow; on a GPU you can drop this toward the interactive SLA.
    inference_timeout_s: float = Field(120.0)

    @property
    def inference_base_url(self) -> str:
        """OpenAI-compatible base URL for the selected real backend."""
        if self.inference_backend == "ollama":
            return self.ollama_base_url
        return self.vllm_base_url

    # --- Deterministic OCR of the cropped warning region -------------------- #
    # "mock" returns a canned reading (laptop/CI). "tesseract" runs pytesseract
    # on the cropped bbox. The warning is the one field that must NOT trust the
    # VLM transcription — see verify_warning.
    ocr_backend: Literal["mock", "tesseract"] = Field("mock")

    # --- Warning verification ---------------------------------------------- #
    warning_text_version: str = Field(WARNING_TEXT_VERSION)
    # Per-character edit budget for OCR noise, as a fraction of component length.
    # 0.06 ~= tolerate one bad char per ~16. Calibrate to the OCR engine.
    ocr_error_rate: float = Field(OCR_ERROR_RATE, ge=0.0, le=0.5)
    # rapidfuzz ratio (0-100) at/above which two independent readings "agree".
    agreement_ratio: float = Field(AGREEMENT_RATIO, ge=0.0, le=100.0)

    # --- Field matching (brand, class/type, ABV, net contents) -------------- #
    # similarity >= pass_threshold        -> pass
    # review_threshold <= sim < pass      -> needs_review  (Dave's STONE'S THROW)
    # similarity <  review_threshold      -> fail
    fuzzy_pass_threshold: float = Field(88.0, ge=0.0, le=100.0)
    fuzzy_review_threshold: float = Field(72.0, ge=0.0, le=100.0)
    # Below this model confidence a field routes to needs_review (bad photo).
    min_field_confidence: float = Field(0.40, ge=0.0, le=1.0)

    # --- Batch lane / ephemeral store -------------------------------------- #
    # Empty REDIS_URL -> in-process batch executor (laptop/CI). Set it (compose
    # does) to use the Redis stream + arq worker pool described in the README.
    redis_url: str = Field("")
    result_ttl_seconds: int = Field(3600, ge=1)
    max_batch_retries: int = Field(3, ge=0)
    arq_queue_name: str = Field("label_jobs")

    # --- Service ------------------------------------------------------------ #
    # Informational SLA target for the interactive lane (README §2).
    interactive_latency_budget_s: float = Field(5.0)
    max_upload_bytes: int = Field(15 * 1024 * 1024)  # 15 MiB per image
    max_batch_images: int = Field(500, ge=1)  # reject /jobs requests above this
    static_dir: str = Field("static")

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url.strip())


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached Settings. Tests can clear via ``get_settings.cache_clear()``."""
    return Settings()
