"""Inference clients = the *only* component that touches the VLM.

The single most important design rule of this tool is **the model extracts,
deterministic code judges** (README §5). So everything here returns a
``ModelExtraction`` of *verbatim text* and nothing else — no verdicts, no
"corrections". Three pieces:

  * ``EXTRACTION_PROMPT`` / ``EXTRACTION_JSON_SCHEMA`` — what Gemma 4 is told to
    return and the guided-decoding schema that enforces it.
  * ``MockInferenceClient`` — deterministic, model-free. It is what makes the repo
    runnable on a laptop and in CI, and it yields a clean PASS for a well-formed
    label out of the box.
  * ``OllamaInferenceClient`` — talks to a self-hosted Ollama endpoint serving
    Gemma 4 on **CPU** (the default ``docker compose`` profile). OpenAI-compatible.
  * ``VLLMInferenceClient`` — talks to a self-hosted vLLM endpoint serving Gemma 4
    on a **GPU** (the production throughput target, README §6). OpenAI-compatible.

Both real backends stay *inside the boundary* — no outbound ML traffic (README §4)
— and share one OpenAI-compatible client (``_OpenAICompatClient``); they differ only
in base URL and how they request structured JSON output. Every backend error is
wrapped in :class:`InferenceError` with a sanitized message so the API can return a
clean 502 (never a stack trace) and the batch worker can treat it as a retryable
item failure.

``build_inference_client`` dispatches on ``settings.inference_backend``.
"""

from __future__ import annotations

import abc
import base64
import json
import re
from io import BytesIO
from typing import Optional

import httpx
from pydantic import ValidationError

# Cap the longest image side sent to the VLM. Vision prefill cost scales with the
# number of image tiles, so a full-res ~2 MP label is slow; field text stays
# legible at this size and the warning's authoritative reading is full-res OCR.
_MAX_IMAGE_DIM = 1024
# Anti-runaway bound on the VLM extraction ONLY (fields + boxes; ~285 tokens in
# practice). NOTE: this never trims the warning — the warning text is read by
# Tesseract OCR, not the VLM. The cap exists because a busy label once made the
# model ramble; if a label still blows it, the truncated JSON → InferenceError →
# the pipeline degrades to needs_review (never a 502). Generous headroom so a
# legitimate extraction is never trimmed.
_MAX_OUTPUT_TOKENS = 1536


def _downscale_for_vlm(image_bytes: bytes, max_dim: int = _MAX_IMAGE_DIM) -> bytes:
    """Shrink an oversized label for a faster vision prefill. Best-effort: on any
    failure (non-image bytes, Pillow missing) return the original unchanged."""
    try:
        from PIL import Image

        im = Image.open(BytesIO(image_bytes))
        longest = max(im.size)
        if longest <= max_dim:
            return image_bytes
        scale = max_dim / longest
        im = im.convert("RGB").resize(
            (round(im.size[0] * scale), round(im.size[1] * scale))
        )
        out = BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return image_bytes

from config import Settings, canonical_warning, get_settings
from enums import ImageQuality
from schemas import (
    ApplicationData,
    FieldReading,
    ModelExtraction,
    WarningReading,
)

# --------------------------------------------------------------------------- #
# Model -> verifier contract: the extraction prompt + guided-decoding schema
# --------------------------------------------------------------------------- #

# The prompt is deliberately strict about *transcription, not correction*: a VLM
# asked to "read the warning" will helpfully emit the canonical text even over a
# defective label, which would silently pass a non-compliant warning. We forbid
# that here, and the warning is cross-checked against deterministic OCR anyway
# (see verify_warning).
EXTRACTION_PROMPT = (
    "You are an OCR/vision engine for U.S. alcohol-beverage (TTB COLA) label "
    "artwork. Read THIS label image and return the requested fields as JSON.\n"
    "\n"
    "Rules:\n"
    "- Transcribe each value EXACTLY as printed on this label — character for "
    "character, preserving casing and punctuation. Do NOT correct, complete, "
    "translate, or invent text, and NEVER copy any sample/placeholder value: every "
    "value must come from pixels in this image. If a field is not present, use an "
    "empty string and confidence 0.\n"
    "- Keep each value to just that one field — exclude surrounding marketing, "
    "slogans, addresses, and decorative text.\n"
    "\n"
    "Fields to return:\n"
    "- brand_name: the brand / product name.\n"
    "- class_type: the class or type designation (the spirit / wine / beer type).\n"
    "- abv: the alcohol-content statement as printed (the percent, with any proof).\n"
    "- net_contents: the net-contents statement as printed.\n"
    "- warning: the government health-warning region. Set 'located' true if you can "
    "find it (else false). Do NOT transcribe it — leave 'vlm_text' and 'ocr_text' "
    "empty; a separate OCR step reads the warning text.\n"
    "- image_quality: 'ok', 'poor' (glare / angle / blur), or 'unreadable'.\n"
    "\n"
    "Each text field carries a 'confidence' in [0, 1].\n"
    "\n"
    "BOUNDING BOXES: set 'box' = [ymin, xmin, ymax, xmax] normalized to 0-1000 "
    "(top-left origin) for brand_name, class_type, abv, net_contents, AND the "
    "warning (set warning.box). Use null for a field that is not present.\n"
    "\n"
    "Return ONLY a single JSON object matching the schema. No prose, no markdown."
)

# Exactly the verifier-side contract model, for vLLM guided/structured decoding.
EXTRACTION_JSON_SCHEMA: dict = ModelExtraction.model_json_schema()


# --------------------------------------------------------------------------- #
# Abstract client
# --------------------------------------------------------------------------- #


class InferenceClient(abc.ABC):
    """Extracts verbatim label text into a ``ModelExtraction``. No judgement."""

    @abc.abstractmethod
    async def extract(
        self,
        image_bytes: bytes,
        *,
        application: Optional[ApplicationData] = None,
    ) -> ModelExtraction:
        """Read ``image_bytes`` and return the extracted fields (text only)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release any held resources (HTTP connections, etc.)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Mock client — deterministic, no GPU
# --------------------------------------------------------------------------- #


class MockInferenceClient(InferenceClient):
    """Deterministic, GPU-free extractor that makes the service runnable anywhere.

    Two modes:

      * **Fixture** — if ``image_bytes`` decode to UTF-8 JSON, treat that JSON as
        a hand-authored ``ModelExtraction`` fixture (used by tests and demos to
        exercise defective-label paths). Missing value fields are filled by
        echoing the application.
      * **Echo** — otherwise, echo the application's expected values straight
        into the extracted fields and emit a clean, compliant warning. This
        yields a tidy PASS for a well-formed label with no model at all.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    async def extract(
        self,
        image_bytes: bytes,
        *,
        application: Optional[ApplicationData] = None,
    ) -> ModelExtraction:
        app = application or ApplicationData()
        fixture = self._try_parse_fixture(image_bytes)
        if fixture is not None:
            return self._extraction_from_fixture(fixture, app)
        return self._echo_extraction(app)

    async def aclose(self) -> None:  # noqa: D401 - nothing to release
        """No-op: the mock holds no resources."""
        return None

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _try_parse_fixture(image_bytes: bytes) -> Optional[dict]:
        """Return a dict if ``image_bytes`` are UTF-8 JSON, else ``None``."""
        try:
            decoded = image_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None
        decoded = decoded.strip()
        if not decoded:
            return None
        try:
            obj = json.loads(decoded)
        except (json.JSONDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None

    def _extraction_from_fixture(
        self, fixture: dict, app: ApplicationData
    ) -> ModelExtraction:
        """Build a ``ModelExtraction`` from a fixture, echoing the application
        only for value fields the fixture omits entirely."""
        data = dict(fixture)
        for name in ("brand_name", "class_type", "abv", "net_contents"):
            if name not in data:
                expected = getattr(app, name, None)
                if expected is not None:
                    data[name] = {"text": expected, "confidence": 0.97}
        return ModelExtraction.model_validate(data)

    def _echo_extraction(self, app: ApplicationData) -> ModelExtraction:
        """Echo the application values and emit a clean, compliant warning."""
        warning_text = canonical_warning(self._settings.warning_text_version)
        return ModelExtraction(
            brand_name=FieldReading(text=app.brand_name or "", confidence=0.97),
            class_type=FieldReading(text=app.class_type or "", confidence=0.97),
            abv=FieldReading(text=app.abv or "", confidence=0.97),
            net_contents=FieldReading(
                text=app.net_contents or "", confidence=0.97
            ),
            warning=WarningReading(
                vlm_text=warning_text,
                ocr_text=warning_text,
                located=True,
                bbox=[10, 200, 400, 80],
                confidence=0.95,
            ),
            image_quality=ImageQuality.OK,
        )


# --------------------------------------------------------------------------- #
# Real backends — self-hosted, OpenAI-compatible, structured-output
# --------------------------------------------------------------------------- #


class InferenceError(RuntimeError):
    """A failure talking to or parsing the inference backend.

    Raised with a *sanitized* message instead of leaking httpx / JSON / pydantic
    internals: the API turns it into a clean ``502`` (never a stack trace), and
    the batch worker treats it as a retryable item failure.
    """


def _extract_json(content: Optional[str]) -> dict:
    """Parse a model response into a JSON object, tolerantly.

    Structured decoding should return bare JSON, but a model can still wrap it in
    prose or a ```json fence. Strip a fence if present, else fall back to the
    outermost ``{...}`` span, then parse. Raises :class:`InferenceError` on any
    non-object / non-JSON result.
    """
    if not content or not content.strip():
        raise InferenceError("inference backend returned empty content")
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise InferenceError("model response was not valid JSON")
        try:
            obj = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError) as exc:
            raise InferenceError("model response was not valid JSON") from exc
    if not isinstance(obj, dict):
        raise InferenceError("model response was not a JSON object")
    return obj


def _clamp01(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _coerce_box(v: object) -> Optional[list[int]]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    out: list[int] = []
    for x in v:
        try:
            out.append(int(round(float(x))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return out


def _sanitize_field(f: object) -> dict:
    if isinstance(f, str):  # model returned a bare string instead of {text, ...}
        return {"text": f, "confidence": 0.9}
    if not isinstance(f, dict):
        return {"text": "", "confidence": 0.0}
    if "confidence" in f:
        f["confidence"] = _clamp01(f["confidence"])
    if "box" in f:
        f["box"] = _coerce_box(f["box"])
    return f


def _sanitize_extraction(d: dict) -> dict:
    """Coerce the model's near-miss values into the schema instead of rejecting.

    Ollama's ``format`` grammar enforces JSON *types*, not value constraints, so a
    model occasionally emits a confidence outside [0,1], a float box coordinate, a
    bare-string field, or an off-enum image_quality — all valid JSON that fails
    Pydantic. Clamping/coercing here keeps a readable label from being needlessly
    downgraded to needs_review."""
    if not isinstance(d, dict):
        return d
    for key in ("brand_name", "class_type", "abv", "net_contents"):
        if key in d:
            d[key] = _sanitize_field(d[key])
    w = d.get("warning")
    if isinstance(w, dict):
        if "confidence" in w:
            w["confidence"] = _clamp01(w["confidence"])
        if "box" in w:
            w["box"] = _coerce_box(w["box"])
        if "bbox" in w:
            w["bbox"] = _coerce_box(w["bbox"])
    elif w is not None and not isinstance(w, dict):
        d["warning"] = {}  # a non-object warning is meaningless; use defaults
    iq = d.get("image_quality")
    d["image_quality"] = iq.lower() if isinstance(iq, str) and iq.lower() in (
        "ok", "poor", "unreadable") else "ok"
    return d


def _parse_and_validate(content: Optional[str]) -> ModelExtraction:
    """Parse the model's JSON, sanitize near-misses, and validate. On a genuine
    schema failure, surface the offending field in the error (so logs are
    diagnosable). The warning's ocr_text is dropped (it comes from OCR)."""
    parsed = _sanitize_extraction(_extract_json(content))
    try:
        extraction = ModelExtraction.model_validate(parsed)
    except ValidationError as exc:
        first = (exc.errors() or [{}])[0]
        raise InferenceError(
            "model output did not match the extraction schema "
            f"(loc={first.get('loc')} type={first.get('type')})"
        ) from exc
    extraction.warning.ocr_text = None  # authoritative reading comes from OCR
    return extraction


class _OpenAICompatClient(InferenceClient):
    """Shared OpenAI-compatible chat-completions client for the self-hosted
    backends (vLLM on GPU, Ollama on CPU).

    Subclasses differ only in their base URL and in ``_structured_output`` (how
    they force a ``ModelExtraction``-shaped JSON reply). The warning's
    ``ocr_text`` is always discarded here; the deterministic OCR step in the
    pipeline fills it — the authoritative reading must not come from the VLM.
    """

    def __init__(self, settings: Optional[Settings], base_url: str) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=self._settings.inference_timeout_s,
            headers={
                "Authorization": f"Bearer {self._settings.vllm_api_key}",
                "Content-Type": "application/json",
            },
        )

    def _structured_output(self, payload: dict) -> None:
        """Mutate ``payload`` in place to force structured JSON output."""
        raise NotImplementedError

    async def extract(
        self,
        image_bytes: bytes,
        *,
        application: Optional[ApplicationData] = None,
    ) -> ModelExtraction:
        vlm_bytes = _downscale_for_vlm(image_bytes)
        data_uri = "data:image/png;base64," + base64.b64encode(vlm_bytes).decode(
            "ascii"
        )
        payload: dict = {
            "model": self._settings.gemma_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
            # Bound generation: without this, a verbose prompt can make the model
            # run away to thousands of tokens and blow the request timeout.
            "max_tokens": _MAX_OUTPUT_TOKENS,
        }
        self._structured_output(payload)

        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.TimeoutException as exc:
            raise InferenceError(
                f"inference backend timed out after "
                f"{self._settings.inference_timeout_s:g}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise InferenceError(
                f"inference backend returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError("could not reach the inference backend") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                "inference backend returned an unexpected response shape"
            ) from exc

        return _parse_and_validate(content)

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()


class VLLMInferenceClient(_OpenAICompatClient):
    """Self-hosted Gemma 4 via **vLLM** (GPU). Uses vLLM's ``guided_json``
    extension to force a valid ``ModelExtraction`` — no brittle prose-parsing."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        settings = settings or get_settings()
        super().__init__(settings, settings.vllm_base_url)

    def _structured_output(self, payload: dict) -> None:
        payload["guided_json"] = EXTRACTION_JSON_SCHEMA


class OllamaInferenceClient(InferenceClient):
    """Self-hosted Gemma 4 via Ollama's **native** ``/api/chat`` endpoint.

    Gemma 4 is a *reasoning* model. Through Ollama's OpenAI-compatible ``/v1``
    endpoint there's no way to turn that off, so its "thinking" tokens silently
    consume the generation budget and the JSON answer comes back **empty** (a
    502 in testing). The native API exposes two controls that fix it:

      * ``think: false`` — skip reasoning and answer directly;
      * ``format: <schema>`` — constrain output to a valid ``ModelExtraction``.

    Together these give a fast (~10s on an L4), correctly-shaped extraction. The
    image is downscaled first (vision prefill cost) and output is token-bounded.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        # ollama_base_url is the OpenAI-compatible ".../v1"; the native API is at the root.
        base = self._settings.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        self._chat_url = f"{base}/api/chat"
        self._client = httpx.AsyncClient(timeout=self._settings.inference_timeout_s)

    async def extract(
        self,
        image_bytes: bytes,
        *,
        application: Optional[ApplicationData] = None,
    ) -> ModelExtraction:
        img_b64 = base64.b64encode(_downscale_for_vlm(image_bytes)).decode("ascii")
        payload = {
            "model": self._settings.gemma_model,
            "messages": [
                {"role": "user", "content": EXTRACTION_PROMPT, "images": [img_b64]}
            ],
            "stream": False,
            "think": False,  # Gemma 4 reasons by default; disable it or content comes back empty
            "format": EXTRACTION_JSON_SCHEMA,  # schema-constrained -> valid ModelExtraction
            "keep_alive": -1,  # keep the model resident — avoids a ~30s cold-start reload after idle
            "options": {"num_predict": _MAX_OUTPUT_TOKENS, "temperature": 0.0},
        }
        try:
            resp = await self._client.post(self._chat_url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.TimeoutException as exc:
            raise InferenceError(
                f"inference backend timed out after "
                f"{self._settings.inference_timeout_s:g}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise InferenceError(
                f"inference backend returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError("could not reach the inference backend") from exc

        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise InferenceError(
                "inference backend returned an unexpected response shape"
            ) from exc

        return _parse_and_validate(content)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_inference_client(settings: Settings) -> InferenceClient:
    """Construct the inference client selected by ``settings.inference_backend``."""
    if settings.inference_backend == "vllm":
        return VLLMInferenceClient(settings)
    if settings.inference_backend == "ollama":
        return OllamaInferenceClient(settings)
    return MockInferenceClient(settings)
