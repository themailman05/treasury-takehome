"""FastAPI service = the two request lanes from README §4.

This is the HTTP front door for the whole tool. It wires the shared
dependencies (settings, the VLM inference client, the deterministic OCR client,
and the batch-job store) into two lanes:

  * **Interactive lane** — ``POST /verify`` runs one label synchronously through
    :func:`verify_label` and returns the full ``VerificationResult`` in the same
    response, targeting the sub-5s SLA (README §2).
  * **Batch lane** — ``POST /jobs`` accepts many images + a manifest, creates a
    ``job_id``, and fans the work out. With Redis configured it enqueues one
    ``arq`` task per item onto the shared queue; with no Redis it spawns one
    in-process ``asyncio`` task per item that runs ``verify_label`` and writes the
    result back into the (in-memory) store. ``GET /jobs/{job_id}`` polls progress.

The single-page UI is served from ``settings.static_dir`` and mounted at ``/``
**last**, so it never shadows the JSON API routes or ``/docs``.

Nothing is persisted to disk; results carry a TTL via the job store (README §1).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from config import Settings, get_settings
from inference import InferenceClient, InferenceError, build_inference_client
from jobs import JobStore, build_job_store, new_job_id
from ocr import OCRClient, build_ocr_client
from pipeline import verify_label
from schemas import ApplicationData, ItemStatus, JobStatus, VerificationResult

logger = logging.getLogger("label_verification.app")


# --------------------------------------------------------------------------- #
# Lifespan: build the shared dependencies once, tear them down on shutdown.
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build settings + the shared clients/store (and an ``arq`` pool when Redis
    is configured) onto ``app.state``, and release them all on shutdown."""
    settings = get_settings()
    app.state.settings = settings
    app.state.inference_client = build_inference_client(settings)
    app.state.ocr_client = build_ocr_client(settings)
    app.state.job_store = build_job_store(settings)
    app.state.background_tasks = set()
    app.state.arq_pool = None

    if settings.use_redis:
        # The batch lane enqueues onto the same arq queue the worker pool drains.
        from arq import create_pool
        from arq.connections import RedisSettings

        app.state.arq_pool = await create_pool(
            RedisSettings.from_dsn(settings.redis_url)
        )

    try:
        yield
    finally:
        # Let any in-flight in-process batch tasks finish so their results land
        # in the store before we close it.
        pending = [t for t in app.state.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if app.state.arq_pool is not None:
            await app.state.arq_pool.aclose()
        await app.state.inference_client.aclose()
        await app.state.ocr_client.aclose()
        await app.state.job_store.aclose()


app = FastAPI(
    title="AI Label Verification Prototype",
    description=(
        "Verifies alcohol-beverage label artwork against COLA application data "
        "(TTB label compliance). The model extracts; deterministic code judges."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Dependency accessors (typed views onto app.state)
# --------------------------------------------------------------------------- #


def _settings() -> Settings:
    return app.state.settings


def _inference_client() -> InferenceClient:
    return app.state.inference_client


def _ocr_client() -> OCRClient:
    return app.state.ocr_client


def _job_store() -> JobStore:
    return app.state.job_store


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _read_upload(image: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, rejecting anything over ``max_bytes`` with 413.

    Checks the declared size first (cheap), then reads in bounded chunks and
    aborts the instant the accumulated body exceeds the limit — so an oversized
    POST can't force full buffering (memory/disk) before the rejection.
    """
    too_big = HTTPException(
        status_code=413,
        detail=f"image {image.filename!r} exceeds the {max_bytes}-byte limit",
    )
    if image.size is not None and image.size > max_bytes:
        raise too_big
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await image.read(1024 * 1024)  # 1 MiB
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise too_big
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_application(raw: str) -> ApplicationData:
    """Validate the ``application`` form field (JSON of ``ApplicationData``)."""
    try:
        return ApplicationData.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid application JSON: {exc.errors()}"
        ) from exc


def _parse_manifest(raw: str) -> dict[str, Any]:
    """Parse the batch ``manifest`` form field into a JSON object."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise HTTPException(
            status_code=422,
            detail="manifest must be a JSON object (per-filename map or a single "
            "field set applied to all)",
        )
    return obj


def _application_for(manifest: dict[str, Any], filename: str) -> ApplicationData:
    """Resolve the application values for one batch item.

    The manifest is either a per-filename map (``{filename: {field..}}``) or a
    single field set (``{field..}``) applied to every image. The per-filename
    shape is detected by whether *any* value is itself an object. In that case a
    filename with no entry yields an empty ``ApplicationData`` (nothing to
    check) rather than silently reinterpreting the whole map as one field set —
    which would mask a mismatched filename by checking the item against nothing.
    """
    entry = manifest.get(filename)
    if isinstance(entry, dict):
        payload: dict[str, Any] = entry
    elif any(isinstance(v, dict) for v in manifest.values()):
        # Per-filename map, but this file isn't listed: no expected values.
        payload = {}
    else:
        payload = manifest
    try:
        return ApplicationData.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"invalid manifest entry for {filename!r}: {exc.errors()}",
        ) from exc


# --------------------------------------------------------------------------- #
# In-process batch executor (no Redis): one asyncio task per item.
# --------------------------------------------------------------------------- #


async def _run_item_inprocess(
    job_id: str,
    index: int,
    image_bytes: bytes,
    application: ApplicationData,
) -> None:
    """Run one batch item in-process and write its outcome into the store.

    Mirrors the worker's flow without arq: mark ``processing``, verify, then
    record ``done`` (with the result) or ``error`` (with the message). A bad
    photo resolves to ``needs_review`` inside the result, not to an item error —
    one bad image never fails the surrounding job (README §9).
    """
    store = _job_store()
    try:
        await store.set_item(job_id, index, status="processing")
        result: VerificationResult = await verify_label(
            image_bytes,
            application,
            inference_client=_inference_client(),
            ocr_client=_ocr_client(),
            settings=_settings(),
        )
        await store.set_item(job_id, index, status="done", result=result)
    except Exception as exc:  # noqa: BLE001 - record any failure as an item error
        logger.exception("in-process batch item %s/%s failed", job_id, index)
        await store.set_item(job_id, index, status="error", error=str(exc))


def _spawn_inprocess(
    job_id: str,
    index: int,
    image_bytes: bytes,
    application: ApplicationData,
) -> None:
    """Fire-and-track a background task for one in-process batch item.

    The task is held in ``app.state.background_tasks`` so it isn't garbage
    collected mid-flight and so shutdown can await any still running.
    """
    task = asyncio.create_task(
        _run_item_inprocess(job_id, index, image_bytes, application)
    )
    tasks: set[asyncio.Task] = app.state.background_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness + the active backend wiring (for the UI and ops checks)."""
    settings = _settings()
    return {
        "status": "ok",
        "inference_backend": settings.inference_backend,
        "ocr_backend": settings.ocr_backend,
        "use_redis": settings.use_redis,
        "canonical_version": settings.warning_text_version,
    }


@app.post("/verify", response_model=VerificationResult)
async def verify(
    image: UploadFile = File(...),
    application: str = Form(...),
) -> VerificationResult:
    """Interactive lane: verify a single label synchronously (README §4).

    Multipart ``image`` (the artwork) + ``application`` (a JSON
    ``ApplicationData`` of expected values). Returns the full
    ``VerificationResult`` in the same response.
    """
    settings = _settings()
    image_bytes = await _read_upload(image, settings.max_upload_bytes)
    app_data = _parse_application(application)
    try:
        return await verify_label(
            image_bytes,
            app_data,
            inference_client=_inference_client(),
            ocr_client=_ocr_client(),
            settings=settings,
        )
    except InferenceError as exc:
        # The self-hosted model is unreachable/erroring — surface a clean 502
        # (never a stack trace) to the non-technical agent on the other end.
        raise HTTPException(
            status_code=502, detail=f"inference backend unavailable: {exc}"
        ) from exc


@app.post("/jobs", status_code=202)
async def create_job(
    images: list[UploadFile] = File(...),
    manifest: str = Form(...),
) -> dict[str, str]:
    """Batch lane: accept many labels + a manifest, fan out, return a job id.

    The manifest is either a per-filename map (``{filename: {field..}}``) or a
    single field set applied to all images. With Redis configured, each item is
    enqueued as an ``arq`` ``process_label_item`` task on the shared queue; with
    no Redis, each item runs as an in-process background task. Returns ``202``
    with the ``job_id`` to poll at ``GET /jobs/{job_id}``.
    """
    settings = _settings()
    store = _job_store()
    if not images:
        raise HTTPException(status_code=422, detail="no images supplied")
    if len(images) > settings.max_batch_images:
        raise HTTPException(
            status_code=413,
            detail=(
                f"too many images: {len(images)} exceeds the "
                f"{settings.max_batch_images}-image batch limit"
            ),
        )
    manifest_obj = _parse_manifest(manifest)

    # Read + validate every item up front so a bad payload is rejected before the
    # job is created (rather than surfacing as a per-item error mid-batch).
    prepared: list[tuple[str, bytes, ApplicationData]] = []
    items: list[ItemStatus] = []
    for index, image in enumerate(images):
        filename = image.filename or f"image_{index}"
        image_bytes = await _read_upload(image, settings.max_upload_bytes)
        app_data = _application_for(manifest_obj, filename)
        prepared.append((filename, image_bytes, app_data))
        items.append(ItemStatus(index=index, filename=filename))

    job_id = new_job_id()
    await store.create(job_id, items)

    if settings.use_redis:
        pool = app.state.arq_pool
        for index, (filename, image_bytes, app_data) in enumerate(prepared):
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            try:
                await pool.enqueue_job(
                    "process_label_item",
                    job_id,
                    index,
                    filename,
                    image_b64,
                    app_data.model_dump(),
                    _queue_name=settings.arq_queue_name,
                )
            except Exception as exc:  # noqa: BLE001 - never leave an item un-enqueued
                # A Redis hiccup mid-loop must not leave items stuck 'pending' so
                # the job never completes — mark the un-enqueued item 'error' so
                # the job still reaches a terminal state and the UI stops polling.
                logger.exception("failed to enqueue %s item %s", job_id, index)
                await store.set_item(
                    job_id, index, status="error", error=f"enqueue failed: {exc}"
                )
    else:
        for index, (filename, image_bytes, app_data) in enumerate(prepared):
            _spawn_inprocess(job_id, index, image_bytes, app_data)

    return {"job_id": job_id}


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Batch status + per-item results, or ``404`` if unknown/expired."""
    status = await _job_store().get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return status


# --------------------------------------------------------------------------- #
# Static SPA — mounted LAST so it never shadows /verify, /jobs, /health, /docs.
# --------------------------------------------------------------------------- #

app.mount(
    "/",
    StaticFiles(directory=get_settings().static_dir, html=True),
    name="static",
)
