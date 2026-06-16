"""The batch lane's engine room: the ``arq`` worker that drains the queue.

``POST /jobs`` fans one message out per label onto the ``arq`` queue; this module
is what consumes them. Each task runs the very same ``verify_label`` pipeline the
interactive lane uses — extract, OCR-fill, verify — and writes its per-item verdict
back into the shared :class:`JobStore` so the UI can poll progress (README §4).

Three operational guarantees the brief calls out (README §9) live here:

  * **Idempotency.** Every item is keyed by a content hash. ``arq`` (and a Redis
    stream in general) is at-least-once, so before doing any work we *atomically
    claim* the hash via ``store.claim`` — if we don't win the claim a concurrent
    redelivery or an earlier successful run owns it, and the item is recorded as a
    ``duplicate`` and skipped, never verified twice. A successful run keeps the
    claim; a failed one releases it so a retry/resubmission can run.
  * **Retries.** A transient failure (a momentary model blip) raises ``Retry()``
    so ``arq`` re-runs the task, up to ``max_batch_retries`` attempts. (A *plain*
    re-raise would not be retried by ``arq`` — it only retries ``Retry``.)
    ``ctx['job_try']`` is the 1-based attempt counter ``arq`` maintains.
  * **Dead-lettering.** Once retries are exhausted the item is marked ``error`` and
    pushed onto a ``dlq`` list rather than re-raising forever — one poisoned label
    can never stall a 300-item job.

The shared clients (inference, OCR, job store) are built once in ``startup`` and
closed in ``shutdown`` so every task reuses one set of connection pools.
"""

from __future__ import annotations

import base64
from typing import Any

from arq.connections import RedisSettings
from arq.worker import Retry

from config import get_settings
from inference import build_inference_client
from jobs import build_job_store, content_hash
from ocr import build_ocr_client
from pipeline import verify_label
from schemas import ApplicationData

# Redis list that exhausted (poison) items are pushed onto for later inspection.
DEAD_LETTER_KEY = "dlq"


async def startup(ctx: dict[str, Any]) -> None:
    """Build the shared inference/OCR clients and job store once per worker.

    Stashing them on ``ctx`` means every ``process_label_item`` invocation reuses
    the same HTTP/Redis connection pools instead of reconnecting per label.
    """
    settings = get_settings()
    ctx["settings"] = settings
    ctx["inference_client"] = build_inference_client(settings)
    ctx["ocr_client"] = build_ocr_client(settings)
    ctx["store"] = build_job_store(settings)


async def shutdown(ctx: dict[str, Any]) -> None:
    """Release everything ``startup`` built. Best-effort: never mask a failure."""
    for key in ("inference_client", "ocr_client", "store"):
        client = ctx.get(key)
        if client is not None:
            await client.aclose()


async def _dead_letter(ctx: dict[str, Any], payload: dict[str, Any]) -> None:
    """Push an exhausted item onto the Redis ``dlq`` list, best-effort.

    ``arq`` exposes its Redis connection on ``ctx['redis']``; if it's somehow
    absent (e.g. an in-process test harness) we silently skip — the item is still
    recorded as ``error`` on the job, which is the user-visible outcome.
    """
    import json

    redis = ctx.get("redis")
    if redis is None:
        return
    await redis.rpush(DEAD_LETTER_KEY, json.dumps(payload, separators=(",", ":")))


async def process_label_item(
    ctx: dict[str, Any],
    job_id: str,
    index: int,
    filename: str,
    image_b64: str,
    application: dict,
) -> None:
    """Verify one batch item and record its verdict on the job.

    Parameters
    ----------
    job_id, index, filename : locate this item within its batch job.
    image_b64 : the label artwork, base64-encoded for JSON transport over Redis.
    application : the applicant's expected field values (an ``ApplicationData`` dict).

    Flow: decode, idempotency-check (skip as ``duplicate`` if seen), mark
    ``processing``, run ``verify_label``, mark ``done`` with the result. A failure
    re-raises for ``arq`` to retry until ``max_batch_retries`` attempts are spent,
    after which the item is marked ``error`` and dead-lettered.
    """
    settings = ctx["settings"]
    store = ctx["store"]

    image_bytes = base64.b64decode(image_b64)
    app = ApplicationData(**application)
    item_hash = content_hash(image_bytes, app)

    # Idempotency: atomically claim the content hash. If we don't win the claim,
    # a concurrent at-least-once redelivery or an earlier successful run owns this
    # label, so record it as a duplicate and skip — never verified twice.
    if not await store.claim(item_hash):
        # Only annotate a still-open slot. If THIS (job_id, index) slot already
        # holds an outcome (a redelivery of the same completed message) or is being
        # processed by another delivery, leave it intact — set_item would otherwise
        # overwrite a finished verdict with status='duplicate', result=None.
        snap = await store.get(job_id)
        current = snap.items[index] if snap and 0 <= index < len(snap.items) else None
        if current is not None and current.status == "pending":
            await store.set_item(job_id, index, status="duplicate")
        return

    await store.set_item(job_id, index, status="processing")

    try:
        result = await verify_label(
            image_bytes,
            app,
            inference_client=ctx["inference_client"],
            ocr_client=ctx["ocr_client"],
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - any failure follows the retry policy
        # Release the claim so a retry (or a later resubmission) can re-claim and
        # run — only the success path below keeps the claim.
        await store.release(item_hash)
        # ``job_try`` is arq's 1-based attempt counter. While retries remain, raise
        # Retry() so arq re-enqueues the item (a plain raise is NOT retried by arq).
        # Once exhausted, fail the item gracefully and dead-letter it so one poison
        # label can never stall the job (README §9).
        job_try = ctx.get("job_try", 1)
        if job_try <= settings.max_batch_retries:
            raise Retry() from exc
        await store.set_item(job_id, index, status="error", error=str(exc))
        await _dead_letter(
            ctx,
            {
                "job_id": job_id,
                "index": index,
                "filename": filename,
                "error": str(exc),
            },
        )
        return

    # Success keeps the claim, so a later redelivery is recognised as a duplicate.
    await store.set_item(job_id, index, status="done", result=result)


class WorkerSettings:
    """``arq`` worker configuration (the class ``arq worker`` discovers).

    ``max_tries`` is ``max_batch_retries + 1`` so the attempt budget matches the
    retry policy in ``process_label_item`` (N retries == N+1 total attempts); the
    final attempt is the one that dead-letters instead of re-raising.
    """

    functions = [process_label_item]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url or "redis://localhost")
    queue_name = get_settings().arq_queue_name
    on_startup = startup
    on_shutdown = shutdown
    max_tries = get_settings().max_batch_retries + 1
