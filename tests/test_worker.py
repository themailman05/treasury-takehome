"""Worker idempotency tests (in-memory store + mock clients, no Redis/arq).

Exercises the claim/release flow of ``process_label_item`` directly:
  * a redelivery of an already-succeeded item must NOT clobber its verdict;
  * a different slot carrying identical content IS recorded as a duplicate.
Driven with ``asyncio.run`` (no pytest-asyncio)."""

from __future__ import annotations

import asyncio
import base64

from config import get_settings
from inference import MockInferenceClient
from jobs import InMemoryJobStore, new_job_id
from ocr import MockOCR
from schemas import ItemStatus
from worker import process_label_item

SETTINGS = get_settings()

CLEAN_APP = {
    "brand_name": "OLD TOM DISTILLERY",
    "abv": "45% Alc./Vol. (90 Proof)",
    "net_contents": "750 mL",
}
# Non-JSON bytes -> MockInferenceClient echo path -> a clean PASS verdict.
IMAGE_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n not-json image").decode("ascii")


def _ctx(store: InMemoryJobStore) -> dict:
    return {
        "settings": SETTINGS,
        "store": store,
        "inference_client": MockInferenceClient(SETTINGS),
        "ocr_client": MockOCR(SETTINGS),
        "job_try": 1,
    }


def test_redelivery_after_success_preserves_result():
    store = InMemoryJobStore(SETTINGS)

    async def _go():
        job_id = new_job_id()
        await store.create(job_id, [ItemStatus(index=0, filename="a.png")])
        ctx = _ctx(store)

        # First delivery records a PASS verdict.
        await process_label_item(ctx, job_id, 0, "a.png", IMAGE_B64, CLEAN_APP)
        snap = await store.get(job_id)
        assert snap.items[0].status == "done"
        assert snap.items[0].result is not None
        assert snap.items[0].result.overall.value == "pass"

        # Redelivery of the SAME completed message: claim fails (kept on success),
        # and the recorded verdict must be left intact — never wiped to duplicate/None.
        await process_label_item(ctx, job_id, 0, "a.png", IMAGE_B64, CLEAN_APP)
        snap = await store.get(job_id)
        assert snap.items[0].status == "done"
        assert snap.items[0].result is not None
        assert snap.items[0].result.overall.value == "pass"

    asyncio.run(_go())


def test_cross_item_duplicate_is_marked_duplicate():
    # A different slot with identical content (same hash) is a genuine duplicate.
    store = InMemoryJobStore(SETTINGS)

    async def _go():
        job_id = new_job_id()
        await store.create(
            job_id,
            [ItemStatus(index=0, filename="a.png"), ItemStatus(index=1, filename="b.png")],
        )
        ctx = _ctx(store)
        await process_label_item(ctx, job_id, 0, "a.png", IMAGE_B64, CLEAN_APP)
        await process_label_item(ctx, job_id, 1, "b.png", IMAGE_B64, CLEAN_APP)
        snap = await store.get(job_id)
        assert snap.items[0].status == "done"
        assert snap.items[1].status == "duplicate"
        assert snap.status == "complete"

    asyncio.run(_go())
