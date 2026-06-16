"""Job-store tests: the in-memory backend's claim/release idempotency contract
and the per-item rollup. Driven with ``asyncio.run`` (no pytest-asyncio)."""

from __future__ import annotations

import asyncio

from config import get_settings
from jobs import InMemoryJobStore, content_hash, new_job_id
from schemas import ApplicationData, ItemStatus

SETTINGS = get_settings()


def _store() -> InMemoryJobStore:
    return InMemoryJobStore(SETTINGS)


def test_claim_is_exclusive_then_release_allows_reclaim():
    store = _store()

    async def _go():
        h = "deadbeef"
        # First claim wins; a concurrent/second claim loses (treated as duplicate).
        assert await store.claim(h) is True
        assert await store.claim(h) is False
        # Releasing (on failure) lets a retry/resubmission re-claim.
        await store.release(h)
        assert await store.claim(h) is True

    asyncio.run(_go())


def test_content_hash_is_stable_and_field_order_independent():
    img = b"\x89PNG fake bytes"
    a = ApplicationData(brand_name="X", abv="45")
    b = ApplicationData(abv="45", brand_name="X")
    assert content_hash(img, a) == content_hash(img, b)
    assert content_hash(img, a) != content_hash(img + b"!", a)


def test_rollups_complete_when_all_items_terminal():
    store = _store()

    async def _go():
        job_id = new_job_id()
        await store.create(job_id, [ItemStatus(index=0, filename="a"),
                                    ItemStatus(index=1, filename="b")])
        await store.set_item(job_id, 0, status="done")
        snap = await store.get(job_id)
        assert snap.status == "processing"  # one still pending
        await store.set_item(job_id, 1, status="error", error="boom")
        snap = await store.get(job_id)
        assert snap.status == "complete"
        assert snap.completed == 1 and snap.failed == 1
        assert snap.is_complete is True

    asyncio.run(_go())
