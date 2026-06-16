"""Batch-job state: the ephemeral results store behind the throughput lane.

A ``/jobs`` POST fans one label out per item; workers (or, with no Redis, in-process
tasks) report progress back into a :class:`JobStore`. The store also carries the
idempotency ledger — a content hash per item — so an at-least-once Redis redelivery
or an ``arq`` retry never double-processes the same label (README §9).

Two backends, one interface:

  * :class:`InMemoryJobStore` — a dict guarded by an ``asyncio.Lock``; correct under
    concurrent ``set_item`` calls and zero-setup for a laptop/CI run.
  * :class:`RedisJobStore` — ``redis.asyncio``; the ``JobStatus`` is stored as JSON
    under ``job:{id}`` with a TTL, per-item updates are an optimistic
    read-modify-write under ``WATCH``, and idempotency is a ``SET nx`` on the hash.

Nothing is persisted to disk and every key carries a TTL — the prototype stores no
sensitive data (README §1).
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import time
import uuid
from typing import Optional

from config import Settings, get_settings
from schemas import ApplicationData, ItemStatus, JobStatus, VerificationResult

# --------------------------------------------------------------------------- #
# Idempotency + identifiers
# --------------------------------------------------------------------------- #


def content_hash(image_bytes: bytes, application: ApplicationData) -> str:
    """Stable SHA-256 over the image bytes + the application values.

    Used as the idempotency key: the same image with the same expected fields
    hashes identically, so a redelivered or retried batch message is recognised
    and skipped rather than verified twice.
    """
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(b"\x00")
    # Sorted keys -> field ordering can't change the hash for the same payload.
    app_json = json.dumps(
        application.model_dump(), sort_keys=True, separators=(",", ":")
    )
    digest.update(app_json.encode("utf-8"))
    return digest.hexdigest()


def new_job_id() -> str:
    """A fresh, URL-safe batch identifier."""
    return uuid.uuid4().hex


def _recompute(status: JobStatus) -> None:
    """Recompute the derived ``completed``/``failed``/``status`` rollups in place.

    Counts are recomputed from the items on every update so the rollup can never
    drift from the per-item truth. ``duplicate`` items are terminal and count as
    completed (they were processed earlier).
    """
    completed = sum(1 for it in status.items if it.status in ("done", "duplicate"))
    failed = sum(1 for it in status.items if it.status == "error")
    status.completed = completed
    status.failed = failed
    if status.total > 0 and (completed + failed) >= status.total:
        status.status = "complete"
    elif any(it.status == "processing" for it in status.items) or completed or failed:
        status.status = "processing"
    else:
        status.status = "queued"


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class JobStore(abc.ABC):
    """Async store for batch-job progress + per-item results, plus the
    idempotency ledger. Both backends honour this contract."""

    @abc.abstractmethod
    async def create(self, job_id: str, items: list[ItemStatus]) -> None:
        """Register a new job with its (initially pending) item list."""

    @abc.abstractmethod
    async def get(self, job_id: str) -> Optional[JobStatus]:
        """Return the current job snapshot, or ``None`` if unknown/expired."""

    @abc.abstractmethod
    async def set_item(
        self,
        job_id: str,
        index: int,
        *,
        status: str,
        result: Optional[VerificationResult] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update one item and recompute the job rollups atomically."""

    @abc.abstractmethod
    async def claim(self, content_hash: str) -> bool:
        """Atomically claim a content hash for processing.

        Returns ``True`` if this caller won the claim (and should process the
        item), ``False`` if the hash is already claimed or processed — a
        concurrent redelivery or an earlier run owns it, so treat it as a
        duplicate. The claim is held until :meth:`release` (on failure, so a
        retry can re-claim) or its TTL; a *successful* run keeps the claim so a
        later redelivery is still recognised as a duplicate (README §9).
        """

    @abc.abstractmethod
    async def release(self, content_hash: str) -> None:
        """Release a won claim so a failed item can be retried/resubmitted."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release any backend resources (connections, pools)."""


# --------------------------------------------------------------------------- #
# In-memory backend (laptop / CI; no Redis)
# --------------------------------------------------------------------------- #


class InMemoryJobStore(JobStore):
    """Process-local store guarded by a single ``asyncio.Lock``.

    Every mutation runs under the lock, so concurrent ``set_item`` calls (the
    in-process batch executor fans out one task per item) serialise cleanly and
    the rollups stay consistent. The idempotency ledger carries a best-effort TTL
    so it can't grow without bound during a long-lived process.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._ttl = self._settings.result_ttl_seconds
        self._lock = asyncio.Lock()
        # job_id -> (expiry_epoch, JobStatus)
        self._jobs: dict[str, tuple[float, JobStatus]] = {}
        # content_hash -> expiry_epoch
        self._processed: dict[str, float] = {}

    def _purge_expired(self, now: float) -> None:
        """Drop expired jobs and processed hashes. Caller holds the lock."""
        expired_jobs = [jid for jid, (exp, _) in self._jobs.items() if exp <= now]
        for jid in expired_jobs:
            del self._jobs[jid]
        expired_hashes = [h for h, exp in self._processed.items() if exp <= now]
        for h in expired_hashes:
            del self._processed[h]

    async def create(self, job_id: str, items: list[ItemStatus]) -> None:
        now = time.time()
        status = JobStatus(
            job_id=job_id,
            total=len(items),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            items=list(items),
        )
        _recompute(status)
        async with self._lock:
            self._purge_expired(now)
            self._jobs[job_id] = (now + self._ttl, status)

    async def get(self, job_id: str) -> Optional[JobStatus]:
        now = time.time()
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry[0] <= now:
                self._jobs.pop(job_id, None)
                return None
            # Deep copy so callers can't mutate stored state out from under us.
            return entry[1].model_copy(deep=True)

    async def set_item(
        self,
        job_id: str,
        index: int,
        *,
        status: str,
        result: Optional[VerificationResult] = None,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            expiry, job = entry
            if not 0 <= index < len(job.items):
                return
            item = job.items[index]
            item.status = status
            item.result = result
            item.error = error
            _recompute(job)
            # Refresh the TTL on activity so an in-flight job doesn't expire.
            self._jobs[job_id] = (time.time() + self._ttl, job)

    async def claim(self, content_hash: str) -> bool:
        now = time.time()
        async with self._lock:
            exp = self._processed.get(content_hash)
            if exp is not None and exp > now:
                return False  # already claimed/processed and unexpired
            self._processed[content_hash] = now + self._ttl
            return True

    async def release(self, content_hash: str) -> None:
        async with self._lock:
            self._processed.pop(content_hash, None)

    async def aclose(self) -> None:
        async with self._lock:
            self._jobs.clear()
            self._processed.clear()


# --------------------------------------------------------------------------- #
# Redis backend (shared across API + worker processes)
# --------------------------------------------------------------------------- #


class RedisJobStore(JobStore):
    """``redis.asyncio``-backed store shared by the API and the worker pool.

    The whole ``JobStatus`` lives as JSON under ``job:{id}`` with a TTL; a per-item
    update is an optimistic read-modify-write under ``WATCH`` (retried on a
    concurrent change) so two workers finishing at once can't clobber each other's
    rollups. Idempotency is a single ``SET nx`` against ``hash:{h}``.
    """

    _MAX_TXN_RETRIES = 32

    def __init__(self, settings: Optional[Settings] = None) -> None:
        import redis.asyncio as aioredis  # local import: redis only needed here

        self._settings = settings or get_settings()
        self._ttl = self._settings.result_ttl_seconds
        self._redis = aioredis.from_url(
            self._settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def _hash_key(content_hash: str) -> str:
        return f"hash:{content_hash}"

    async def create(self, job_id: str, items: list[ItemStatus]) -> None:
        status = JobStatus(
            job_id=job_id,
            total=len(items),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            items=list(items),
        )
        _recompute(status)
        await self._redis.set(
            self._job_key(job_id), status.model_dump_json(), ex=self._ttl
        )

    async def get(self, job_id: str) -> Optional[JobStatus]:
        raw = await self._redis.get(self._job_key(job_id))
        if raw is None:
            return None
        return JobStatus.model_validate_json(raw)

    async def set_item(
        self,
        job_id: str,
        index: int,
        *,
        status: str,
        result: Optional[VerificationResult] = None,
        error: Optional[str] = None,
    ) -> None:
        key = self._job_key(job_id)
        for _ in range(self._MAX_TXN_RETRIES):
            async with self._redis.pipeline() as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if raw is None:
                        await pipe.unwatch()
                        return
                    job = JobStatus.model_validate_json(raw)
                    if not 0 <= index < len(job.items):
                        await pipe.unwatch()
                        return
                    item = job.items[index]
                    item.status = status
                    item.result = result
                    item.error = error
                    _recompute(job)
                    pipe.multi()
                    pipe.set(key, job.model_dump_json(), ex=self._ttl)
                    await pipe.execute()
                    return
                except _watch_error():
                    # Another writer touched the job between WATCH and EXEC; retry.
                    continue
        raise RuntimeError(
            f"set_item for job {job_id!r} item {index} contended past "
            f"{self._MAX_TXN_RETRIES} retries"
        )

    async def claim(self, content_hash: str) -> bool:
        # SET nx returns truthy only for the first caller to create the key, so
        # the claim is atomic across the API and every worker sharing this Redis
        # — a concurrent at-least-once redelivery can't double-process the label.
        won = await self._redis.set(
            self._hash_key(content_hash), "1", nx=True, ex=self._ttl
        )
        return bool(won)

    async def release(self, content_hash: str) -> None:
        await self._redis.delete(self._hash_key(content_hash))

    async def aclose(self) -> None:
        await self._redis.aclose()


def _watch_error() -> type[BaseException]:
    """The ``WatchError`` raised on a failed optimistic transaction.

    Imported lazily so the module imports cleanly without ``redis`` installed
    (the in-memory backend has no Redis dependency)."""
    from redis.exceptions import WatchError

    return WatchError


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def build_job_store(settings: Settings) -> JobStore:
    """Pick the store backend: Redis when configured, in-memory otherwise."""
    if settings.use_redis:
        return RedisJobStore(settings)
    return InMemoryJobStore(settings)
