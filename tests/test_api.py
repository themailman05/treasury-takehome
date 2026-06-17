"""HTTP-surface tests via FastAPI's TestClient (mock backends, in-process batch).

Covers both lanes from README §4:

  * ``GET /health`` reports the active wiring;
  * ``POST /verify`` runs one label synchronously and returns a PASS verdict for
    a clean (echo-path) label;
  * ``POST /jobs`` fans two images out to the in-process executor and returns
    ``202`` + a ``job_id``; polling ``GET /jobs/{id}`` converges to ``complete``
    with per-item PASS results.

The mock/no-Redis backends are pinned in ``conftest`` (env + ``cache_clear``)
before ``app`` is imported, so the in-process batch executor is exercised.
"""

from __future__ import annotations

import json
import time
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import app

CLEAN_APP = {
    "brand_name": "OLD TOM DISTILLERY",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "abv": "45% Alc./Vol. (90 Proof)",
    "net_contents": "750 mL",
}


def _png_bytes(color=(255, 255, 255)) -> bytes:
    """A tiny valid PNG (not UTF-8 JSON) -> mock echo path -> clean PASS label."""
    buf = BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["inference_backend"] == "mock"
    assert body["ocr_backend"] == "mock"
    assert body["use_redis"] is False
    assert body["canonical_version"]


def test_verify_single_label_passes():
    with TestClient(app) as client:
        resp = client.post(
            "/verify",
            files={"image": ("label.png", _png_bytes(), "image/png")},
            data={"application": json.dumps(CLEAN_APP)},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == "pass"
    assert body["warning"]["verdict"] == "pass"
    assert set(body["fields"]) == {"brand_name", "class_type", "abv", "net_contents"}
    assert all(fv["verdict"] == "pass" for fv in body["fields"].values())


def test_batch_job_lifecycle():
    manifest = json.dumps(CLEAN_APP)  # one field set applied to every image
    with TestClient(app) as client:
        resp = client.post(
            "/jobs",
            files=[
                ("images", ("a.png", _png_bytes((250, 250, 250)), "image/png")),
                ("images", ("b.png", _png_bytes((240, 240, 240)), "image/png")),
            ],
            data={"manifest": manifest},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert job_id

        # Poll the in-process executor until the job completes (short bounded loop).
        body = None
        for _ in range(50):
            poll = client.get(f"/jobs/{job_id}")
            assert poll.status_code == 200
            body = poll.json()
            if body["status"] == "complete":
                break
            time.sleep(0.05)

    assert body is not None
    assert body["status"] == "complete"
    assert body["total"] == 2
    assert body["completed"] == 2
    assert body["failed"] == 0
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert item["status"] == "done"
        assert item["result"]["overall"] == "pass"


def test_unknown_job_returns_404():
    with TestClient(app) as client:
        resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_verify_degrades_to_needs_review_on_inference_error():
    # A model/backend failure (or truncated/unparseable output on a hard photo)
    # degrades to needs_review:unreadable — a clean 200, never a 5xx — so one bad
    # label never errors the agent (README §9). The pipeline catches InferenceError.
    from inference import InferenceError

    class _FailingClient:
        async def extract(self, image_bytes, *, application=None):
            raise InferenceError("model output did not match the extraction schema")

        async def aclose(self):
            return None

    with TestClient(app) as client:
        app.state.inference_client = _FailingClient()  # override the mock
        resp = client.post(
            "/verify",
            files={"image": ("label.png", _png_bytes(), "image/png")},
            data={"application": json.dumps(CLEAN_APP)},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == "needs_review"
    assert "extraction_failed" in body["review_flags"]
