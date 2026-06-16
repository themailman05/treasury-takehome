# AI Label Verification Prototype

> A proof-of-concept that verifies alcohol-beverage label artwork against COLA application data. Built for the TTB Label Compliance discovery brief: extract the fields from a label image, compare them to what the applicant submitted, and return a clear pass / fail / needs-review verdict per field — fast enough that an agent will actually use it, and self-hosted so it survives a federal firewall.

**Live demo:** https://treasury.liam.cool — self-hosted Gemma 4 on a GPU (no outbound ML traffic). See §11 for the full deployment runbook.

---

## Quickstart

**1. Try the live demo (nothing to install).** Open **https://treasury.liam.cool**, upload a label from [`samples/labels/`](./samples/labels) — each is mapped to its expected verdict and the values to enter in that folder's [README](./samples/labels/README.md) — optionally type the application values, and click **Verify**. The result shows the label with a colour-coded box per field.

**2. Run locally — no GPU, no model** (deterministic mock backend; good for code review + tests):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app                 # UI: http://localhost:8000   ·   API docs: /docs
pytest -q                       # 49 tests
```

**3. Run the full self-hosted stack** (Docker; Gemma 4 on **CPU** via Ollama + Redis + worker — no GPU needed):
```bash
cp .env.example .env
docker compose up --build       # http://localhost:8000  (first boot pulls the model; allow Docker ≥12 GB RAM)
```
For the **GPU** production target (vLLM) or the exact runbook behind the live demo, see **§11**.

**Quick API check:**
```bash
curl -F "image=@samples/labels/L000_gin_abv_mismatch.png" \
     -F 'application={"brand_name":"Wandering Provisions","abv":"44","net_contents":"1 L"}' \
     http://localhost:8000/verify
```

> Backends are config-driven (`.env`): `INFERENCE_BACKEND=mock|ollama|vllm`, `OCR_BACKEND=mock|tesseract`. Local defaults to `mock`; Compose uses `ollama` + `tesseract`.

---

## 1. The problem

TTB reviews ~150,000 label applications a year with ~47 agents. A large share of that work is mechanical matching: does the brand name on the artwork match the application, is the ABV right, is the mandatory government health warning present and correctly formatted. This tool automates the mechanical pass so agents spend their judgment where judgment is needed.

It is a **standalone prototype**, not a COLA integration. It stores nothing sensitive and makes no outbound calls to third-party ML endpoints.

## 2. Stakeholder requirements

Requirements below are derived from the discovery interviews, mapped to the design choices that satisfy them.

| Source | Requirement | How it's met |
|---|---|---|
| Sarah (Deputy Director) | Results in **≤ ~5 seconds** for a single label — the prior vendor's 30–40s killed adoption | Synchronous "interactive lane" with priority scheduling and small constrained outputs |
| Sarah | Usable by non-technical agents (half the team is 50+) | Minimal UI: upload, see a checklist, done. No hunting for buttons |
| Sarah | **Batch upload** — importers drop 200–300 applications at once | Async "batch lane" backed by a Redis stream + worker pool |
| Marcus (IT) | **No COLA integration** — standalone POC only | Self-contained service; no COLA auth or coupling |
| Marcus | Firewall blocks outbound to many ML endpoints | Model is **self-hosted on a GPU inside the boundary** — no external API calls |
| Marcus | No sensitive data stored for the exercise | Images held ephemerally; results carry a TTL; nothing persisted to disk |
| Dave (Senior Agent) | Don't naively string-match — "STONE'S THROW" vs "Stone's Throw" is the same brand | Normalized + fuzzy matching with a review threshold, not exact equality |
| Jenny (Junior Agent) | Government warning must be **exact**, with "GOVERNMENT WARNING:" in caps and bold | Verbatim transcription + deterministic exact match + caps check |
| Jenny | Handle imperfect photos (angle, glare, lighting) | Vision-language model tolerates real-world images; unreadable cases route to review, not failure |

## 3. Scope

In scope for the prototype:

- Single-label interactive verification (the latency-sensitive path).
- Batch verification of many labels in one job.
- Field checks: brand name, class/type, alcohol content, net contents, and the government health warning.
- Per-field verdicts: `pass`, `fail`, `needs_review`.

Explicitly out of scope (documented as limitations, see §9): COLA write-back, persistent storage, user accounts, and physical type-size measurement.

## 4. Architecture

Two request lanes converge on one shared GPU inference service and one shared deterministic verifier.

```
                         ┌──────────────┐
                         │   Agent UI   │   upload 1 or many
                         └──────┬───────┘
                  ┌─────────────┴──────────────┐
                  ▼                             ▼
        ┌───────────────────┐        ┌────────────────────┐
        │  Interactive lane │        │     Batch lane      │
        │  sync, size 1     │        │  Redis stream, async│
        └─────────┬─────────┘        └──────────┬──────────┘
                  └──────────────┬──────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │     Gemma 4 on GPU      │   vLLM continuous batching
                    │  (extract → JSON)       │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │     Verification        │   fuzzy match + exact warning
                    │  (deterministic, Python)│
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │        Result           │   checklist / job results
                    └─────────────────────────┘
```

### Interactive lane (the SLA path)
The agent opens one label; the API server sends it to the inference service as a synchronous request and holds the connection. vLLM's continuous batching slots a high-priority request into the running batch immediately, so a single interactive label never waits behind a 300-item batch. The model returns constrained JSON, it passes through the verifier, and the verdict returns in the same HTTP response — targeting the full round trip under 5 seconds.

### Batch lane (the throughput path)
An importer uploads many labels. The API server creates a `job_id`, fans out one message per label onto a Redis stream, and returns `202 Accepted` immediately. A consumer group of workers drains the stream (dynamic batching here, to maximize GPU utilization), runs the same extract → verify steps, and writes per-item results into a results store keyed by `job_id`. The UI polls (or subscribes via SSE) for progress and pulls the finished set.

### Why self-hosted Gemma 4
A cloud vision API (the obvious move) fails Marcus's firewall constraint and could never ship inside TTB. Running Gemma 4 ourselves turns that constraint into a feature: everything — UI, API, Redis, and the GPU service — sits inside the VPC with no outbound ML traffic. Gemma 4 is Apache-2.0 licensed and strong at OCR and chart/label understanding, with native bounding-box output that helps locate the warning block on a messy photo.

## 5. Verification logic

The single most important design rule: **the model extracts, deterministic code judges.** Vision-language models paraphrase and "helpfully" correct text — exactly wrong for a word-for-word legal requirement. So Gemma transcribes; Python compares.

> **As-built note.** This section is the original design. Building it surfaced two refinements, detailed in §12: (1) the warning comparator uses **fuzzy clause-similarity bands** rather than a per-character edit budget (robust to real OCR noise), and (2) the warning is judged on the **deterministic OCR alone** — testing confirmed the VLM hallucinates the canonical warning even on a defective/absent label, so it can't be trusted to read the warning (it still reads the other fields and the bounding boxes).

### Field matching (brand, class/type, ABV, net contents)
Normalize both sides (case-fold, collapse whitespace, strip punctuation, unify unit formatting like `45% Alc./Vol.` ↔ `45%`), then fuzzy-match with a configurable similarity threshold (`rapidfuzz`). Above the threshold → `pass`. Borderline → `needs_review` rather than auto-reject — this is Dave's "STONE'S THROW" case. A clear mismatch → `fail`.

### Government warning (strict)
This is federal regulatory text (27 CFR 16.21). The canonical reference string the verifier matches against:

```
GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems.
```

"Exact match" here does **not** mean string equality, and the warning is the one field that must **not** rely on VLM transcription. The warning is famous, fixed, memorized text, which creates two opposing failure modes:

- **VLM false-pass.** A vision-language model asked to transcribe the warning leans on its prior toward the canonical string. Shown a *defective* label (dropped clause, reworded, title-case header) it may emit the *correct* text anyway — and a literal `==` then passes a non-compliant label. Passing a bad warning is the worst outcome this tool can produce.
- **Deterministic-OCR false-fail.** Real OCR is noisy (`rn`→`m`, `O`→`0`, glare). A literal `==` against the canonical string fails compliant labels over recognition artifacts.

So the warning uses a different pipeline from the other fields:

1. **Authoritative reading = deterministic OCR** of the cropped warning region (Tesseract/PaddleOCR). OCR reports what is *literally on the label*, not what should be there.
2. **Clause-level comparison, not blob equality.** Split the canonical into header + clause (1) + clause (2) and match each against the OCR reading with a small edit-distance budget (default 6% of the clause length, calibrated to the OCR error rate). This tolerates OCR noise while still catching a missing or reworded clause.
3. **Casing check** on the literal `GOVERNMENT WARNING` token, verified on the case-preserving OCR output. Title case → fail (Jenny's reject).
4. **Dual-path agreement as a cross-check.** The VLM transcription is compared against the *OCR reading* (never against the canonical). If the two independent readings of the same label disagree, one of them misread → route to review. This single check neutralizes both failure modes above: a hallucinated-canonical VLM disagrees with OCR that read the real defect, and noisy OCR disagrees with a clean VLM read.
5. **Versioned canonical text.** Keep it as a config value, not a literal, so a future rule change is a config bump (a cancer-warning amendment has been under discussion since the Jan 2025 Surgeon General advisory, though the 1988 text above remains the legally required statement).

Verdict policy is conservative — it never auto-passes a defect:

| Situation | Verdict |
|---|---|
| Dual path, readings agree, reading matches canonical | `pass` |
| Dual path, readings agree, reading deviates | `fail` |
| Dual path, readings disagree | `needs_review` |
| Single path (no VLM), matches canonical | `pass` (flagged `single_path`) |
| Single path (no VLM), deviates | `needs_review` |

The comparator is implemented in [`verify_warning.py`](./verify_warning.py); its `__main__` runs smoke tests covering clean, OCR-noise, title-case, missing-clause, reworded, and the VLM-hallucination case.

For reference, the minimum type-size rule keyed to container volume (27 CFR 16.22) — see §9 for why this is not auto-checked:

| Container size | Minimum type size |
|---|---|
| ≤ 237 ml (8 fl oz) | 1 mm |
| > 8 fl oz and ≤ 3 L | 2 mm |
| > 3 L (101 fl oz) | 3 mm |

### What's checkable vs review-only
- **Reliable**: field-value matches (brand, class/type, ABV, net contents) from the VLM, and warning text + header casing from deterministic OCR cross-checked against the VLM reading.
- **Routed to `needs_review`**: bold detection and physical type size in mm — both require information an uncontrolled photo doesn't carry (a known scale reference for size; reliable weight detection for bold) — plus any case where the OCR and VLM readings of the warning disagree. The tool is honest about this rather than faking a verdict.

## 6. Tech stack

Reference implementation (adjust to your actual repo):

- **API**: Python 3.11, FastAPI + Uvicorn
- **Queue & ephemeral store**: Redis 7 (Streams with consumer groups; results as keys with TTL)
- **Workers**: `arq` (async, Redis-native; pairs with FastAPI)
- **Inference**: Gemma 4 (vision-capable, multimodal), self-hosted and OpenAI-compatible. Two interchangeable backends selected by `INFERENCE_BACKEND`:
  - **`ollama` (CPU)** — the default `docker compose up` profile. No GPU required. Default tag `gemma4:e4b` (~9.6 GB, the "cheap deploy"); `gemma4:e2b` for constrained machines, `gemma4:12b` for higher accuracy. ⚠️ Use only **vision-capable** tags — never a `*-mlx` tag (text-only). CPU inference is slow, so the sub-5s SLA is a GPU target, not a CPU one.
  - **`vllm` (GPU)** — the production throughput target (`--profile gpu`): `google/gemma-4-12B` for the extraction task, `gemma-4-31B` for max accuracy, with continuous batching + guided JSON decoding. SGLang is a drop-in alternative.
  - **`mock`** — deterministic, model-free; runs the whole service on a laptop / in CI with no model at all.
- **Verification**: pure Python, `rapidfuzz` for fuzzy matching; deterministic OCR (Tesseract or PaddleOCR) on the cropped warning region
- **Frontend**: lightweight SPA (Vite + React) — upload control, result checklist, batch progress
- **Orchestration**: Docker Compose; GPU via the NVIDIA Container Toolkit

## 7. Getting started

### Prerequisites
- **CPU profile (default):** just Docker + Docker Compose. Allocate Docker enough RAM (≥ 12 GB for `gemma4:e4b`; use `gemma4:e2b` if constrained). No GPU, no Hugging Face token — Ollama pulls the public Gemma 4 weights.
- **GPU profile (`--profile gpu`, production target):** an NVIDIA GPU + the NVIDIA Container Toolkit, and a Hugging Face token for the gated `google/gemma-4-*` weights. VRAM depends on the model (the 31B dense fits a single 80 GB H100 in bf16; 12B/E4B fit much smaller cards).
- **Local / no Docker:** Python 3.11+ only — the `mock` backend runs the whole service with no model.

### Setup
```bash
git clone <your-repo-url>
cd label-verification

cp .env.example .env
# CPU default works out of the box. Optionally set GEMMA_MODEL (gemma4:e4b |
# gemma4:e2b | gemma4:12b), WARNING_TEXT_VERSION, FUZZY_PASS_THRESHOLD,
# RESULT_TTL_SECONDS. HF_TOKEN is only needed for the GPU (vllm) profile.
```

### Run
```bash
# CPU: Gemma 4 on Ollama + Redis + API + worker. No GPU.
docker compose up --build

# GPU production target instead (also starts vLLM):
#   set INFERENCE_BACKEND=vllm and VLLM_BASE_URL=http://vllm:8001/v1 in .env, then:
# docker compose --profile gpu up --build

# Or fully local, no Docker / no model (mock backend, in-process batch):
# pip install -r requirements.txt && uvicorn app:app
```

The CPU stack starts Ollama (pulls the model on first boot — be patient), Redis, the API server, the worker pool, and serves the UI from the API. Once healthy:

- UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Quick test
```bash
# single label (interactive lane)
curl -F "image=@samples/old_tom_bourbon.png" \
     -F 'application={"brand_name":"OLD TOM DISTILLERY","abv":"45","net_contents":"750 mL"}' \
     http://localhost:8000/verify
```

Generate additional test labels with any AI image tool, or photograph real bottles to exercise the imperfect-image path.

## 8. API

### `POST /verify` — interactive, single label
Multipart: `image` (file) + `application` (JSON of expected field values). Returns synchronously.

### `POST /jobs` — batch
Accepts multiple images + a manifest of expected values. Returns `202` with a `job_id`.

### `GET /jobs/{job_id}` — batch status & results
Returns progress and per-item verdicts as they complete.

### Model → verifier contract
The model is prompted to return only this JSON (enforced via guided decoding). The verifier never trusts a model "verdict" — it only consumes extracted text. Note the warning carries **two** readings: the VLM transcription plus the deterministic OCR of the cropped region (the authoritative one). See §5 for why.

```json
{
  "brand_name":   { "text": "OLD TOM DISTILLERY", "confidence": 0.97 },
  "class_type":   { "text": "Kentucky Straight Bourbon Whiskey", "confidence": 0.95 },
  "abv":          { "text": "45% Alc./Vol. (90 Proof)", "confidence": 0.96 },
  "net_contents": { "text": "750 mL", "confidence": 0.98 },
  "warning": {
    "vlm_text":  "GOVERNMENT WARNING: (1) According to the Surgeon General...",
    "ocr_text":  "GOVERNMENT WARNING: (1) According to the Surgeon General...",
    "located":   true,
    "bbox":      [x, y, w, h],
    "confidence": 0.93
  },
  "image_quality": "ok"
}
```

The verifier produces:

```json
{
  "fields": {
    "brand_name":   { "verdict": "pass", "similarity": 1.0 },
    "class_type":   { "verdict": "pass", "similarity": 0.98 },
    "abv":          { "verdict": "pass", "extracted": "45", "expected": "45" },
    "net_contents": { "verdict": "pass" },
    "warning": {
      "verdict": "pass",
      "exact_match": true,
      "caps_ok": true,
      "readings_agree": true,
      "agreement_ratio": 99.1,
      "canonical_version": "1988-ABLA"
    }
  },
  "overall": "pass",
  "review_flags": ["warning_bold_unverified", "warning_typesize_unverified"]
}
```

## 9. Limitations & assumptions

Documented deliberately — the brief rewards naming trade-offs.

- **Bold and physical type size are not auto-verified.** They require information a photo doesn't carry; both surface as review flags instead of fabricated verdicts.
- **No COLA integration, no persistence, no auth.** This is a standalone POC per the brief. A production path would add PII handling, document-retention compliance, and COLA's own authorization requirements.
- **Deployment trade-off.** The "deployed URL" deliverable means an always-on GPU. For the review window, the live demo can run the smaller `gemma-4-E4B` on a modest card (or scale-to-zero with a cold-start notice) while the Compose file documents the full GPU design as the production target.
- **Idempotency & retries.** Each batch message carries a content hash so an at-least-once stream redelivery or worker retry doesn't double-process a label. Repeated failures dead-letter rather than stall a job.
- **Bad-image handling.** Low model confidence or an unlocatable required field resolves to `needs_review: unreadable` — one bad photo never fails an entire 300-item batch.

## 10. Next steps / production path (future work)

**Accuracy & model**
- **Fine-tune the VLM on known/historical labels.** A model tuned on real COLA artwork would read fields and bounding boxes more accurately than the off-the-shelf `gemma4:12b`, and a *distilled* fine-tune could be both accurate **and** fast enough to hit the ≤5s SLA on a modest GPU (today 12B is accurate but ~13s; the smaller `e4b` is faster but regresses field reading — see §12).
- **Infer plausible ABV ranges per beverage type** (e.g. table wine ~5–24%, beer ~3–14%, distilled spirits ~30–95%) and flag an ABV that's out-of-range for the declared class, or a class/ABV mismatch — a sanity check that needs no application value.
- **Better warning OCR** (PaddleOCR, or a recognizer fine-tuned on the warning font) to tighten the similarity bands and turn today's `needs_review` (OCR-unsure) cases into crisp pass/fail.
- **Box-guided per-field re-OCR**: use the model's bounding boxes to crop and re-read each field at high resolution for a second, higher-confidence pass.

**More compliance checks**
- **Bold / font-weight detection** for the `GOVERNMENT WARNING:` header (currently a review flag) via stroke-width analysis.
- **Physical type-size (mm)** per 27 CFR 16.22, using container dimensions as a known scale reference (currently out of scope — a photo carries no absolute scale).
- **Other mandatory elements** the brief lists: name/address of bottler, country of origin for imports, sulfite/allergen declarations, and **standard-of-fill** (net contents must be an authorized size, not just "matches the application").

**Platform**
- COLA write-back and integration behind its auth model.
- Persistent, compliant storage with retention policies and audit logging; authentication / role-based access for agents.
- **Active-learning loop**: feed agent corrections back to improve extraction prompts, thresholds, and the fine-tune set.
- **Confidence-calibrated auto-approve** for the cleanest cases, escalating only ambiguity to humans.
- Content-hash caching to skip re-verifying identical resubmissions; a faster/bigger GPU (L40S/A100) to serve the accurate 12B under the ≤5s SLA.


## 11. Deployment runbook — self-hosted GPU (the live demo)

The live demo runs **entirely self-hosted** on one GCP GPU VM: Gemma 4 on the GPU via Ollama, the FastAPI app, and Caddy for automatic HTTPS — no outbound ML traffic (Marcus's firewall constraint). Reproduce it end-to-end:

**0. Provision.** GCP Compute Engine `g2-standard-4` + 1× **NVIDIA L4 (24 GB)**, **Debian 13**, firewall allowing `tcp:22,80,443`. Point a DNS `A` record (e.g. `treasury.liam.cool`) at the VM's external IP.

**1. NVIDIA driver (Debian 13).**
```bash
# enable non-free, install kernel headers + DKMS + driver, reboot to load the module
sudo sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) linux-headers-cloud-amd64 build-essential dkms
sudo apt-get install -y nvidia-driver firmware-misc-nonfree
sudo reboot
nvidia-smi          # after reboot -> "NVIDIA L4", driver 550.x
```
> Secure Boot is **disabled** on this VM, so the DKMS module loads unsigned. With Secure Boot **on**, the apt/DKMS module won't load — use Google's signed-driver installer instead.

**2. Serve Gemma 4 on the GPU (Ollama).**
```bash
curl -fsSL https://ollama.com/install.sh | sh     # installs + starts a systemd service, auto-detects the L4
ollama pull gemma4:12b                             # ~7.6 GB, vision-capable, fits the 24 GB L4
curl -s localhost:11434/api/generate -d '{"model":"gemma4:12b","prompt":"READY","stream":false}'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # expect VRAM + util during a call
```

**3. Install the app.**
```bash
# copy the repo to the VM (scp/rsync/git clone), then:
cd ~/treasury
sudo apt-get install -y python3-venv tesseract-ocr   # tesseract = the warning's authoritative OCR
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**4. Run the API as a service** — `/etc/systemd/system/treasury.service`:
```ini
[Unit]
Description=TTB Label Verification API
After=network-online.target ollama.service
Wants=ollama.service
[Service]
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/treasury
Environment=INFERENCE_BACKEND=ollama
Environment=OLLAMA_BASE_URL=http://localhost:11434/v1
Environment=GEMMA_MODEL=gemma4:12b
Environment=OCR_BACKEND=tesseract
Environment=REDIS_URL=
Environment=INFERENCE_TIMEOUT_S=120
ExecStart=/home/YOUR_USER/treasury/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
Restart=on-failure
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now treasury.service
curl -s localhost:8000/health    # {"status":"ok","inference_backend":"ollama","ocr_backend":"tesseract",...}
```
> `REDIS_URL=` (empty) uses the **in-process batch executor** — one uvicorn process, batch upload supported, no Redis to operate. For horizontal scale set `REDIS_URL=redis://…` and run `arq worker.WorkerSettings` alongside; the batch lane is identical either way.

**5. HTTPS reverse proxy (Caddy → automatic Let's Encrypt).**
```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
printf 'treasury.liam.cool {\n    reverse_proxy localhost:8000\n}\n' | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy     # provisions a TLS cert (needs DNS->VM + 80/443 open)
```

**6. Verify.**
```bash
curl -s https://treasury.liam.cool/health
# UI: https://treasury.liam.cool   — upload one label, or drop a batch.
```

Redeploy after a code change: `scp` the changed files to `~/treasury` and `sudo systemctl restart treasury`.

## 12. Design trade-offs & deployment notes

- **AI-generated test labels can't render the 40-word warning verbatim** — image models mangle a long legal paragraph, so a pure text-to-image set can't test the strict warning comparator. Hence the **hybrid** test set (`test_images/generate_labels_hybrid.py`): generate the background art, then composite **pixel-exact** field + warning text with Pillow — ground-truth labels precise enough to exercise missing-clause / reworded / title-case defects.
- **Ollama-on-GPU vs vLLM.** We serve via **Ollama** — the simplest reliable path: OpenAI-compatible, no HF token, and `gemma4:12b` fits the 24 GB L4 with room to spare. **vLLM** remains the higher-throughput production target (continuous batching, guided JSON) and is still wired in (`INFERENCE_BACKEND=vllm`, compose `--profile gpu`); on a 24 GB L4 it needs the E4B variant or FP8-quantized 12B to fit VRAM. The inference client is OpenAI-compatible either way, so swapping backends is a config change.
- **Gemma 4 is a reasoning model — disable thinking.** Through Ollama's OpenAI `/v1` endpoint there's no way to turn reasoning off: the model burns its whole token budget "thinking" (≈3k thinking tokens observed) and returns **empty** content → a 502. The fix (`OllamaInferenceClient`) is Ollama's **native `/api/chat`** with `think: false` + `format: <schema>` (schema-constrained, valid `ModelExtraction`). That took a single label from a 120s timeout to ~12s on the L4. (Output is also token-bounded and the image is downscaled for a faster vision prefill.)
- **Model choice: 12B vs E4B (latency vs accuracy).** On the L4, `gemma4:12b` does a label in ~13s and reads fields accurately. We tried `gemma4:e4b` to chase the ≤5s SLA: it came in at **~7s (still not <5s) and regressed accuracy** — five compliant labels flipped to `needs_review` because the smaller model misreads the value fields (the warning, read by Tesseract, was unaffected). So we kept **12B**. The ≤5s SLA is therefore a target met by *better hardware* (an L40S/A100 runs 12B well under 5s) or a *fine-tuned distilled model* (§10) — not by trading accuracy on this card. Even at ~13s the tool beats the 30–40s vendor the brief mentions.
- **The warning must be read by OCR, not the LLM — verified.** A vision LLM asked to read the *famous* warning hallucinates the canonical text even over a defective label: on the absent-warning sample, `gemma4` confidently returned the full warning that **isn't on the label**; on the title-case sample it silently uppercased the header. So the deterministic OCR reading is authoritative; the VLM is used only for the fields (where it excels) and the bounding boxes, plus an agreement ratio reported for transparency.
- **Warning matching: similarity bands, not an edit budget.** `TesseractOCR` boxes the warning (bottom crop, 2× upscale, `--psm 6`) because full-page OCR mis-segments the small text. We first used a per-character edit *budget*, but label-dependent OCR noise (8–24 edits even on compliant labels) put it on a tightrope. The fix: judge each clause by **fuzzy similarity** — ≥85% present, <60% missing/reworded, in between → review; header casing from the case-preserving OCR. Robust to OCR noise, still catches missing/absent/reworded/title-case. (Inherent limit: a *subtle* single-word reword can't be distinguished from OCR noise — documented; caught only at lower confidence → review.)
- **Visual audit overlay.** The single-label result renders the uploaded image with a labelled, colour-coded box per region (the model returns per-field boxes in normalized 0–1000 coords) — green pass / red fail / amber review — so a non-technical agent sees *where* a defect is at a glance, immediately auditable.
- **Accuracy on the hybrid sample set.** With 12B the tool flags every defect (ABV / class-type / net-contents mismatch, missing-clause / reworded / title-case / absent warning, brand variant) as `fail`/`needs_review`, and passes compliant labels — the conservative bias the brief wants (never auto-pass a defect).
- **In-memory batch.** The live demo runs the in-process batch executor (no Redis) — enough for the demo and trivial to operate; Redis + arq is documented above for scale.
- **Security / exposure.** uvicorn binds `127.0.0.1`; only Caddy (TLS) is public. Nothing is persisted to disk — job results live in memory with a TTL. A production deploy would add auth, narrow the firewall source range, and keep secrets out of the repo (the HF token used only for the vLLM weights path must never be committed, and should be rotated if it ever leaks).

## 13. Performance & hitting the ≤5s SLA

Sarah's hard requirement is **results in ~5s** — the prior vendor's 30–40s killed adoption (§2). Where we are and how to get there:

**Measured:** ~13s per label end-to-end on the live demo. Almost all of it is the **Gemma 4 vision call** (image prefill + JSON decode); Tesseract OCR of the warning is ~1s, and the deterministic verification (rapidfuzz + the warning comparator) is sub-millisecond. So latency is gated entirely by the model call — the verification logic itself is effectively free.

**Why we're at ~13s: the GPU.** The demo runs on an **NVIDIA L4** — Nvidia's *entry-level, economical* inference card (24 GB, ~300 GB/s memory bandwidth). LLM token decode is memory-bandwidth-bound, so a 12B multimodal model is slow on an L4 relative to data-center GPUs. We chose the L4 for cost, not speed.

**How to get under 5s (in order of leverage):**
1. **Use a faster GPU — the primary lever, zero code change.** The same `gemma4:12b` on a higher-bandwidth card decodes several× faster and lands a single label **well under 5s with no accuracy loss**: L40S (~864 GB/s), A100 (~2 TB/s), or H100 (~3.3 TB/s). It's a pure infra swap — identical Ollama/vLLM config, just a bigger instance. **This is the recommended path: the SLA is a hardware-tier choice, not a code problem.**
2. **Serve with vLLM instead of Ollama on the GPU.** Continuous batching + paged attention cut per-request latency under load and maximize throughput for the 200–300-label batch lane (already wired in: `INFERENCE_BACKEND=vllm`, `docker compose --profile gpu`).
3. **Smaller or fine-tuned model.** `gemma4:e4b` is ~7s on the L4 but less accurate (§12); a model **fine-tuned/distilled on real labels** (§10) could be both fast *and* accurate — the best long-term answer if cheap hardware is a hard constraint.
4. **Already applied:** the image is downscaled before the vision prefill, output is token-bounded, structured `format` decoding avoids reasoning-token waste, and the interactive lane is prioritized ahead of the batch lane (§4).

Even at ~13s on the cheapest GPU, the tool already beats the 30–40s vendor that killed adoption; reaching ≤5s is a matter of provisioning an L40S-class (or better) instance.
