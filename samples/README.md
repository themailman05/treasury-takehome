# Samples

Example application data and notes for generating test label images.

## `manifest.json`

A `POST /jobs` batch manifest. The API accepts **two** manifest shapes
(see `app.py`):

1. **Per-file map** — `{ "<filename>": { ...fields }, ... }` (this file). Each
   uploaded image is matched to its expected values by filename.
2. **Single application** — a flat `{ "brand_name": ..., "abv": ... }` object,
   applied to *every* uploaded image. Useful when all labels share one expected
   record.

Field keys mirror `ApplicationData` in `schemas.py` — all optional:
`brand_name`, `class_type`, `abv`, `net_contents`. Omitted fields are simply
not verified (skipped, not failed).

## Single-label quick test (interactive lane)

Matches the README §7 curl example:

```bash
curl -F "image=@samples/old_tom_bourbon.png" \
     -F 'application={"brand_name":"OLD TOM DISTILLERY","abv":"45","net_contents":"750 mL"}' \
     http://localhost:8000/verify
```

## Batch test (throughput lane)

```bash
curl -F "images=@samples/old_tom_bourbon.png" \
     -F "images=@samples/blue_heron_bourbon.png" \
     -F "manifest=$(cat samples/manifest.json)" \
     http://localhost:8000/jobs
# -> 202 {"job_id": "..."}; then poll:
curl http://localhost:8000/jobs/<job_id>
```

## Generating test label images

There is no need for real images to exercise the service: with the default
`INFERENCE_BACKEND=mock` the mock client echoes the application values back and
emits a clean canonical warning, so any (even empty) image PASSes out of the
box. For realistic and **defective** labels (missing warning clause, title-case
header, ABV/brand/net-contents mismatch, bad photos), use one of:

- **`test_images/generate_labels.py`** (in this repo) — generates ~150 labels
  with an AI image model plus a ground-truth `manifest.jsonl` / `manifest.csv`
  whose `expected_overall` column doubles as a verifier assertion table.
  Defect categories map 1:1 to what the verifier checks.

  ```bash
  pip install google-genai python-dotenv pillow
  export GEMINI_API_KEY=...
  python test_images/generate_labels.py --count 20      # smaller batch
  python test_images/generate_labels.py --dry-run       # specs + manifest, no API calls
  ```

- **Any AI image tool** — prompt for a bottle/can label that renders the brand,
  class/type, ABV, net contents, and the government warning paragraph.

- **Photograph a real bottle** — angle, glare, and low light exercise the
  imperfect-image path (`image_quality` -> `needs_review`, OCR robustness).

### Fixture shortcut (deterministic, no GPU)

The mock inference client treats an uploaded file that decodes to UTF-8 JSON as
a `ModelExtraction` fixture. This lets you unit-test specific extractions (e.g.
a dropped warning clause) without any image model:

```bash
cat > /tmp/fixture.json <<'JSON'
{"brand_name":{"text":"OLD TOM DISTILLERY","confidence":0.97},
 "class_type":{"text":"Kentucky Straight Bourbon Whiskey","confidence":0.95},
 "abv":{"text":"45% Alc./Vol. (90 Proof)","confidence":0.96},
 "net_contents":{"text":"750 mL","confidence":0.98},
 "warning":{"vlm_text":"GOVERNMENT WARNING: ...","ocr_text":"GOVERNMENT WARNING: ...",
            "located":true,"bbox":[10,200,400,80],"confidence":0.93},
 "image_quality":"ok"}
JSON

curl -F "image=@/tmp/fixture.json" \
     -F 'application={"brand_name":"OLD TOM DISTILLERY","abv":"45","net_contents":"750 mL"}' \
     http://localhost:8000/verify
```
