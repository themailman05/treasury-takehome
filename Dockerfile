# API + worker image for the AI Label Verification Prototype.
#
# One image serves both compose roles (README §4): the FastAPI app
# (`uvicorn app:app`) and the arq batch worker (`arq worker.WorkerSettings`),
# selected per service via `command:` in docker-compose.yml.
#
# README §6 pins the API to Python 3.11; this is the deployment runtime (the
# repo also runs fine on newer Python locally).
FROM python:3.11-slim

# Faster, quieter, log-friendly Python in a container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System packages ---------------------------------------------------------
# The compose stack runs OCR_BACKEND=tesseract (the warning's authoritative
# reading), so the pytesseract wrapper needs the tesseract system binary baked in.
# (For a pure OCR_BACKEND=mock build this layer is harmless dead weight.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (layer-cached separately from source) ---------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- Application source ------------------------------------------------------
# Flat module layout at the repo root: enums.py, config.py, schemas.py,
# verify_warning.py, verify_fields.py, inference.py, ocr.py, pipeline.py,
# jobs.py, worker.py, app.py, plus the static SPA and samples.
COPY . .

EXPOSE 8000

# Default role is the interactive API server. The `worker` compose service
# overrides this with `command: arq worker.WorkerSettings`.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
