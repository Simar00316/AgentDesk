# syntax=docker/dockerfile:1.7
# Worker image: same code, plus system libs for document parsing and a pre-downloaded embedding model.
# LAYOUT=1 (default) adds layout-aware PDF parsing (Unstructured hi_res: torch CPU, poppler, tesseract,
# ~1.5 GB). Build with --build-arg LAYOUT=0 for a slim image that parses PDFs with pdfminer only.
ARG LAYOUT=1
FROM python:3.12-slim AS builder
ARG LAYOUT
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
COPY apps/api/pyproject.toml apps/api/uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv EXTRA=$([ "$LAYOUT" = "1" ] && echo "--extra layout"); \
    uv sync --frozen --no-dev --no-install-project $EXTRA || uv sync --no-dev --no-install-project $EXTRA
COPY apps/api/ ./

FROM python:3.12-slim
ARG LAYOUT
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" APP_ENV=production FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface
RUN groupadd -r app && useradd -r -g app -d /app app && apt-get update && apt-get install -y --no-install-recommends libmagic1 libxml2 libxslt1.1 \
    $([ "$LAYOUT" = "1" ] && echo "poppler-utils tesseract-ocr libgl1 libglib2.0-0") \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app
# pre-download the default local embedding + BM25 models so first ingestion is fast
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')" || true
# pre-download the layout (YOLOX) and table-structure models used by hi_res parsing
RUN [ "$LAYOUT" != "1" ] || python -c "from unstructured_inference.models.base import get_model; get_model('yolox'); \
from unstructured_inference.models.tables import load_agent; load_agent()" || true
CMD ["arq", "app.worker.WorkerSettings"]
