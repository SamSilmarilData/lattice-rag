# ==============================================================================
# lattice-rag: Multi-stage Production Container Build for Google Cloud Run
# ==============================================================================

# Stage 1: Build React single-page frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python dependencies (compilers stay out of the runtime image)
FROM python:3.12-slim AS python-builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Stage 3: Slim runtime as an unprivileged user
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
RUN chown app:app /app
USER app

# Pre-download ONNX embedding & cross-encoder reranker weights to eliminate cold-start download delays
# Built from AppConfig so the pinned reranker revision is the one cached, not the latest
RUN python -c "from lattice_rag.config import load_config; from lattice_rag.retrieval.embeddings import EmbeddingService; c = load_config(require_typesafe=False); s = EmbeddingService(c.embed_model, c.reranker_model, onnx_reranker_repo=c.reranker_onnx_repo, onnx_reranker_revision=c.reranker_onnx_revision); s.embed_query('warmup'); s.rerank('warmup', ['doc'])"

COPY --chown=app:app --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY --chown=app:app eval_dataset.json eval_baseline.json fallback_queries.json ./

# Pre-seed LatticeDB with the golden corpus. `seed` is storage-only and needs no API keys.
RUN lattice-rag seed --dataset eval_dataset.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["sh", "-c", "lattice-rag serve --host 0.0.0.0 --port ${PORT:-8000}"]
