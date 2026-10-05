# ==============================================================================
# lattice-rag: Multi-stage Production Container Build for Google Cloud Run
# ==============================================================================

# Stage 1: Build React single-page frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python ASGI runner
FROM python:3.12-slim
WORKDIR /app

# Install system build dependencies for C/C++ native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Pre-download ONNX embedding & cross-encoder reranker weights to eliminate cold-start download delays
RUN python -c "from lattice_rag.retrieval.embeddings import EmbeddingService; s = EmbeddingService(); s.embed_query('warmup'); s.rerank('warmup', ['doc'])"

# Copy compiled frontend from Stage 1 into the container
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY eval_dataset.json eval_baseline.json fallback_queries.json ./

# Pre-seed LatticeDB database with the 5 golden corpus documents for instant zero-latency startup
RUN python -m lattice_rag.cli seed --dataset eval_dataset.json

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "lattice-rag serve --host 0.0.0.0 --port ${PORT:-8000}"]
