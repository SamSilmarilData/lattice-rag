# Production Readiness & Operator Runbook

This guide provides operational guidance for deploying, maintaining, monitoring, and scaling `lattice-rag` in production environments.

---

## 1. System Architecture & Lifecycle

`lattice-rag` is packaged as a single self-contained application service powered by:
- **Rust ASGI Server**: Granian running HTTP/1.1 and SSE streaming.
- **Web Framework**: Litestar 2.24+ with `msgspec` DTOs.
- **Embedded Database**: LatticeDB 0.15+ (in-process SQLite-based property graph + HNSW + BM25).
- **Embedded ML Inferences**: FastEmbed (`bge-small-en-v1.5`), GLiNER (`gliner-2.5-decide`), and INT8 ONNX Cross-Encoder (`bge-reranker-v2-m3`).
- **Orchestration**: LangGraph state graph with TypeSafe AI front-door decision triage and context guardrailing.

### 1.1 Single-Container Production Deployment Topology

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/d9-deployment-topology-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="diagrams/d9-deployment-topology-light.svg">
  <img alt="Single-Container Production Deployment Topology" src="diagrams/d9-deployment-topology-dark.svg" width="100%">
</picture>

<sub>Interactive &amp; High-Resolution Inspection: <a href="diagrams/d9-deployment-topology-dark.html">Dark Specimen</a> | <a href="diagrams/d9-deployment-topology-light.html">Light Specimen</a></sub>

### 1.2 ApplicationCore Lifecycle & State Machine

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/d8-core-lifespan-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="diagrams/d8-core-lifespan-light.svg">
  <img alt="ApplicationCore Lifespan & Service State Machine" src="diagrams/d8-core-lifespan-dark.svg" width="100%">
</picture>

<sub>Interactive &amp; High-Resolution Inspection: <a href="diagrams/d8-core-lifespan-dark.html">Dark Specimen</a> | <a href="diagrams/d8-core-lifespan-light.html">Light Specimen</a></sub>

---

## 2. Health Monitoring & Container Probes

The service exposes an unauthenticated health probe at `/health`:

```bash
curl -f http://localhost:8000/health
```

### Response Schema:
```json
{
  "status": "healthy",
  "latticedbConnected": true,
  "redisConnected": true,
  "typesafeConfigured": true,
  "groqConfigured": true,
  "geminiConfigured": true
}
```

### 2.1 Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 2
  failureThreshold: 2
```

### 2.2 Google Cloud Run Probes

Cloud Run automatically probes container ports. For startup probes:
```bash
gcloud run services update lattice-rag \
  --startup-probe-path=/health \
  --startup-probe-initial-delay=10s \
  --startup-probe-timeout=3s
```

---

## 3. Database Backup, Restore & Disaster Recovery

LatticeDB persists all entities, relation edges, chunk texts, HNSW vector graphs, and BM25 inverted indices into a single binary file located at `data/lattice_rag.db`.

### 3.1 Hot Backup Procedure

Because LatticeDB uses transactional locking, safe backups can be taken while the service is operating:

1. **Filesystem Snapshot (Recommended)**:
   If using cloud persistent volumes (e.g. AWS EBS, GCP Persistent Disk, DigitalOcean Volumes), snapshot the volume directly:
   ```bash
   # GCP Persistent Disk Snapshot
   gcloud compute disks snapshot lattice-rag-data --snapshot-names=lattice-rag-backup-$(date +%Y%m%d%H%M)
   ```

2. **Atomic File Copy**:
   ```bash
   # Copy database to backup target
   cp data/lattice_rag.db data/lattice_rag.db.backup-$(date +%Y%m%d_%H%M%S)
   ```

3. **Subgraph Snapshot Export**:
   Export graph knowledge as JSON via the REST API:
   ```bash
   curl -s "http://localhost:8000/api/v1/graph/subgraph?limit=5000" > backup_graph.json
   ```

### 3.2 Restoration Procedure

1. Stop the `lattice-rag` process or scale container replicas to 0.
2. Replace `data/lattice_rag.db` with the backup copy:
   ```bash
   cp data/lattice_rag.db.backup-20261006_000000 data/lattice_rag.db
   chmod 644 data/lattice_rag.db
   ```
3. Restart the service:
   ```bash
   lattice-rag serve --port 8000
   ```
4. Verify integrity via the benchmark tool:
   ```bash
   lattice-rag benchmark --iterations 2 --warmup 1 --sla-check
   ```

---

## 4. Security Hardening & Zero-Leakage Protections

`lattice-rag` includes multi-layered native security defenses verified by automated security audit test suites (`tests/unit/test_security_audit.py`):

1. **`SecretStr` Memory Wrapping**:
   - `TYPESAFE_API_KEY`, `GROQ_API_KEY`, and `GEMINI_API_KEY` are wrapped in `SecretStr`.
   - `__str__` and `__repr__` automatically redact the value (`sk-****1234` or `[REDACTED]`).
   - Plaintext keys can only be retrieved by calling `.get_secret_value()`.

2. **Streamed Log Redaction**:
   - `RedactingFilter` scans all Python standard logging and `structlog` records before emission.
   - Regex patterns automatically mask API keys (`sk-`, `gsk_`, `AIza`, `Bearer <token>`, and `://user:pass@host` database URIs).

3. **RFC 9457 Problem Details Error Shielding**:
   - Production error handlers redact internal tracebacks, OS file paths, and database errors into client-safe problem JSON objects:
     ```json
     {
       "type": "https://lattice-rag.io/errors/internal-error",
       "title": "Internal Server Error",
       "status": 500,
       "detail": "An internal error occurred while processing the request."
     }
     ```

4. **Static Secret Scanning in CI/CD**:
   - Automated git test scans all tracked repository files to guarantee no accidental commits of `apikeys.md`, `.env`, or hardcoded API tokens.

---

## 5. Scaling Strategy

### 5.1 Vertical Scaling (Single Node)
- **CPU**: Allocate at least 2 vCPUs (recommended: 4 vCPUs). ONNX runtime uses `intra_op_num_threads=4` for cross-encoder reranking.
- **RAM**: Allocate at least 1 GB (recommended: 2 GB) to provide comfortable headroom for concurrent FastEmbed embedding batches and graph traversals.

### 5.2 Horizontal Scaling
- **Stateless Replicas**: The application code, routing, and synthesis stages are completely stateless.
- **Shared Storage**: Mount `data/lattice_rag.db` over a shared block volume with read-only replicas, or use an active-passive setup with primary write node and read replicas.
- **External Redis Tier 2**: For multi-node deployments, set `REDIS_URL=redis://redis-cluster:6379/0` so that circuit breaker fallback caching is shared across all application instances.

---

## 6. Continuous Integration & Quality Gates Pipeline

`lattice-rag` enforces strict multi-stage quality gates on every push and pull request via GitHub Actions (`.github/workflows/eval-gate.yml`):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/d10-cicd-pipeline-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="diagrams/d10-cicd-pipeline-light.svg">
  <img alt="Strict CI/CD Quality Gates & Release Pipeline" src="diagrams/d10-cicd-pipeline-dark.svg" width="100%">
</picture>

<sub>Interactive &amp; High-Resolution Inspection: <a href="diagrams/d10-cicd-pipeline-dark.html">Dark Specimen</a> | <a href="diagrams/d10-cicd-pipeline-light.html">Light Specimen</a></sub>

### Gate Execution Sequence:
1. **Parallel Quality Gates (Fail-Fast Verification)**:
   - **Gate 01 (Ruff Lint & Format)**: Fast AST linting (`ruff check --select E,F,W,I,N,UP,B,A,C4,SIM`).
   - **Gate 02 (Mypy Strict Typing)**: Enforces complete type safety with zero missing annotations across all modules (`mypy --strict src/`).
   - **Gate 03 (Zero-Leakage Security Scan)**: Scans for credential exposures, verifies `SecretStr` masking, and confirms `.env` / `apikeys.md` git blocklists (`test_security_audit.py`).
   - **Gate 04 (Hermetic Pytest Matrix)**: Executes 130+ unit tests under mocked environments with zero external network dependencies.
2. **Golden Dataset Evaluation Gate**:
   - Triggers `lattice-rag eval --dataset eval_dataset.json --baseline eval_baseline.json`.
   - Validates multi-hop retrieval accuracy, groundedness scores, and verifies the Sub-Second SLA ($p95 < 1,000$ms).
3. **Hermetic Docker Release**:
   - Compiles and publishes signed container images to GitHub Container Registry (`ghcr.io/lattice-rag:latest`) upon all gates passing.
