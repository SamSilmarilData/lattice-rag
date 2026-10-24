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

The service exposes an unauthenticated health probe at `/health`. `status` is `degraded` when the database or Redis is unavailable, or when the server booted without `TYPESAFE_API_KEY` (hermetic mode, `typesafeConfigured: false`). The container image also declares a Docker `HEALTHCHECK` against this endpoint.

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
   curl -s -H "X-Admin-Key: $ADMIN_API_KEY" "http://localhost:8000/api/v1/graph/subgraph?limit=500" > backup_graph.json
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

2. **Log Redaction**:
   - `RedactingFormatter` redacts every rendered log line, tracebacks included. structlog is routed through stdlib logging so its events are covered too.
   - The pattern masks API keys (`sk-`, `gsk_`, `AIza`), `Bearer <token>`, `token=` and URI credentials (`://user:pass@host`, `://:pass@host`).

3. **RFC 9457 Problem Details Error Shielding**:
   - Production error handlers redact internal tracebacks, OS file paths, and database errors into client-safe problem JSON objects:
     ```json
     {
       "type": "https://api.lattice-rag.io/errors/internal",
       "title": "Internal Server Error",
       "status": 500,
       "detail": "An internal error occurred. Reference: 3f2c…",
       "requestId": "3f2c…"
     }
     ```
   - The sanitized exception is logged server-side under the same `requestId`.

4. **Access Control**:
   - Ingest, graph snapshot, cache stats, eval and benchmark require `X-Admin-Key` (`ADMIN_API_KEY`); query endpoints are rate-limited per client; request sizes are bounded. See `docs/security.md` §3.4.

5. **Static Secret Scanning in CI/CD**:
   - Automated git test scans all tracked repository files to guarantee no accidental commits of `apikeys.md`, `.env`, or hardcoded API tokens.

---

## 5. Scaling Strategy

### 5.1 Vertical Scaling (Single Node)
- **CPU**: Allocate at least 2 vCPUs (recommended: 4 vCPUs). ONNX runtime uses `intra_op_num_threads=4` for cross-encoder reranking.
- **RAM**: Allocate at least 1 GB (recommended: 2 GB) to provide comfortable headroom for concurrent FastEmbed embedding batches and graph traversals.

### 5.2 Horizontal Scaling
LatticeDB is an embedded, single-file database, so each process owns its own copy of the graph. Run **one** writable instance (the Cloud Run script pins `--max-instances 1`). Scaling out needs extra work that is not built in:
- **Read replicas**: copy a seeded `lattice_rag.db` into each replica and route `/api/v1/ingest` to a single writer. Replicas only see new documents after the file is redistributed.
- **Per-process state**: the Tier 1 semantic cache, the Groq circuit breaker and the rate-limit counters are in-memory per process.
- **Shared FAQ store**: set `REDIS_URL=redis://redis-cluster:6379/0` so every instance serves the same Tier 2 fallback answers.

---

## 6. Continuous Integration & Quality Gates Pipeline

`lattice-rag` runs quality gates on every push and pull request via GitHub Actions (`.github/workflows/eval-gate.yml`):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/d10-cicd-pipeline-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="diagrams/d10-cicd-pipeline-light.svg">
  <img alt="CI Quality Gates & Deploy Pipeline" src="diagrams/d10-cicd-pipeline-dark.svg" width="100%">
</picture>

<sub>Interactive &amp; High-Resolution Inspection: <a href="diagrams/d10-cicd-pipeline-dark.html">Dark Specimen</a> | <a href="diagrams/d10-cicd-pipeline-light.html">Light Specimen</a></sub>

### Gate Execution Sequence:
1. **Parallel Quality Gates** (all actions pinned to commit SHAs):
   - **`lint`**: `ruff check src tests` (rules `E,F,W,I,N,UP,B,A,C4,SIM`), `ruff format --check src tests` (ruff version pinned in `pyproject.toml`), and `mypy src` in strict mode.
   - **`unit-tests`**: the hermetic `pytest tests/unit/` suite with mock keys and no network, including the security audit (`test_security_audit.py`: secret scan of tracked files, `SecretStr` masking, DTO field audit, log redaction, error shielding).
   - **`frontend`**: `npm ci && npm run build` (`tsc -b` type-checks the studio against `types.ts`, then Vite builds it).
2. **Golden Dataset Evaluation Gate** (after all three gates pass; skipped for fork PRs or without `TYPESAFE_API_KEY`):
   - Runs `lattice-rag eval --dataset eval_dataset.json --baseline eval_baseline.json` on an ephemeral database.
   - Fails on a mean regression above 0.03, any query's faithfulness below 0.50, or a Noul contradiction veto. Latency SLAs are not part of this gate; check them with `lattice-rag benchmark --sla-check`.

Container images are built by Cloud Run from source at deploy time (`scripts/deploy_cloud_run.sh`); there is no registry publishing step in CI.
