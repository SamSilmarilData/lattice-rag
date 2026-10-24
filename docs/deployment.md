# Zero-Cost Production Deployment Guide

This guide details how to host and operate `lattice-rag` for **$0.00/month** in production without compromising on retrieval quality or downgrading our state-of-the-art **`bge-reranker-v2-m3`** cross-encoder.

---

## 1. Memory Architecture & The 512MB RAM Ceiling

During Phase 6 profiling, resident memory consumption across all pipeline subsystems was empirically benchmarked on macOS/Linux:

| Component | Resident Memory (RAM) | Notes |
|---|---|---|
| **Base Python Runtime + Granian** | ~14 MB | In-process ASGI server overhead |
| **FastEmbed `bge-small-en-v1.5`** | ~289 MB | INT8 ONNX dense bi-encoder |
| **Cross-Encoder `bge-reranker-v2-m3`** | **1,176 MB (1.17 GB)** | SOTA multilingual cross-encoder (544MB ONNX weights expand to ~887MB during inference) |
| **LatticeDB + Caches** | ~35 MB | SQLite property graph + FTS5 indices + vector tables |
| **Total Peak Working Set** | **~1.51 GB** | Required allocation: **$\ge$ 2 GB RAM** |

> [!WARNING]
> **512MB Free Tier OOM Hazard:**
> Platforms offering only 512MB RAM (such as Render Free, Koyeb Free, or basic Hugging Face CPU Spaces) **will crash with an Out-Of-Memory (OOM) kill** when executing cross-encoder reranking. To maintain uncompromised retrieval precision, `lattice-rag` is deployed to free-tier environments providing **$\ge$ 2GB RAM**.

---

## 2. Option A: Google Cloud Run (Always Free Tier) — Recommended

Google Cloud Run provides an **Always Free tier** every single month that perfectly accommodates `lattice-rag`:
- **360,000 GiB-seconds** of memory per month
- **180,000 vCPU-seconds** per month
- **2,000,000 requests** per month
- Configurable instance memory up to **2 GiB or 4 GiB**

### Zero-Cost Calculation
Cloud Run scales to zero when idle. An active GraphRAG query with hybrid retrieval, graph traversal, cross-encoder reranking, and Jev guardrail finishes in ~800ms:
$$\text{Free Queries} = \frac{360,000 \text{ GiB-seconds}}{2 \text{ GiB} \times 0.8\text{s}} \approx \mathbf{225,000 \text{ queries / month for } \$0.00}$$

### Deployment Steps

1. **Prerequisites**: Install the [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Store the API keys in Secret Manager** (once per project). The deploy script refuses to run until all four exist, and keys are never passed as plain environment variables or baked into the image:
   ```bash
   for name in TYPESAFE_API_KEY GROQ_API_KEY GEMINI_API_KEY ADMIN_API_KEY; do
     read -rsp "$name: " value && echo
     printf '%s' "$value" | gcloud secrets create "$name" --data-file=-
   done
   ```
   Grant the Cloud Run runtime service account `roles/secretmanager.secretAccessor` on these secrets.

3. **One-Command Automated Deployment**:
   ```bash
   ./scripts/deploy_cloud_run.sh YOUR_PROJECT_ID us-central1
   ```
   The service is deployed with `--max-instances 1` because the LatticeDB file lives on the instance's ephemeral disk: documents ingested at runtime are lost on restart or redeploy, while the golden corpus is pre-seeded into the image. The service is publicly reachable, but ingest, eval and benchmark require the `X-Admin-Key` header.

4. **Verify Deployment**:
   Open the Cloud Run service URL in your browser:
   - Root UI: `https://<service-url>/`
   - API Health: `https://<service-url>/health`
   - Interactive Scalar API Docs: `https://<service-url>/schema/scalar`

---

## 3. Option B: Cloudflare Tunnel (`cloudflared`) — Zero-Cost Edge Exposure

If you have a local workstation (e.g., Apple Silicon M-series Mac or Linux box) with $\ge 8$GB RAM, you can host `lattice-rag` locally with near-instant hardware-accelerated reranking (<100ms) and expose it globally via Cloudflare:

1. **Install `cloudflared`**:
   ```bash
   brew install cloudflared  # macOS
   ```

2. **Start the Engine**:
   ```bash
   lattice-rag serve --host 127.0.0.1 --port 8000
   ```

3. **Launch the Public Tunnel**:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
   Cloudflare will output a public HTTPS URL (e.g., `https://random-words.trycloudflare.com`) with automated SSL, global edge caching, and DDoS mitigation — **100% free with zero configuration**.

---

## 4. Option C: Oracle Cloud Infrastructure (OCI Always Free) — 24GB Cloud VM

Oracle Cloud provides the industry's most generous persistent compute tier for free:
- **4 Ampere ARM vCPUs**
- **24 GB RAM**
- **200 GB persistent NVMe storage**
- Runs 24/7 without scaling down

1. Create an Ubuntu VM instance in the OCI Console.
2. Install Docker:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io
   ```
3. Clone and build `lattice-rag`:
   ```bash
   git clone https://github.com/your-org/lattice-rag.git
   cd lattice-rag
   docker build -t lattice-rag .
   docker run -d --restart=always -p 8000:8000 --env-file .env lattice-rag
   ```
   If you put nginx or Caddy in front of the container, set `TRUSTED_PROXY_HOPS=1` in `.env` (see below).

---

## Client identity behind reverse proxies (`TRUSTED_PROXY_HOPS`)

The query rate limit (`QUERY_RATE_LIMIT_PER_MINUTE`) and the admin-key lockout both key on the client's address. `TRUSTED_PROXY_HOPS` says how many reverse proxies sit between the client and the app:

| Value | Client identity | Use for |
|-------|-----------------|---------|
| `0` (default) | Socket peer; `X-Forwarded-For` is ignored | Direct exposure, local dev |
| `1` | Rightmost `X-Forwarded-For` entry | Cloud Run, a single nginx/Caddy, Cloudflare Tunnel |
| `N` | `N`th entry from the right | `N` proxies that each append to `X-Forwarded-For` |

Every proxy appends the address it received the request from, so entries to the left of the trusted ones were written by the client and are ignored. A header with fewer than `N` entries is not trusted and the socket peer is used.

Set the value to the real hop count. Too low, and every client behind the proxy shares one identity (one rate-limit budget, one lockout). Too high, and clients can choose their identity by sending their own `X-Forwarded-For`. `scripts/deploy_cloud_run.sh` sets `TRUSTED_PROXY_HOPS=1`; override it (`TRUSTED_PROXY_HOPS=2 ./scripts/deploy_cloud_run.sh ...`) when an external HTTPS load balancer fronts the service.

**Admin-key lockout.** A client that sends a wrong `X-Admin-Key` 5 times within 5 minutes gets `429` from the admin endpoints for the rest of that window, even with the right key. Other clients are unaffected, and a successful attempt clears the count. Counts are kept in process memory per server worker and reset on restart.

---

## 5. Cold-Start Optimization

`lattice-rag`'s multi-stage `Dockerfile` incorporates two key production optimizations:
1. **Pre-Baked Model Weights**: ONNX weights for `bge-small-en-v1.5` and `bge-reranker-v2-m3` are downloaded during image creation (`docker build`), eliminating 600MB+ network pulls on container cold start.
2. **Pre-Seeded Database**: `lattice-rag seed` ingests the golden corpus at build time (it needs no API keys and skips documents that already exist), so the graph, HNSW and BM25 indexes are ready when the container starts.

The runtime stage runs as an unprivileged user (`uid 10001`), excludes compilers (dependencies are built in a separate stage), and declares a `HEALTHCHECK` on `/health`.
