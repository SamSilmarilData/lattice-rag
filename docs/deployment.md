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
Cloud Run scales to zero when idle. An active GraphRAG query with hybrid retrieval, Cypher traversal, cross-encoder reranking, and Jev guardrail finishes in ~800ms:
$$\text{Free Queries} = \frac{360,000 \text{ GiB-seconds}}{2 \text{ GiB} \times 0.8\text{s}} \approx \mathbf{225,000 \text{ queries / month for } \$0.00}$$

### Deployment Steps

1. **Prerequisites**: Install the [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **One-Command Automated Deployment**:
   ```bash
   ./scripts/deploy_cloud_run.sh YOUR_PROJECT_ID us-central1
   ```

3. **Configure Environment Secrets**:
   Set your production API keys in Cloud Run (via Cloud Console or `gcloud`):
   ```bash
   gcloud run services update lattice-rag \
     --region us-central1 \
     --set-env-vars TYPESAFE_API_KEY="your-typesafe-key",GROQ_API_KEY="your-groq-key",GEMINI_API_KEY="your-gemini-key"
   ```

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

---

## 5. Cold-Start Optimization

`lattice-rag`'s multi-stage `Dockerfile` incorporates two key production optimizations:
1. **Pre-Baked Model Weights**: ONNX weights for `bge-small-en-v1.5` and `bge-reranker-v2-m3` are downloaded during image creation (`docker build`), eliminating 600MB+ network pulls on container cold start.
2. **Pre-Seeded Database Schema**: LatticeDB property graph tables and BM25 FTS5 indexes are initialized at build time, guaranteeing sub-second container readiness.
