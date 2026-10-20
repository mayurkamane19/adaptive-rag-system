# Production Deployment & Cloud Hosting Guide

This guide details deployment options for the **Adaptive Retrieval-Augmented Generation (RAG) System** across local container engines and cloud providers.

---

## 1. Architecture & Service Topology

```text
                  [User Browser / Client]
                             │
            ┌────────────────┴────────────────┐
            │                                 │
     HTTP :8501                        HTTP :8000
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │ Streamlit UI    │ ── internal ──▶ │ FastAPI Backend │
   │ (Frontend)      │               │ (RAG Workflow)  │
   └─────────────────┘               └────────┬────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
             ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
             │ Persistent Disk │     │ Hugging Face    │     │ Ollama Service  │
             │   (data/ dir)   │     │ Embeddings/Rerank│    │ (Local/Remote)  │
             └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 2. Docker & Container Orchestration

### Prerequisites

- Docker Engine 20.10+ & Docker Compose v2+
- 4GB+ RAM recommended for local embeddings & re-ranking

### Quick Start

```bash
# 1. Clone repository & configure environment
cp .env.example .env

# 2. Build and start containers in detached mode
docker compose up -d --build

# 3. Verify health probes
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

### Starting with Embedded Ollama Server

```bash
# Launch with Ollama profile
docker compose --profile with-ollama up -d

# Pull model into Ollama volume
docker exec -it adaptive-rag-ollama ollama pull llama3
```

---

## 3. Recommended Cloud Deployment Options (Free & Low-Cost)

### Option A: Railway (Recommended for Simplicity)

1. Fork or push repository to GitHub.
2. In Railway Dashboard, create a **New Project from GitHub Repo**.
3. Deploy two services:
   - **Service 1 (Backend)**: Point to root `Dockerfile`. Set port `8000`. Add a persistent volume mounted at `/app/data`.
   - **Service 2 (Frontend)**: Point to `Dockerfile.streamlit`. Set port `8501`. Add environment variable `BACKEND_API_URL` pointing to backend's private Railway domain.
4. Set environment variable `OLLAMA_BASE_URL` to point to a cloud Ollama instance or remote OpenAI-compatible endpoint.

### Option B: Render (Web Service + Persistent Disk)

1. Create a **Web Service** on Render using Docker runtime.
2. Root directory: `./`, Dockerfile: `Dockerfile`.
3. Add a persistent disk attached at `/app/data` (1GB+).
4. Set environment variables:
   - `HOST=0.0.0.0`
   - `PORT=8000`
   - `ALLOWED_ORIGINS=https://your-frontend.onrender.com`
   - `API_KEY=your-secure-api-key`
5. Create a second Web Service for Streamlit using `Dockerfile.streamlit`.

### Option C: AWS Lightsail / Small VPS (Single Instance Docker Compose)

- **Cost**: \$3.50 – \$5.00/month (or AWS Free Tier).
- **Setup**:
  1. Provision an Ubuntu 22.04 LTS Lightsail instance (2GB RAM + 1 vCPU).
  2. Install Docker and Docker Compose.
  3. Clone repository and run `docker compose up -d`.
  4. Configure firewall ports for `80` (reverse proxy via Nginx/Caddy) or `8000`/`8501`.

---

## 4. Security Checklist for Production

- [ ] **CORS**: Configure `ALLOWED_ORIGINS` to specific domains instead of `*`.
- [ ] **Authentication**: Set a strong `API_KEY` in environment variables to protect admin endpoints (`/api/v1/evaluation/run` and session deletions).
- [ ] **TLS / HTTPS**: Terminate TLS at the reverse proxy (Cloudflare, Nginx, or provider load balancer).
- [ ] **File Size Boundary**: Set `MAX_UPLOAD_SIZE_MB` appropriately (default `25MB`).
- [ ] **Volume Persistence**: Ensure the `/app/data` volume is backed up periodically.
- [ ] **Secrets**: Never commit `.env` containing sensitive credentials to version control.

---

## 5. Persistent Storage & Backup

All application state is contained in the `data/` directory:

- `data/documents/`: Raw uploaded PDF files.
- `data/vectorstore/`: FAISS index (`index.faiss` and `index.pkl`).
- `data/conversations/`: Session chat histories in JSON format.
- `data/evaluation/`: Benchmark evaluation dataset.

### Periodic Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backups/rag"
mkdir -p "$BACKUP_DIR"
tar -czvf "$BACKUP_DIR/rag_data_$(date +%Y%m%d_%H%M%S).tar.gz" data/
find "$BACKUP_DIR" -type f -mtime +30 -delete
```
