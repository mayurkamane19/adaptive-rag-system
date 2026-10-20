# Production Dockerfile for Adaptive RAG System Backend (FastAPI)
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS dependencies for network operations & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create an unprivileged user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Pre-create data directories with write permissions for appuser
RUN mkdir -p /app/data/documents \
             /app/data/vectorstore \
             /app/data/conversations \
             /app/data/evaluation && \
    chown -R appuser:appgroup /app

# Copy application source code
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup data/ /app/data/
COPY --chown=appuser:appgroup run.py /app/run.py

# Switch to non-root user
USER appuser

EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Launch FastAPI via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
