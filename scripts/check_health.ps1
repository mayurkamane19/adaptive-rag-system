# ==============================================================================
# Adaptive RAG System — Subsystem Health & Status Probe
# ==============================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "ADAPTIVE RAG SYSTEM - SUBSYSTEM HEALTH CHECK" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Ollama LLM Service
Write-Host "`n[1/4] Probing Ollama Engine (http://localhost:11434)..." -NoNewline
try {
    $ollama = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
    $modelList = ($ollama.models | ForEach-Object { $_.name }) -join ", "
    Write-Host " [ONLINE]" -ForegroundColor Green
    Write-Host "      Installed Models: $modelList" -ForegroundColor Gray
} catch {
    Write-Host " [OFFLINE/UNREACHABLE]" -ForegroundColor Red
    Write-Host "      Hint: Start Ollama by running 'ollama serve' in a separate terminal." -ForegroundColor Yellow
}

# 2. FastAPI Backend
Write-Host "`n[2/4] Probing FastAPI Backend (http://127.0.0.1:8000)..." -NoNewline
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
    $live = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health/live" -Method Get -TimeoutSec 5 -ErrorAction Stop
    $ready = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health/ready" -Method Get -TimeoutSec 10 -ErrorAction Stop

    $vsStatus = $ready.components.vector_store.status
    $vsChunks = if ($ready.components.vector_store.chunks_indexed -ne $null) { $ready.components.vector_store.chunks_indexed } else { $ready.components.vector_store.total_chunks }
    $rrStatus = $ready.components.reranker.status
    $rrModel = $ready.components.reranker.model

    Write-Host " [HEALTHY]" -ForegroundColor Green
    Write-Host "      Service: $($health.service) v$($health.version)" -ForegroundColor Gray
    Write-Host "      Liveness: $($live.status) | Readiness: $($ready.status)" -ForegroundColor Gray
    Write-Host "      Vector Store: $vsStatus ($vsChunks chunks indexed)" -ForegroundColor Gray
    Write-Host "      Re-ranker: $rrStatus (Model: $rrModel)" -ForegroundColor Gray
} catch {
    Write-Host " [OFFLINE/UNREACHABLE]" -ForegroundColor Red
    Write-Host "      Hint: Run '.\scripts\start_system.ps1' or 'uvicorn app.main:app --port 8000'." -ForegroundColor Yellow
}

# 3. Streamlit Frontend
Write-Host "`n[3/4] Probing Streamlit UI (http://localhost:8501)..." -NoNewline
try {
    $st_resp = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    if ($st_resp.StatusCode -eq 200) {
        Write-Host " [ONLINE]" -ForegroundColor Green
        Write-Host "      Access URL: http://localhost:8501" -ForegroundColor Gray
    } else {
        Write-Host " [HTTP $($st_resp.StatusCode)]" -ForegroundColor Yellow
    }
} catch {
    Write-Host " [OFFLINE/UNREACHABLE]" -ForegroundColor Red
    Write-Host "      Hint: Run 'streamlit run frontend/streamlit_app.py'." -ForegroundColor Yellow
}

# 4. Port Status Summary
Write-Host "`n[4/4] Active Network Ports:" -ForegroundColor Cyan
$ports = Get-NetTCPConnection -LocalPort 8000, 8501, 11434 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess
if ($ports) {
    $ports | Format-Table -AutoSize
} else {
    Write-Host "      No active listening sockets found on 8000, 8501, or 11434." -ForegroundColor Yellow
}

Write-Host "=====================================================================" -ForegroundColor Cyan
