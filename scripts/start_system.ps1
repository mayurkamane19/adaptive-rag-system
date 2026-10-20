# ==============================================================================
# Adaptive RAG System — Windows PowerShell Startup Script
# ==============================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "STARTING ADAPTIVE RAG SYSTEM SERVICES" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 0. Check Virtual Environment
$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$venvStreamlit = Join-Path $PSScriptRoot "..\.venv\Scripts\streamlit.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment not found at .venv\Scripts\python.exe." -ForegroundColor Red
    Write-Host "Please create and initialize the environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Gray
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Gray
    exit 1
}

# 1. Check Ollama Service
Write-Host "`n[1/3] Checking Ollama Engine..." -NoNewline
try {
    $ollama = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
    Write-Host " [ONLINE]" -ForegroundColor Green
} catch {
    Write-Host " [WARNING: OFFLINE]" -ForegroundColor Yellow
    Write-Host "      Notice: Ollama is not responding at http://localhost:11434." -ForegroundColor Yellow
    Write-Host "      Ensure Ollama is started by running 'ollama serve' in a separate terminal." -ForegroundColor Gray
}

# 2. Check & Start FastAPI Backend (Port 8000)
Write-Host "`n[2/3] Checking FastAPI Backend (Port 8000)..." -NoNewline
$port8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($port8000) {
    Write-Host " [ALREADY RUNNING (PID: $($port8000.OwningProcess))]" -ForegroundColor Green
} else {
    Write-Host " [STARTING]" -ForegroundColor Cyan
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    Start-Process -FilePath $venvPython -ArgumentList "-m uvicorn app.main:app --host 127.0.0.1 --port 8000" -WorkingDirectory $projectRoot -WindowStyle Minimized
    
    # Wait for backend liveness
    $backendReady = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $live = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health/live" -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($live.status -eq "alive") {
                $backendReady = $true
                break
            }
        } catch { }
    }
    if ($backendReady) {
        Write-Host "      FastAPI Backend successfully started on http://127.0.0.1:8000" -ForegroundColor Green
    } else {
        Write-Host "      [WARNING] Backend took longer than expected to report liveness." -ForegroundColor Yellow
    }
}

# 3. Check & Start Streamlit Frontend (Port 8501)
Write-Host "`n[3/3] Checking Streamlit Frontend (Port 8501)..." -NoNewline
$port8501 = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($port8501) {
    Write-Host " [ALREADY RUNNING (PID: $($port8501.OwningProcess))]" -ForegroundColor Green
} else {
    Write-Host " [STARTING]" -ForegroundColor Cyan
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    Start-Process -FilePath $venvStreamlit -ArgumentList "run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1" -WorkingDirectory $projectRoot -WindowStyle Minimized
    Start-Sleep -Seconds 3
    Write-Host "      Streamlit UI successfully started on http://localhost:8501" -ForegroundColor Green
}

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "SYSTEM ACCESS URLS:" -ForegroundColor Green
Write-Host "  Streamlit User Interface : http://localhost:8501" -ForegroundColor White
Write-Host "  FastAPI REST API Docs    : http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  Health Status Endpoint   : http://127.0.0.1:8000/health/ready" -ForegroundColor White
Write-Host "=====================================================================" -ForegroundColor Cyan
