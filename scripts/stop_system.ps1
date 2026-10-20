# ==============================================================================
# Adaptive RAG System — Windows PowerShell Graceful Stop Script
# ==============================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "STOPPING ADAPTIVE RAG SYSTEM SERVICES" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Stop FastAPI Backend (Port 8000)
Write-Host "`n[1/2] Checking port 8000 (FastAPI Backend)..." -NoNewline
$port8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($port8000) {
    $pid8000 = $port8000.OwningProcess
    try {
        Stop-Process -Id $pid8000 -Force -ErrorAction Stop
        Write-Host " [STOPPED PID $pid8000]" -ForegroundColor Green
    } catch {
        Write-Host " [FAILED TO STOP PID ${pid8000}: $_]" -ForegroundColor Red
    }
} else {
    Write-Host " [NOT RUNNING]" -ForegroundColor Gray
}

# 2. Stop Streamlit Frontend (Port 8501)
Write-Host "`n[2/2] Checking port 8501 (Streamlit Frontend)..." -NoNewline
$port8501 = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($port8501) {
    $pid8501 = $port8501.OwningProcess
    try {
        Stop-Process -Id $pid8501 -Force -ErrorAction Stop
        Write-Host " [STOPPED PID $pid8501]" -ForegroundColor Green
    } catch {
        Write-Host " [FAILED TO STOP PID ${pid8501}: $_]" -ForegroundColor Red
    }
} else {
    Write-Host " [NOT RUNNING]" -ForegroundColor Gray
}

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "Service termination complete. (Ollama service was left untouched)." -ForegroundColor Gray
Write-Host "=====================================================================" -ForegroundColor Cyan
