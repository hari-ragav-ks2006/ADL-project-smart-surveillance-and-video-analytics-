# PowerShell Launcher for Smart Video Surveillance System
# Automatically activates project virtual environment (.venv)

$VenvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment (.venv) not found. Creating one now..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[INFO] Installing dependencies..." -ForegroundColor Cyan
    & .\.venv\Scripts\pip install -r requirements.txt
}

Write-Host "[INFO] Launching Smart Video Surveillance System with venv..." -ForegroundColor Green
& $VenvPython main.py @args
