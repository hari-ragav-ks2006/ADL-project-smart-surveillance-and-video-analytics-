@echo off
REM Convenience launcher for Smart Video Surveillance System
REM Automatically activates project virtual environment (.venv)

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found. Creating one now...
    python -m venv .venv
    echo [INFO] Installing requirements...
    .\.venv\Scripts\pip install -r requirements.txt
)

echo [INFO] Launching Smart Video Surveillance System...
.\.venv\Scripts\python.exe main.py %*
