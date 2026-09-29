@echo off
setlocal
title Whisper Studio - Local Speech Recognition

echo ======================================================================
echo             Whisper Studio - Speech Recognition
echo             (faster-whisper + CTranslate2 + NVIDIA CUDA)
echo ======================================================================
echo.

:: 1. Check NVIDIA GPU
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    set "GPU_FOUND="
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu=name^,driver_version --format=csv^,noheader 2^>nul') do (
        if not "%%b"=="" (
            echo [OK] NVIDIA GPU detected:
            echo      GPU: %%a ^| Driver: %%b
            set "GPU_FOUND=1"
        )
    )
    if not defined GPU_FOUND (
        echo [NOTICE] nvidia-smi has restricted permissions.
        echo          CUDA runtime and GPU acceleration will be loaded directly by Python.
    )
) else (
    echo [NOTICE] nvidia-smi not found. Running in CPU mode.
)
echo.

:: 2. Check virtual environment
if exist ".venv\Scripts\python.exe" (
    goto activate_and_run
)

echo [*] Virtual environment not found. Starting first-time setup...
echo.

:: Find compatible Python (3.10 - 3.13)
set "PY_CMD="
py -3.13 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.13"
if not defined PY_CMD py -3.12 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.12"
if not defined PY_CMD py -3.11 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.11"
if not defined PY_CMD py -3.10 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.10"
if not defined PY_CMD python -c "import sys; v=sys.version_info; sys.exit(0 if 310<=v[0]*100+v[1]<=313 else 1)" >nul 2>nul && set "PY_CMD=python"

if defined PY_CMD goto create_venv

echo [!] Compatible Python 3.10 - 3.13 not detected.
echo     faster-whisper and CTranslate2 require Python 3.10 - 3.13 on Windows.
echo.
echo [*] Attempting automatic installation of Python 3.13 via winget...
where winget >nul 2>nul
if %errorlevel% equ 0 (
    winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
    if %errorlevel% equ 0 (
        set "PY_CMD=py -3.13"
        goto create_venv
    )
)

echo [*] Downloading official Python 3.13 installer...
curl -L -o "%TEMP%\python-3.13.2-amd64.exe" "https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe"
if exist "%TEMP%\python-3.13.2-amd64.exe" (
    echo [*] Installing Python 3.13 silently...
    start /wait "" "%TEMP%\python-3.13.2-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
    del /f /q "%TEMP%\python-3.13.2-amd64.exe" >nul 2>nul
    set "PY_CMD=py -3.13"
    goto create_venv
)

echo [ERROR] Failed to automatically install Python 3.13.
echo Please install Python 3.13 manually: https://www.python.org/downloads/
pause
exit /b 1

:create_venv
echo [1/3] Creating virtual environment (.venv) using %PY_CMD%...
%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>nul

echo [3/4] Installing dependencies (CUDA 12, faster-whisper, UI, HF transfer)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [4/4] Checking llama-cpp-python (CUDA 12 LLM acceleration)...
python -c "from core.cuda_utils import setup_cuda_dlls; setup_cuda_dlls(); import llama_cpp" >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] Installing pre-compiled CUDA 12 llama-cpp-python wheel...
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to install pre-compiled wheel. Trying standard install...
        pip install llama-cpp-python
    )
)
echo [*] Verifying CPU compatibility (Intel Alder Lake / AVX2)...
python -c "from core.ensure_llamacpp import ensure_compatible_llamacpp; ensure_compatible_llamacpp()"

echo.
echo [OK] Setup completed successfully!
echo.

:activate_and_run
if not exist "models" mkdir "models"
if not exist "models\llm" mkdir "models\llm"

:: Disable buggy Xet on Windows to ensure reliable, high-speed CDN downloads with resume
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

:: Load saved HF_TOKEN from config.json if available
if exist "config.json" (
    for /f "tokens=2 delims=:, " %%T in ('findstr /i "hf_token" config.json 2^>nul') do (
        if not "%%~T"=="" if not "%%~T"=="null" (
            set "HF_TOKEN=%%~T"
            set "HUGGING_FACE_HUB_TOKEN=%%~T"
        )
    )
)

call .venv\Scripts\activate.bat
echo [*] Starting Whisper Studio...
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Whisper Studio exited with code %errorlevel%.
    pause
)
