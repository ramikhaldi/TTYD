@echo off
setlocal EnableDelayedExpansion

chcp 65001
echo ================================
echo [LAUNCH] Starting Talk to Your Data (TTYD)
echo ================================

:: Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker first.
    exit /b 1
)
echo [OK] Docker found!

:: Check if Docker Compose is installed
docker compose version >nul 2>nul
if %errorlevel% neq 0 (
    where docker-compose >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Docker Compose is not installed. Please install Docker Compose.
        exit /b 1
    )
)
echo [OK] Docker Compose found!

:: Detect NVIDIA GPU
where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARN] No NVIDIA GPU detected. Running in CPU mode.
    set GPU_SUPPORT=no
) else (
    echo [OK] NVIDIA GPU detected!
    set GPU_SUPPORT=yes
)

:: Check if NVIDIA Container Toolkit is installed (for GPU support)
if "%GPU_SUPPORT%"=="yes" (
    docker info | find "NVIDIA" >nul
    if %errorlevel% neq 0 (
        echo [WARN] NVIDIA Container Toolkit is not installed. GPU acceleration will not work.
        set GPU_SUPPORT=no
    ) else (
        echo [OK] NVIDIA Container Toolkit is installed. GPU acceleration enabled!
    )
)

:: Load environment variables from .env
if exist .env (
    for /f "tokens=1,2 delims==" %%A in (.env) do (
        set "%%A=%%B"
    )
) else (
    echo [WARN] No .env file found! Using default settings from docker-compose.yml.
)

:: Check if `my_files/` directory exists and is NOT empty
set DIR=my_files
if not exist "%DIR%" (
    echo ERROR: "%DIR%" directory is missing. Please create it and add your files before running the application!
    exit /b 1
)

dir /a-d /b "%DIR%" | findstr /r /c:"^." >nul
if %errorlevel% neq 0 (
    echo ERROR: "%DIR%" directory is empty! Please add files.
    exit /b 1
)

echo [OK] "%DIR%" directory exists and is not empty.

:: Start Docker Compose with appropriate flags
echo [LAUNCH] Starting services...
if "%GPU_SUPPORT%"=="yes" (
    docker compose --env-file .env up --build --gpus all
) else (
    docker compose --env-file .env up --build
)

exit /b 0
