chcp 65001
@echo off
setlocal EnableDelayedExpansion

echo ================================
echo  🚀 Starting Talk to Your Data (TTYD)
echo ================================

:: Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker first.
    exit /b 1
)
echo ✅ Docker found!

:: Check if Docker Compose is installed
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose.
    exit /b 1
)
echo ✅ Docker Compose found!

:: Check if Ollama is installed
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Ollama is not installed. Please install Ollama from https://ollama.com.
    exit /b 1
)
echo ✅ Ollama found!

:: Check if Ollama is running
curl -s --head http://localhost:11434 | find "200 OK" >nul
if %errorlevel% neq 0 (
    echo ❌ Ollama server is NOT running. Please start it manually.
    exit /b 1
)
echo ✅ Ollama is running!

:: Detect NVIDIA GPU
where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ No NVIDIA GPU detected. Running in CPU mode.
    set GPU_SUPPORT=no
) else (
    echo ✅ NVIDIA GPU detected!
    set GPU_SUPPORT=yes
)

:: Check if NVIDIA Container Toolkit is installed (for GPU support)
if "%GPU_SUPPORT%"=="yes" (
    docker info | find "NVIDIA" >nul
    if %errorlevel% neq 0 (
        echo ❌ NVIDIA Container Toolkit is not installed. GPU acceleration will not work.
        set GPU_SUPPORT=no
    ) else (
        echo ✅ NVIDIA Container Toolkit is installed. GPU acceleration enabled!
    )
)

:: Load environment variables from .env
if exist .env (
    for /f "delims=" %%x in (.env) do (
        echo %%x | findstr /R /C:"^[A-Za-z0-9_]*=" >nul && set %%x
    )
) else (
    echo ⚠️  No .env file found! Using default settings from docker-compose.yml.
)

:: Start Docker Compose with appropriate flags
echo 🔥 Starting services...
if "%GPU_SUPPORT%"=="yes" (
    docker compose --env-file .env up --build --gpus all
) else (
    docker compose --env-file .env up --build
)

exit /b 0