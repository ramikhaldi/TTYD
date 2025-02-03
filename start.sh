#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Running System Sanity Check...${NC}"

# ✅ Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed! Please install Docker.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Docker is installed.${NC}"
fi

# ✅ Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed! Please install Docker Compose.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Docker Compose is installed.${NC}"
fi

# ✅ Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama is not installed! Please install Ollama: https://ollama.com${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Ollama is installed.${NC}"
fi

# ✅ Check if Ollama is running
if ! pgrep -f "ollama" > /dev/null; then
    echo -e "${RED}❌ Ollama is not running! Please start Ollama before launching TTYD.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Ollama is running.${NC}"
fi

# ✅ Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected.${NC}"

    # ✅ Check if NVIDIA Container Toolkit is installed
    if ! docker info | grep -q "nvidia"; then
        echo -e "${RED}❌ NVIDIA Container Toolkit is not installed!${NC}"
        echo -e "${YELLOW}🔧 Install it!${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ NVIDIA Container Toolkit is installed.${NC}"
        sed -i 's/DOCKER_RUNTIME=runc/DOCKER_RUNTIME=nvidia/' .env  # Update .env dynamically
    fi
else
    echo -e "${YELLOW}⚠️ No NVIDIA GPU detected. Running on CPU.${NC}"
    sed -i 's/DOCKER_RUNTIME=nvidia/DOCKER_RUNTIME=runc/' .env  # Ensure CPU mode
fi

echo -e "${YELLOW}🚀 Starting TTYD...${NC}"

# ✅ Start the container
docker-compose up --build
