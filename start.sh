#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================${NC}"
echo -e "${GREEN} 🚀 Starting Talk to Your Data (TTYD)${NC}"
echo -e "${YELLOW}================================${NC}"

# ✅ Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ ERROR: Docker is not installed! Please install Docker.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Docker is installed.${NC}"
fi

# ✅ Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ ERROR: Docker Compose is not installed! Please install Docker Compose.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Docker Compose is installed.${NC}"
fi

# ✅ Check if NVIDIA GPU is available
GPU_SUPPORT="no"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected.${NC}"
    GPU_SUPPORT="yes"

    # ✅ Check if NVIDIA Container Toolkit is installed
    if ! docker info | grep -q "nvidia"; then
        echo -e "${RED}❌ ERROR: NVIDIA Container Toolkit is not installed!${NC}"
        echo -e "${YELLOW}🔧 Install it using: ${NC}"
        echo -e "   sudo apt install -y nvidia-container-toolkit"
        exit 1
    else
        echo -e "${GREEN}✅ NVIDIA Container Toolkit is installed.${NC}"

        # ✅ Update `.env` ONLY if it exists & contains `DOCKER_RUNTIME`
        if [ -f .env ] && grep -q "DOCKER_RUNTIME=" .env; then
            sed -i 's/DOCKER_RUNTIME=.*/DOCKER_RUNTIME=nvidia/' .env
        fi
    fi
else
    echo -e "${YELLOW}⚠️ No NVIDIA GPU detected. Running on CPU.${NC}"

    # ✅ Update `.env` ONLY if it exists & contains `DOCKER_RUNTIME`
    if [ -f .env ] && grep -q "DOCKER_RUNTIME=" .env; then
        sed -i 's/DOCKER_RUNTIME=.*/DOCKER_RUNTIME=runc/' .env
    fi
fi

# ✅ Check if `my_files/` directory exists & is NOT empty
DIR="my_files"
if [ ! -d "$DIR" ]; then
    echo -e "${RED}❌ ERROR: '$DIR' directory is missing!${NC}"
    exit 1
fi

if [ -z "$(ls -A $DIR 2>/dev/null)" ]; then
    echo -e "${RED}❌ ERROR: '$DIR' directory is empty! Please add files.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ '$DIR' directory exists and is not empty.${NC}"

# ✅ Start Services
echo -e "${YELLOW}🚀 Starting TTYD...${NC}"
if [ "$GPU_SUPPORT" = "yes" ]; then
    docker compose --env-file .env up --build --gpus all
else
    docker compose --env-file .env up --build
fi
