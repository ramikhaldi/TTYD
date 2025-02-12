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

# ✅ Remove any existing `docker-compose.override.yml`
rm -f docker-compose.override.yml

# ✅ Check if NVIDIA GPU is available
GPU_SUPPORT="no"
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
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
    fi

    # ✅ Generate `docker-compose.override.yml` to enable GPU only for `ollama`
    cat <<EOF > docker-compose.override.yml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
    echo -e "${GREEN}✅ GPU support enabled for 'ollama' in docker-compose.override.yml.${NC}"
else
    echo -e "${YELLOW}⚠️ No NVIDIA GPU detected. Running in CPU mode.${NC}"
fi

# ✅ Check if `my_files/` directory exists & is NOT empty
DIR="my_files"
if [ ! -d "$DIR" ]; then
    echo -e "${RED}❌ ERROR: '$DIR' directory is missing! Please create it and add your files before running the application!${NC}"
    exit 1
fi

if [ -z "$(ls -A $DIR 2>/dev/null)" ]; then
    echo -e "${RED}❌ ERROR: '$DIR' directory is empty! Please add files.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ '$DIR' directory exists and is not empty.${NC}"

# ✅ Start Services
echo -e "${YELLOW}🚀 Starting TTYD...${NC}"
docker compose --env-file .env up --build
