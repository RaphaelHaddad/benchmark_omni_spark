#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Load environment variables
set -a
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  source "${SCRIPT_DIR}/.env"
fi
set +a

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR:-/tmp/vlm_logs}"

echo "=========================================="
echo "Starting Qwen3-Omni vLLM server"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Host: ${VLLM_HOST:-0.0.0.0}"
echo "Port: ${VLLM_PORT:-8078}"
echo "GPU Memory: ${GPU_MEMORY_UTILIZATION:-0.90}"
echo "=========================================="

# Quick check for NVIDIA GPU support
if ! docker info 2>/dev/null | grep -qi "nvidia"; then
  echo "WARNING: Docker does not show NVIDIA runtime info."
  echo "Ensure NVIDIA Container Toolkit is installed." >&2
fi

# Stop any existing container and start fresh
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" down --remove-orphans
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d --build

echo ""
echo "Server started successfully!"
echo "View logs: docker compose -f ${SCRIPT_DIR}/docker-compose.yml logs -f"
echo "API will be available at: http://${VLLM_HOST:-0.0.0.0}:${VLLM_PORT:-8078}/v1"
