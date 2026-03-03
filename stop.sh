#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Stopping Qwen3-Omni vLLM server..."
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" down --remove-orphans

echo "Server stopped."
