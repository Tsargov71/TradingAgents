#!/bin/bash
#
# ollama_ctl.sh
# =============
# Single entry point to start or stop the ollama container for this project.
#
# Usage:
#   bash scripts/ollama_ctl.sh start   # up -d + pre-warm + GPU check
#   bash scripts/ollama_ctl.sh stop    # stop container, keep the model volume
#   bash scripts/ollama_ctl.sh restart # stop then start
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL"

usage() {
    echo "Usage: $0 {start|stop|restart}"
    exit 1
}

do_start() {
    echo ">>> Bringing up the ollama container..."
    docker compose --profile ollama up -d ollama

    echo ">>> Pre-warming the model (${MODEL})..."
    docker compose --profile ollama exec ollama ollama run "$MODEL" "hello"

    echo ">>> Ready. GPU check:"
    docker compose --profile ollama logs ollama | grep -i -E "gpu|cuda|nvidia" | tail -5
}

do_stop() {
    echo ">>> Stopping the ollama container (VRAM freed, model volume kept)..."
    docker compose --profile ollama stop ollama
    echo ">>> Done. Restart with: bash scripts/ollama_ctl.sh start"
}

case "${1:-}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop
        do_start
        ;;
    *)
        usage
        ;;
esac