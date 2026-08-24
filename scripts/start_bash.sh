#!/bin/bash
#
# start_bash.sh
# ==============
# Opens an interactive shell inside the tradingagents-ollama container,
# reusing the service definition in docker-compose.yml AS-IS (env_file,
# environment, volumes, depends_on: ollama, project network) — so there's
# no duplicated config to keep in sync between this script and the compose
# file.
#
# Prerequisite: the ollama container must already be up (GPU + model
# pre-warmed). If not yet done:
#   bash scripts/ollama_ctl.sh start
#
# Usage:
#   bash scripts/start_bash.sh
#   # inside the container:
#   python -u tools_debug/main_debug.py
#
set -euo pipefail

# Moved into scripts/, so go up one level to reach the repo root where
# docker-compose.yml lives.
cd "$(dirname "$0")/.."

# Make sure ollama is up (idempotent: no-op if it's already running)
docker compose --profile ollama up -d ollama

# --entrypoint /bin/bash overrides ENTRYPOINT ["tradingagents"] from the
# Dockerfile, only for this interactive invocation.
docker compose --profile ollama run --rm --entrypoint /bin/bash tradingagents-ollama