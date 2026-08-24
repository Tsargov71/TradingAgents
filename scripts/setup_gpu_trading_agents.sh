#!/bin/bash

# Setup una tantum
docker build -t tradingagents-5090:latest -f Dockerfile_5090 .
docker compose --profile ollama up -d --force-recreate ollama

# Pre-warm del modello (evita il timeout da cold-load)
docker compose --profile ollama exec ollama ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL "ciao"

# Verifica GPU (opzionale, diagnostica)
docker compose --profile ollama logs ollama | grep -i -E "gpu|cuda|nvidia"

# Comandi per avviare trading agents
# cd ..
# docker compose --profile ollama run --rm tradingagents-ollama
