#!/usr/bin/env bash
set -euo pipefail

export WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING="${WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING:-1}"

qwen3-asr-causal transcribe audio.wav \
  --backend vllm \
  --decoder-backend vllm-live \
  --language en
