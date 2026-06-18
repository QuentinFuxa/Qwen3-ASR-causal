"""Standalone causal streaming runtime for Qwen3-ASR."""

from .asr import Qwen3StreamingASR
from .online import Qwen3StreamingOnlineProcessor

try:
    from .vllm import Qwen3VLLMASR, Qwen3VLLMCausalOnlineProcessor
except Exception:  # pragma: no cover - vLLM is an optional dependency.
    Qwen3VLLMASR = None  # type: ignore[assignment]
    Qwen3VLLMCausalOnlineProcessor = None  # type: ignore[assignment]

try:
    from .metal import (
        Qwen3VLLMMetalASR,
        Qwen3VLLMMetalCausalOnlineProcessor,
        Qwen3VLLMMetalOnlineProcessor,
    )
except Exception:  # pragma: no cover - vLLM Metal is an optional dependency.
    Qwen3VLLMMetalASR = None  # type: ignore[assignment]
    Qwen3VLLMMetalCausalOnlineProcessor = None  # type: ignore[assignment]
    Qwen3VLLMMetalOnlineProcessor = None  # type: ignore[assignment]

Qwen3CausalHFASR = Qwen3StreamingASR
Qwen3CausalVLLMASR = Qwen3VLLMASR
Qwen3CausalMetalASR = Qwen3VLLMMetalASR

__all__ = [
    "Qwen3StreamingASR",
    "Qwen3StreamingOnlineProcessor",
    "Qwen3VLLMASR",
    "Qwen3VLLMCausalOnlineProcessor",
    "Qwen3VLLMMetalASR",
    "Qwen3VLLMMetalCausalOnlineProcessor",
    "Qwen3VLLMMetalOnlineProcessor",
    "Qwen3CausalHFASR",
    "Qwen3CausalVLLMASR",
    "Qwen3CausalMetalASR",
]
