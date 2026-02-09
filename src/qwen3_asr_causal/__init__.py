"""Standalone causal streaming runtime for Qwen3-ASR."""

from .asr import Qwen3StreamingASR
from .online import Qwen3StreamingOnlineProcessor

try:
    from .vllm import Qwen3VLLMASR, Qwen3VLLMCausalOnlineProcessor
except Exception:  # pragma: no cover - vLLM is an optional dependency.
    Qwen3VLLMASR = None  # type: ignore[assignment]
    Qwen3VLLMCausalOnlineProcessor = None  # type: ignore[assignment]

Qwen3CausalHFASR = Qwen3StreamingASR
Qwen3CausalVLLMASR = Qwen3VLLMASR

__all__ = [
    "Qwen3StreamingASR",
    "Qwen3StreamingOnlineProcessor",
    "Qwen3VLLMASR",
    "Qwen3VLLMCausalOnlineProcessor",
    "Qwen3CausalHFASR",
    "Qwen3CausalVLLMASR",
]
