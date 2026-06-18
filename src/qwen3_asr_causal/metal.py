"""
Qwen3-ASR backend using vllm-metal's in-process STT runtime.

This backend does not use vLLM's HTTP or WebSocket APIs. It loads the
vllm-metal MLX model directly, re-transcribes the current audio buffer, and
streams by committing every word except the last two.
"""

from __future__ import annotations

import logging
import platform
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np

from .types import ASRToken, Transcript

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_VLLM_METAL_MODEL = "Qwen/Qwen3-ASR-0.6B"
QWEN3_VLLM_METAL_1_7B_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN3_VLLM_METAL_CAUSAL_TOWER = "qfuxa/qwen3-asr-0.6b-streaming"

QWEN3_VLLM_METAL_MODEL_MAPPING = {
    "base": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "tiny": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "small": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "qwen3-asr-0.6b": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "qwen3-0.6b": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "0.6b": DEFAULT_QWEN3_VLLM_METAL_MODEL,
    "qwen3-asr-1.7b": QWEN3_VLLM_METAL_1_7B_MODEL,
    "qwen3-1.7b": QWEN3_VLLM_METAL_1_7B_MODEL,
    "1.7b": QWEN3_VLLM_METAL_1_7B_MODEL,
}

_UNSUPPORTED_QWEN3_VLLM_METAL_ALIASES = {
    "medium",
    "large",
    "large-v3",
}

_SENTENCE_ENDINGS = (".", "!", "?")

_VLLM_METAL_INSTALL_HINT = (
    "Install vLLM first with the official vllm-metal install script, then "
    "install the vllm-metal STT extra. The WhisperLiveKit extra only adds "
    "the vllm-metal wheel on supported Apple Silicon/Python 3.12 builds."
)


class _Qwen3MetalWorker:
    """Run all MLX/vllm-metal work on one thread."""

    def __init__(self):
        self._tasks: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="qwen3-vllm-metal-worker",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._init_error:
            raise self._init_error

    def _run(self) -> None:
        self._thread_id = threading.get_ident()
        try:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
        except BaseException as exc:
            self._init_error = exc
        finally:
            self._ready.set()

        if self._init_error:
            return

        while True:
            task = self._tasks.get()
            if task is None:
                return

            fn, args, kwargs, result_queue = task
            try:
                result_queue.put((True, fn(*args, **kwargs)))
            except BaseException as exc:
                result_queue.put((False, exc))

    def call(self, fn: Callable, *args, **kwargs):
        if threading.get_ident() == self._thread_id:
            return fn(*args, **kwargs)

        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._tasks.put((fn, args, kwargs, result_queue))
        ok, value = result_queue.get()
        if ok:
            return value
        raise value


def _missing_dependency_error(exc: ImportError | None = None) -> ImportError:
    missing_name = getattr(exc, "name", "") if exc is not None else ""

    if missing_name == "vllm" or missing_name.startswith("vllm."):
        return ImportError(
            "qwen3-vllm-metal found vllm-metal STT, but the vLLM CPU package "
            f"is missing. {_VLLM_METAL_INSTALL_HINT}"
        )

    if missing_name == "vllm_metal" or missing_name.startswith("vllm_metal."):
        return ImportError(
            "qwen3-vllm-metal requires vllm-metal with STT support. "
            "On Apple Silicon with Python 3.12, install "
            "`qwen3-asr-causal[metal]` or follow the official "
            f"vllm-metal install instructions. {_VLLM_METAL_INSTALL_HINT}"
        )

    if missing_name == "mlx" or missing_name.startswith("mlx."):
        return ImportError(
            "qwen3-vllm-metal requires MLX on Apple Silicon. "
            "Use Darwin arm64 with a supported vllm-metal installation."
        )

    return ImportError(
        "qwen3-vllm-metal requires vllm-metal STT on Apple Silicon. "
        f"{_VLLM_METAL_INSTALL_HINT}"
    )


def _ensure_supported_platform():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ImportError(
            "qwen3-vllm-metal requires Apple Silicon (Darwin arm64) because "
            "vllm-metal runs on MLX/Metal."
        )


def _resolve_model_path(kwargs: dict) -> str:
    model_path = kwargs.get("model_dir") or kwargs.get("model_path")
    if model_path:
        return model_path

    model_size = (kwargs.get("model_size") or "").strip()
    if not model_size:
        return DEFAULT_QWEN3_VLLM_METAL_MODEL

    lowered = model_size.lower()
    if "/" in model_size or model_size.startswith((".", "/")):
        return model_size
    if lowered in QWEN3_VLLM_METAL_MODEL_MAPPING:
        return QWEN3_VLLM_METAL_MODEL_MAPPING[lowered]
    if lowered in _UNSUPPORTED_QWEN3_VLLM_METAL_ALIASES:
        raise ValueError(
            "qwen3-vllm-metal supports Qwen3-ASR 0.6B and 1.7B; "
            f"got unsupported alias {model_size!r}."
        )
    return model_size


def _resolve_mlx_dtype(mx, kwargs: dict):
    """Resolve shared vLLM dtype names to MLX dtype objects."""
    explicit_dtype = kwargs.get("dtype")
    if explicit_dtype is not None:
        return explicit_dtype

    dtype_name = kwargs.get("vllm_dtype") or "auto"
    if dtype_name == "auto":
        return mx.float16
    if not isinstance(dtype_name, str):
        return dtype_name

    dtype_map = {
        "float16": mx.float16,
        "fp16": mx.float16,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
        "float32": mx.float32,
        "fp32": mx.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(
            "qwen3-vllm-metal vllm_dtype must be one of auto, float16, "
            f"bfloat16, or float32; got {dtype_name!r}"
        ) from exc


def _token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            return int(token_id)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer could not encode required token {token!r}")
    return int(token_ids[0])


def _resolve_audio_backend(kwargs: dict) -> str:
    backend = str(kwargs.get("qwen3_vllm_metal_audio_backend", "standard") or "standard")
    if backend not in {"standard", "causal"}:
        raise ValueError(
            "qwen3_vllm_metal_audio_backend must be 'standard' or 'causal', "
            f"got {backend!r}"
        )
    return backend


@dataclass
class _MLXAudioLayerCache:
    key: Any = None
    value: Any = None


@dataclass
class _MLXCausalAudioState:
    mel_buffer: Any = None
    layer_caches: list[_MLXAudioLayerCache] = field(default_factory=list)
    frames_seen: int = 0
    emitted_steps: int = 0
    pending_frames: int = 0
    last_input_frames: int = 0
    last_recomputed_frames: int = 0


@dataclass
class _MLXDecoderRollingState:
    """Persistent MLX decoder KV over [prompt head + audio embeddings]."""

    cache: list[Any] | None = None
    head_token_ids: tuple[int, ...] = ()
    head_len: int = 0
    audio_steps: int = 0
    disabled: bool = False
    last_raw_tokens: list[int] = field(default_factory=list)
    last_stats: dict[str, Any] = field(default_factory=dict)


def _mlx_decoder_cache_seq_length(cache: list[Any] | None) -> int:
    if not cache or cache[0] is None:
        return 0
    first = cache[0]
    if not isinstance(first, tuple) or len(first) != 2:
        return 0
    return int(first[0].shape[2])


def _crop_mlx_decoder_cache(cache: list[Any] | None, seq_len: int) -> list[Any] | None:
    if cache is None:
        return None
    seq_len = max(0, int(seq_len))
    cropped = []
    for layer_cache in cache:
        if layer_cache is None:
            cropped.append(None)
            continue
        key, value = layer_cache
        cropped.append((key[:, :, :seq_len, :], value[:, :, :seq_len, :]))
    return cropped


class _Qwen3MetalCausalAudioEncoder:
    """Append-only causal audio encoder implemented against vllm-metal's MLX tower."""

    def __init__(
        self,
        audio_tower,
        *,
        left_context_sec: float = 15.0,
        block_frames: int = 192,
        block_bidirectional: bool = True,
    ) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        self.mx = mx
        self.nn = nn
        self.audio_tower = audio_tower
        self.config = audio_tower._config
        self.chunk_frames = 8
        self.block_frames = int(block_frames)
        if self.block_frames < 0:
            raise ValueError("qwen3_vllm_metal_block_frames must be >= 0")
        if self.block_frames and self.block_frames % self.chunk_frames != 0:
            raise ValueError(
                "qwen3_vllm_metal_block_frames must be a multiple of 8 mel frames"
            )
        if left_context_sec <= 0:
            raise ValueError("qwen3_vllm_metal_left_context_sec must be > 0")
        left_context_frames = int(round(left_context_sec * 1000.0 / 10.0))
        self.left_context_steps = max(
            1, self.output_steps_for_mel_frames(left_context_frames)
        )
        self.block_bidirectional = bool(block_bidirectional)

    def init_state(self) -> _MLXCausalAudioState:
        return _MLXCausalAudioState(
            layer_caches=[_MLXAudioLayerCache() for _ in self.audio_tower.layers]
        )

    def empty_features(self):
        mx = self.mx
        return mx.zeros(
            (0, self.config.output_dim),
            dtype=self.audio_tower.proj2.weight.dtype,
        )

    @staticmethod
    def output_steps_for_mel_frames(mel_frames: int) -> int:
        if mel_frames <= 0:
            return 0
        length = int(mel_frames)
        for _ in range(3):
            length = (length - 1) // 2 + 1
        return int(length)

    def _position_embedding(self, *, offset: int, length: int, dtype):
        mx = self.mx
        table = self.audio_tower._positional_embedding
        if offset + length <= table.shape[0]:
            return table[offset : offset + length, :].astype(dtype)

        dim = int(table.shape[1])
        if dim % 2 != 0:
            raise ValueError("Qwen audio sinusoidal dim must be even")
        half = dim // 2
        inv = mx.exp(
            -np.log(10000.0)
            / float(max(1, half - 1))
            * mx.arange(half, dtype=mx.float32)
        )
        positions = mx.arange(offset, offset + length, dtype=mx.float32)
        scaled = positions[:, None] * inv[None, :]
        return mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=1).astype(dtype)

    def _conv_one_block(self, block, *, position_offset: int):
        nn = self.nn
        if block.shape[1] == 0:
            return self.mx.zeros(
                (block.shape[0], 0, self.config.d_model), dtype=block.dtype
            )
        tower = self.audio_tower
        x = block.astype(tower.conv_out.weight.dtype).transpose(0, 2, 1)[..., None]
        x = nn.gelu(tower.conv2d1(x))
        x = nn.gelu(tower.conv2d2(x))
        x = nn.gelu(tower.conv2d3(x))
        batch, freq, steps, channels = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(batch, steps, channels * freq)
        x = tower.conv_out(x)
        pos = self._position_embedding(
            offset=position_offset,
            length=steps,
            dtype=x.dtype,
        )
        return x + pos[None, :, :]

    def _conv_blocks(self, mels, *, position_offset: int):
        mx = self.mx
        if mels.shape[1] == 0:
            return mx.zeros(
                (mels.shape[0], 0, self.config.d_model), dtype=mels.dtype
            )
        outputs = []
        step_offset = int(position_offset)
        for start in range(0, int(mels.shape[1]), self.chunk_frames):
            block = mels[:, start : start + self.chunk_frames, :]
            hidden = self._conv_one_block(block, position_offset=step_offset)
            outputs.append(hidden)
            step_offset += int(hidden.shape[1])
        return mx.concatenate(outputs, axis=1)

    def _attention_chunk(
        self,
        attn,
        hidden_states,
        cache: _MLXAudioLayerCache,
        *,
        position_offset: int,
    ):
        mx = self.mx
        batch, length, _ = hidden_states.shape
        if length == 0:
            return hidden_states, cache
        num_heads = int(attn.n_head)
        head_dim = int(attn.head_dim)
        scale = head_dim**-0.5

        q = attn.q_proj(hidden_states).reshape(batch, length, num_heads, head_dim)
        k = attn.k_proj(hidden_states).reshape(batch, length, num_heads, head_dim)
        v = attn.v_proj(hidden_states).reshape(batch, length, num_heads, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        past_k = cache.key
        past_v = cache.value
        past_len = 0 if past_k is None else int(past_k.shape[2])
        if past_k is not None and past_v is not None:
            all_k = mx.concatenate([past_k, k], axis=2)
            all_v = mx.concatenate([past_v, v], axis=2)
        else:
            all_k = k
            all_v = v

        total_len = int(all_k.shape[2])
        cache_start_pos = int(position_offset) - past_len
        q_positions = mx.arange(position_offset, position_offset + length)
        k_positions = mx.arange(cache_start_pos, cache_start_pos + total_len)
        if self.block_bidirectional:
            block_max = int(position_offset) + length - 1
            allowed = k_positions[None, :] <= block_max
        else:
            allowed = k_positions[None, :] <= q_positions[:, None]
        allowed = allowed & (
            k_positions[None, :]
            >= (q_positions[:, None] - self.left_context_steps + 1)
        )

        scores = (
            q.astype(mx.float32) * scale
        ) @ all_k.astype(mx.float32).transpose(0, 1, 3, 2)
        neg = mx.full(scores.shape, -1.0e9, dtype=scores.dtype)
        scores = mx.where(allowed[None, None, :, :], scores, neg)
        weights = mx.softmax(scores, axis=-1, precise=True)
        context = (weights.astype(all_v.dtype) @ all_v).transpose(0, 2, 1, 3)
        context = context.reshape(batch, length, -1)
        output = attn.out_proj(context.astype(attn.out_proj.weight.dtype))

        keep = min(total_len, self.left_context_steps)
        next_cache = _MLXAudioLayerCache(
            key=all_k[:, :, -keep:, :],
            value=all_v[:, :, -keep:, :],
        )
        return output.astype(hidden_states.dtype), next_cache

    def _layer_chunk(
        self,
        layer,
        hidden_states,
        cache: _MLXAudioLayerCache,
        *,
        position_offset: int,
    ):
        nn = self.nn
        residual = hidden_states
        normed = layer.self_attn_layer_norm(hidden_states)
        attn_out, next_cache = self._attention_chunk(
            layer.self_attn,
            normed,
            cache,
            position_offset=position_offset,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        hidden_states = layer.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = layer.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, next_cache

    def _encode_ready_mels(self, mels, state: _MLXCausalAudioState):
        mx = self.mx
        nn = self.nn
        if mels.shape[1] == 0:
            return mx.zeros(
                (mels.shape[0], 0, self.config.output_dim), dtype=mels.dtype
            )
        if not state.layer_caches:
            state.layer_caches = [_MLXAudioLayerCache() for _ in self.audio_tower.layers]
        if len(state.layer_caches) != len(self.audio_tower.layers):
            raise ValueError("Qwen vllm-metal causal audio cache layer count mismatch")

        position_offset = int(state.emitted_steps)
        hidden_states = self._conv_blocks(mels, position_offset=position_offset)
        next_caches = []
        for layer, cache in zip(self.audio_tower.layers, state.layer_caches, strict=True):
            hidden_states, next_cache = self._layer_chunk(
                layer,
                hidden_states,
                cache,
                position_offset=position_offset,
            )
            next_caches.append(next_cache)
        state.layer_caches = next_caches

        tower = self.audio_tower
        hidden_states = tower.ln_post(hidden_states)
        hidden_states = tower.proj1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = tower.proj2(hidden_states)
        state.emitted_steps += int(hidden_states.shape[1])
        return hidden_states

    def forward_chunk(self, mels, state: _MLXCausalAudioState | None = None):
        mx = self.mx
        if state is None:
            state = self.init_state()
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, frames, n_mels]")
        if int(mels.shape[-1]) != int(self.config.num_mel_bins):
            raise ValueError(
                f"expected {self.config.num_mel_bins} mel bins, got {mels.shape[-1]}"
            )
        state.last_input_frames = int(mels.shape[1])
        state.frames_seen += int(mels.shape[1])
        if state.mel_buffer is not None:
            buffer = mx.concatenate([state.mel_buffer, mels], axis=1)
        else:
            buffer = mels

        consume_frames = self.block_frames if self.block_frames > 0 else self.chunk_frames
        ready_frames = (int(buffer.shape[1]) // consume_frames) * consume_frames
        ready = buffer[:, :ready_frames, :]
        state.mel_buffer = buffer[:, ready_frames:, :]
        state.pending_frames = int(state.mel_buffer.shape[1])
        state.last_recomputed_frames = int(ready.shape[1])
        if ready.shape[1] == 0:
            return (
                mx.zeros((mels.shape[0], 0, self.config.output_dim), dtype=mels.dtype),
                state,
            )

        if self.block_frames > 0:
            outputs = []
            for start in range(0, int(ready.shape[1]), self.block_frames):
                outputs.append(
                    self._encode_ready_mels(
                        ready[:, start : start + self.block_frames, :],
                        state,
                    )
                )
            hidden = mx.concatenate(outputs, axis=1)
        else:
            hidden = self._encode_ready_mels(ready, state)
        return hidden, state

    def flush_pending(self, state: _MLXCausalAudioState):
        mx = self.mx
        if state.mel_buffer is None or state.mel_buffer.shape[1] == 0:
            return self.empty_features()[None, :, :], state
        ready_frames = (int(state.mel_buffer.shape[1]) // self.chunk_frames) * self.chunk_frames
        if ready_frames == 0:
            state.mel_buffer = None
            state.pending_frames = 0
            return self.empty_features()[None, :, :], state
        ready = state.mel_buffer[:, :ready_frames, :]
        state.mel_buffer = None
        state.pending_frames = 0
        state.last_recomputed_frames = int(ready.shape[1])
        hidden = self._encode_ready_mels(ready, state)
        return hidden, state


def _resolve_tower_checkpoint(reference: str) -> Path:
    path = Path(reference).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        from .model_paths import resolve_model_path

        path = resolve_model_path(reference)
        if path.is_file():
            return path
    for pattern in ("*.safetensors", "*.pt"):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"no .safetensors or .pt Qwen3 causal tower checkpoint found at {reference!r}"
    )


def _load_mlx_tower_checkpoint(audio_tower, checkpoint_path: Path, dtype) -> dict:
    import mlx.core as mx

    checkpoint_path = Path(checkpoint_path)
    metadata: dict = {}
    if checkpoint_path.suffix == ".safetensors":
        from safetensors import safe_open

        weights = {}
        with safe_open(str(checkpoint_path), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                weights[key] = mx.array(tensor.detach().cpu().numpy())
    else:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "tower_state_dict" in payload:
            state_dict = payload["tower_state_dict"]
            metadata = {
                key: payload.get(key)
                for key in ("step", "gate_wer", "model_id")
                if key in payload
            }
        else:
            state_dict = payload
        weights = {
            key: mx.array(tensor.detach().cpu().numpy())
            for key, tensor in state_dict.items()
        }

    converted = {}
    for key, value in weights.items():
        if key.startswith("audio_tower."):
            key = key[len("audio_tower.") :]
        if key.startswith("audio_encoder.audio_tower."):
            key = key[len("audio_encoder.audio_tower.") :]
        if "conv2d" in key and key.endswith(".weight") and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        if value.dtype != dtype:
            value = value.astype(dtype)
        converted[key] = value
    audio_tower.load_weights(list(converted.items()), strict=True)
    mx.eval(audio_tower.parameters())
    return metadata


class Qwen3VLLMMetalASR:
    """Model holder for vllm-metal Qwen3-ASR."""

    sep = ""
    SAMPLING_RATE = 16_000
    backend_choice = "qwen3-vllm-metal"

    def __init__(self, logfile=sys.stderr, **kwargs):
        _ensure_supported_platform()

        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None
        self.tokenizer = None
        self.audio_backend = _resolve_audio_backend(kwargs)
        self.holdback_words = int(
            kwargs.get("holdback_words")
            if kwargs.get("holdback_words") is not None
            else Qwen3VLLMMetalOnlineProcessor._HOLDBACK_WORDS
        )
        if self.holdback_words < 0:
            raise ValueError("holdback_words must be >= 0")
        self.trim_sentence_buffer = bool(kwargs.get("trim_sentence_buffer", True))
        self.min_chunk_size = float(
            kwargs.get("min_chunk_size")
            if kwargs.get("min_chunk_size") is not None
            else 1.0
        )
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size must be >= 0")
        max_decode_tokens = kwargs.get("max_tokens")
        self.max_decode_tokens = (
            int(max_decode_tokens) if max_decode_tokens is not None else 1024
        )
        if self.max_decode_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

        self._worker = _Qwen3MetalWorker()
        self._worker.call(self._init_model, kwargs)

    def _init_model(self, kwargs: dict) -> None:
        try:
            import mlx.core as mx
            from vllm_metal.stt.loader import load_model
            from vllm_metal.stt.qwen3_asr.adapter import Qwen3ASRRuntimeAdapter
            from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber
        except ImportError as exc:
            raise _missing_dependency_error(exc) from exc

        self._post_process_output = Qwen3ASRTranscriber.post_process_output

        model_path = _resolve_model_path(kwargs)
        dtype = _resolve_mlx_dtype(mx, kwargs)

        t0 = time.time()
        logger.info("Loading Qwen3 vllm-metal model '%s' ...", model_path)
        self.model = load_model(model_path, dtype=dtype)
        self.adapter = Qwen3ASRRuntimeAdapter(self.model, model_path)
        self.causal_encoder = None
        if self.audio_backend == "causal":
            checkpoint_ref = (
                kwargs.get("qwen3_vllm_metal_tower_checkpoint")
                or DEFAULT_QWEN3_VLLM_METAL_CAUSAL_TOWER
            )
            checkpoint_path = _resolve_tower_checkpoint(str(checkpoint_ref))
            metadata = _load_mlx_tower_checkpoint(
                self.model.audio_tower,
                checkpoint_path,
                dtype,
            )
            self.causal_encoder = _Qwen3MetalCausalAudioEncoder(
                self.model.audio_tower,
                left_context_sec=float(
                    kwargs.get("qwen3_vllm_metal_left_context_sec", 15.0)
                ),
                block_frames=int(kwargs.get("qwen3_vllm_metal_block_frames", 192)),
                block_bidirectional=True,
            )
            logger.info(
                "qwen3-vllm-metal causal: loaded tower checkpoint %s%s",
                checkpoint_path,
                f" (step {metadata['step']})" if metadata.get("step") else "",
            )
        self.adapter.warm_up()
        self.tokenizer = self.adapter.transcriber.tokenizer
        logger.info("Qwen3 vllm-metal model loaded in %.2fs", time.time() - t0)

    def _build_prompt_token_parts(self) -> tuple[list[int], list[int]]:
        tokenizer = self.tokenizer

        head = []
        head.extend(tokenizer.encode("<|im_start|>", add_special_tokens=False))
        head.extend(tokenizer.encode("system\n", add_special_tokens=False))
        head.extend(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        head.extend(tokenizer.encode("<|im_start|>", add_special_tokens=False))
        head.extend(tokenizer.encode("user\n", add_special_tokens=False))
        head.append(_token_id(tokenizer, "<|audio_start|>"))

        tail = []
        tail.append(_token_id(tokenizer, "<|audio_end|>"))
        tail.extend(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        tail.extend(tokenizer.encode("<|im_start|>", add_special_tokens=False))
        tail.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
        return head, tail

    def _build_prompt_token_ids(self, n_audio_tokens: int) -> list[int]:
        head, tail = self._build_prompt_token_parts()
        return head + [self.model.config.audio_token_id] * n_audio_tokens + tail

    def _decode_output_tokens(self, output_tokens: list[int]) -> str:
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return self._post_process_output(text).strip()

    def _extract_asr_output_tokens(self, raw_tokens: list[int]) -> list[int]:
        output_tokens = self.adapter._extract_asr_text_tokens(list(raw_tokens))
        output_tokens.append(self.adapter.eot_token)
        return output_tokens

    def _transcribe_text(self, audio: np.ndarray) -> str:
        """Transcribe raw 16 kHz mono float PCM and return cleaned text."""
        if len(audio) < 400:
            return ""

        try:
            from vllm_metal.stt.audio import log_mel_spectrogram
        except ImportError as exc:
            raise _missing_dependency_error(exc) from exc

        mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=128)
        audio_features = self.adapter.extract_audio_features(mel)
        n_audio_tokens = int(audio_features.shape[0])
        prompt_ids = self._build_prompt_token_ids(n_audio_tokens)
        output_tokens = self.adapter.decode_tokens(audio_features, prompt_ids)
        return self._decode_output_tokens(output_tokens)

    def transcribe_text(self, audio: np.ndarray) -> str:
        return self._worker.call(self._transcribe_text, audio)

    def new_causal_state(self) -> _MLXCausalAudioState:
        if self.causal_encoder is None:
            raise RuntimeError("qwen3-vllm-metal causal encoder is not initialized")
        return self._worker.call(self.causal_encoder.init_state)

    def empty_causal_features(self):
        if self.causal_encoder is None:
            raise RuntimeError("qwen3-vllm-metal causal encoder is not initialized")
        return self._worker.call(self.causal_encoder.empty_features)

    def new_causal_decoder_state(self) -> _MLXDecoderRollingState:
        if self.causal_encoder is None:
            raise RuntimeError("qwen3-vllm-metal causal encoder is not initialized")
        return _MLXDecoderRollingState()

    def causal_decode_from_audio(
        self,
        audio: np.ndarray,
        *,
        mel_frames_encoded: int,
        audio_features,
        state: _MLXCausalAudioState,
        decoder_state: _MLXDecoderRollingState | None = None,
        flush: bool = False,
    ):
        return self._worker.call(
            self._causal_decode_from_audio,
            audio,
            mel_frames_encoded,
            audio_features,
            state,
            decoder_state,
            flush,
        )

    def _causal_decode_from_audio(
        self,
        audio: np.ndarray,
        mel_frames_encoded: int,
        audio_features,
        state: _MLXCausalAudioState,
        decoder_state: _MLXDecoderRollingState | None,
        flush: bool,
    ):
        if self.causal_encoder is None:
            raise RuntimeError("qwen3-vllm-metal causal encoder is not initialized")
        if decoder_state is None:
            decoder_state = _MLXDecoderRollingState()
        if len(audio) < 400:
            return "", mel_frames_encoded, audio_features, state, decoder_state

        try:
            from vllm_metal.stt.audio import log_mel_spectrogram
        except ImportError as exc:
            raise _missing_dependency_error(exc) from exc

        import mlx.core as mx

        mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=128)
        total_frames = int(mel.shape[1])
        if total_frames > mel_frames_encoded:
            delta = mel[:, mel_frames_encoded:total_frames].T[None, :, :]
            hidden, state = self.causal_encoder.forward_chunk(delta, state)
            if hidden.shape[1] > 0:
                audio_features = mx.concatenate([audio_features, hidden[0]], axis=0)
            mel_frames_encoded = total_frames
        if flush:
            hidden, state = self.causal_encoder.flush_pending(state)
            if hidden.shape[1] > 0:
                audio_features = mx.concatenate([audio_features, hidden[0]], axis=0)
        mx.eval(audio_features)
        if int(audio_features.shape[0]) == 0:
            return "", mel_frames_encoded, audio_features, state, decoder_state

        if decoder_state.disabled:
            prompt_ids = self._build_prompt_token_ids(int(audio_features.shape[0]))
            output_tokens = self.adapter.decode_tokens(audio_features, prompt_ids)
        else:
            raw_tokens, decoder_state = self._rolling_decode_raw_tokens(
                audio_features, decoder_state
            )
            output_tokens = self._extract_asr_output_tokens(raw_tokens)
        text = self._decode_output_tokens(output_tokens)
        return text, mel_frames_encoded, audio_features, state, decoder_state

    def _embed_token_ids(self, token_ids: list[int]):
        import mlx.core as mx

        if not token_ids:
            hidden = int(self.model.config.text_config.hidden_size)
            return mx.zeros((1, 0, hidden), dtype=self.model.dtype)
        ids = mx.array([token_ids], dtype=mx.int32)
        return self.model.language_model.embed(ids)

    def _eval_mlx_decoder_cache(self, cache: list[Any] | None) -> None:
        if not cache:
            return
        import mlx.core as mx

        arrays = []
        for layer_cache in cache:
            if layer_cache is None:
                continue
            arrays.extend(layer_cache)
        if arrays:
            mx.eval(*arrays)

    def _next_token_id(self, logits) -> int:
        import mlx.core as mx

        return int(mx.argmax(logits[:, -1, :], axis=-1).item())

    def _rolling_decode_raw_tokens(
        self,
        audio_features,
        decoder_state: _MLXDecoderRollingState,
    ) -> tuple[list[int], _MLXDecoderRollingState]:
        import mlx.core as mx

        if self.max_decode_tokens <= 0:
            decoder_state.last_raw_tokens = []
            decoder_state.last_stats = {"decoder_path": "rolling", "decode_steps": 0}
            return [], decoder_state

        audio_steps = int(audio_features.shape[0])
        if audio_steps == 0:
            decoder_state.last_raw_tokens = []
            return [], decoder_state

        head_ids, tail_ids = self._build_prompt_token_parts()
        eos_token = int(self.model.config.eos_token_id)
        embed_dtype = self.model.language_model.embed_tokens.weight.dtype
        draft = [int(token_id) for token_id in decoder_state.last_raw_tokens]
        if eos_token in draft:
            draft = draft[: draft.index(eos_token)]
        draft = draft[: self.max_decode_tokens]
        draft_len = len(draft)

        cache_valid = (
            decoder_state.cache is not None
            and decoder_state.head_token_ids == tuple(head_ids)
            and 0 <= decoder_state.audio_steps <= audio_steps
            and _mlx_decoder_cache_seq_length(decoder_state.cache)
            == decoder_state.head_len + decoder_state.audio_steps
        )

        parts = []
        if cache_valid:
            cache = decoder_state.cache
            delta = audio_features[decoder_state.audio_steps : audio_steps]
            if int(delta.shape[0]) > 0:
                parts.append(delta[None, :, :].astype(embed_dtype))
        else:
            cache = None
            if head_ids:
                parts.append(self._embed_token_ids(head_ids))
            parts.append(audio_features[None, :, :].astype(embed_dtype))
        if tail_ids:
            parts.append(self._embed_token_ids(tail_ids))
        if draft:
            parts.append(self._embed_token_ids(draft))

        if not parts:
            prompt_ids = self._build_prompt_token_ids(audio_steps)
            raw_tokens = self.adapter.transcriber.greedy_decode_tokens(
                audio_features, prompt_ids, max_tokens=self.max_decode_tokens
            )
            decoder_state.disabled = True
            decoder_state.last_raw_tokens = [int(token_id) for token_id in raw_tokens]
            decoder_state.last_stats = {"decoder_path": "full"}
            return decoder_state.last_raw_tokens, decoder_state

        block = mx.concatenate(parts, axis=1)
        logits, past = self.model.language_model.forward_embeds(block, cache)
        mx.eval(logits)

        verify_logits = logits[:, -(draft_len + 1) :, :]
        prefix_len = len(head_ids) + audio_steps + len(tail_ids)
        pick_ids: list[int] = []
        for index in range(draft_len):
            pick_ids.append(self._next_token_id(verify_logits[:, index : index + 1, :]))

        accepted = draft_len
        corrected: int | None = None
        for index, (pick, wanted) in enumerate(zip(pick_ids, draft)):
            if int(pick) != int(wanted):
                accepted = index
                corrected = int(pick)
                break

        generated_ids = draft[:accepted]
        sequential_steps = 0
        if corrected is not None:
            past = _crop_mlx_decoder_cache(past, prefix_len + accepted)
            if corrected != eos_token:
                generated_ids.append(corrected)
                if len(generated_ids) < self.max_decode_tokens:
                    token_input = mx.array([[corrected]], dtype=mx.int32)
                    logits, past = self.model.decode_step(token_input, past)
                    mx.eval(logits)
                    generated_ids, past, sequential_steps = self._rolling_sequential_tail(
                        logits,
                        past,
                        generated_ids,
                        eos_token=eos_token,
                        max_new_tokens=self.max_decode_tokens - len(generated_ids),
                    )
        elif draft_len < self.max_decode_tokens:
            logits = verify_logits[:, draft_len : draft_len + 1, :]
            generated_ids, past, sequential_steps = self._rolling_sequential_tail(
                logits,
                past,
                generated_ids,
                eos_token=eos_token,
                max_new_tokens=self.max_decode_tokens - draft_len,
            )

        audio_prefix_len = len(head_ids) + audio_steps
        decoder_state.cache = _crop_mlx_decoder_cache(past, audio_prefix_len)
        decoder_state.head_token_ids = tuple(head_ids)
        decoder_state.head_len = len(head_ids)
        decoder_state.audio_steps = audio_steps
        decoder_state.last_raw_tokens = [int(token_id) for token_id in generated_ids]
        decoder_state.last_stats = {
            "decoder_path": "rolling+draft" if draft_len else "rolling",
            "decoder_rebuilt": not cache_valid,
            "draft_tokens": draft_len,
            "draft_accepted": accepted,
            "draft_all_accepted": bool(draft_len) and accepted == draft_len,
            "decode_steps": sequential_steps + (1 if corrected is not None else 0),
            "prefill_positions": int(block.shape[1]),
        }
        self._eval_mlx_decoder_cache(decoder_state.cache)
        return decoder_state.last_raw_tokens, decoder_state

    def _rolling_sequential_tail(
        self,
        logits,
        past,
        generated_ids: list[int],
        *,
        eos_token: int,
        max_new_tokens: int,
    ) -> tuple[list[int], list[Any], int]:
        import mlx.core as mx

        steps = 0
        for step in range(max_new_tokens):
            next_token = self._next_token_id(logits)
            if next_token == eos_token:
                break
            generated_ids.append(next_token)
            steps += 1
            if step == max_new_tokens - 1:
                break
            token_input = mx.array([[next_token]], dtype=mx.int32)
            logits, past = self.model.decode_step(token_input, past)
            mx.eval(logits)
        return generated_ids, past, steps

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> str:
        return self.transcribe_text(audio)

    def use_vad(self):
        return False


class Qwen3VLLMMetalOnlineProcessor:
    """Batch processor committing the current hypothesis except trailing words."""

    SAMPLING_RATE = 16_000
    _HOLDBACK_WORDS = 2

    def __init__(self, asr: Qwen3VLLMMetalASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.holdback_words = getattr(asr, "holdback_words", self._HOLDBACK_WORDS)
        self.trim_sentence_buffer = getattr(asr, "trim_sentence_buffer", True)
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._buffer_time_offset = 0.0
        self._n_committed_words = 0
        self._current_words: list[str] = []
        self._current_text = ""
        self._samples_since_last_inference = 0
        self._min_new_samples = max(
            1,
            int(getattr(asr, "min_chunk_size", 1.0) * self.SAMPLING_RATE),
        )

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio)
        self._samples_since_last_inference += len(audio)

    def _transcribe_words(self) -> list[str]:
        text = self.asr.transcribe_text(self.audio_buffer)
        self._current_text = text
        words = text.split()
        self._current_words = words
        return words

    def _time_for_word(self, word_idx: int, n_words_total: int) -> Tuple[float, float]:
        duration = max(len(self.audio_buffer) / self.SAMPLING_RATE, 0.001)
        n_total = max(n_words_total, 1)
        start = self._buffer_time_offset + (word_idx / n_total) * duration
        end = self._buffer_time_offset + ((word_idx + 1) / n_total) * duration
        return start, end

    def _tokens_for_range(
        self,
        words: list[str],
        start_idx: int,
        end_idx: int,
    ) -> List[ASRToken]:
        tokens: List[ASRToken] = []
        n_total = len(words)
        for idx in range(start_idx, end_idx):
            start, end = self._time_for_word(idx, n_total)
            text = words[idx] if idx == 0 else " " + words[idx]
            tokens.append(ASRToken(start=start, end=end, text=text))
        return tokens

    @staticmethod
    def _sentence_boundary_before(words: list[str], committed_upto: int) -> int | None:
        for idx in range(min(committed_upto, len(words)) - 1, -1, -1):
            if words[idx].rstrip().endswith(_SENTENCE_ENDINGS):
                return idx
        return None

    def _trim_committed_sentence(self, words: list[str]) -> None:
        if not self.trim_sentence_buffer:
            return

        boundary_idx = self._sentence_boundary_before(words, self._n_committed_words)
        if boundary_idx is None:
            return

        _, trim_end = self._time_for_word(boundary_idx, len(words))
        trim_samples = int((trim_end - self._buffer_time_offset) * self.SAMPLING_RATE)
        trim_samples = min(max(trim_samples, 0), len(self.audio_buffer))
        if trim_samples <= 0:
            return

        trimmed_words = boundary_idx + 1
        self.audio_buffer = self.audio_buffer[trim_samples:]
        self._buffer_time_offset += trim_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(
            self._samples_since_last_inference,
            len(self.audio_buffer),
        )
        self._n_committed_words = max(0, self._n_committed_words - trimmed_words)
        self._current_words = words[trimmed_words:]
        self._current_text = " ".join(self._current_words)

    def _commit_available(self, flush: bool = False) -> List[ASRToken]:
        words = self._transcribe_words()
        if flush:
            commit_upto = len(words)
        else:
            commit_upto = max(len(words) - self.holdback_words, 0)
        if commit_upto <= self._n_committed_words:
            return []

        tokens = self._tokens_for_range(words, self._n_committed_words, commit_upto)
        self._n_committed_words = commit_upto
        self._trim_committed_sentence(words)
        return tokens

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            if (
                not is_last
                and self._samples_since_last_inference < self._min_new_samples
            ):
                return [], self.end
            self._samples_since_last_inference = 0
            return self._commit_available(flush=is_last), self.end
        except Exception as e:
            logger.warning("[qwen3-vllm-metal] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        if not self._current_words or self._n_committed_words >= len(self._current_words):
            return Transcript(start=None, end=None, text="")

        words = self._current_words[self._n_committed_words:]
        start, _ = self._time_for_word(self._n_committed_words, len(self._current_words))
        _, end = self._time_for_word(len(self._current_words) - 1, len(self._current_words))
        return Transcript(start=start, end=end, text=" ".join(words))

    def _reset_for_next_utterance(self):
        self._buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._n_committed_words = 0
        self._current_words = []
        self._current_text = ""

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        words = self._commit_available(flush=True)
        logger.info("[qwen3-vllm-metal] start_silence: flushed %d words", len(words))
        self._reset_for_next_utterance()
        return words, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._buffer_time_offset += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        return None

    def finish(self) -> Tuple[List[ASRToken], float]:
        words = self._commit_available(flush=True)
        logger.info("[qwen3-vllm-metal] finish: flushed %d words", len(words))
        return words, self.end


class Qwen3VLLMMetalCausalOnlineProcessor(Qwen3VLLMMetalOnlineProcessor):
    """vllm-metal processor using append-only causal MLX audio encoding."""

    def __init__(self, asr: Qwen3VLLMMetalASR, logfile=sys.stderr):
        super().__init__(asr, logfile=logfile)
        self._causal_state = asr.new_causal_state()
        self._audio_features = asr.empty_causal_features()
        self._decoder_state = asr.new_causal_decoder_state()
        self._mel_frames_encoded = 0

    def _transcribe_words(self, flush: bool = False) -> list[str]:
        (
            text,
            self._mel_frames_encoded,
            self._audio_features,
            self._causal_state,
            self._decoder_state,
        ) = self.asr.causal_decode_from_audio(
            self.audio_buffer,
            mel_frames_encoded=self._mel_frames_encoded,
            audio_features=self._audio_features,
            state=self._causal_state,
            decoder_state=self._decoder_state,
            flush=flush,
        )
        self._current_text = text
        words = text.split()
        self._current_words = words
        return words

    def _commit_available(self, flush: bool = False) -> List[ASRToken]:
        words = self._transcribe_words(flush=flush)
        if flush:
            commit_upto = len(words)
        else:
            commit_upto = max(len(words) - self.holdback_words, 0)
        if commit_upto <= self._n_committed_words:
            return []

        tokens = self._tokens_for_range(words, self._n_committed_words, commit_upto)
        self._n_committed_words = commit_upto
        self._trim_committed_sentence(words)
        return tokens

    def _trim_committed_sentence(self, words: list[str]) -> None:
        if not self.trim_sentence_buffer:
            return

        boundary_idx = self._sentence_boundary_before(words, self._n_committed_words)
        if boundary_idx is None:
            return

        _, trim_end = self._time_for_word(boundary_idx, len(words))
        trim_samples = int((trim_end - self._buffer_time_offset) * self.SAMPLING_RATE)
        trim_samples = min(max(trim_samples, 0), len(self.audio_buffer))
        if trim_samples <= 0:
            return

        trimmed_words = boundary_idx + 1
        self.audio_buffer = self.audio_buffer[trim_samples:]
        self._buffer_time_offset += trim_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(
            self._samples_since_last_inference,
            len(self.audio_buffer),
        )
        self._n_committed_words = max(0, self._n_committed_words - trimmed_words)
        self._current_words = words[trimmed_words:]
        self._current_text = " ".join(self._current_words)
        self._causal_state = self.asr.new_causal_state()
        self._audio_features = self.asr.empty_causal_features()
        self._decoder_state = self.asr.new_causal_decoder_state()
        self._mel_frames_encoded = 0

    def _reset_for_next_utterance(self):
        self._buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._n_committed_words = 0
        self._current_words = []
        self._current_text = ""
        self._causal_state = self.asr.new_causal_state()
        self._audio_features = self.asr.empty_causal_features()
        self._decoder_state = self.asr.new_causal_decoder_state()
        self._mel_frames_encoded = 0
