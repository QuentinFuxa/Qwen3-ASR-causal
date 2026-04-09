"""
Qwen3-ASR backend using vLLM's in-process GPU runtime.

This backend does not use vLLM's HTTP or WebSocket APIs. It keeps one vLLM
engine alive for Qwen3-ASR transcription and another one for Qwen3-ForcedAligner
timestamp prediction. Streaming is implemented by re-transcribing the current
audio buffer and committing only aligned words outside the last 250 ms.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
import uuid
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from importlib import import_module
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import ASRToken, Transcript

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_VLLM_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_QWEN3_VLLM_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_QWEN3_VLLM_CAUSAL_TOWER = "qfuxa/qwen3-asr-0.6b-streaming"

QWEN3_VLLM_MODEL_MAPPING = {
    "base": "Qwen/Qwen3-ASR-0.6B",
    "tiny": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "medium": DEFAULT_QWEN3_VLLM_MODEL,
    "large": DEFAULT_QWEN3_VLLM_MODEL,
    "large-v3": DEFAULT_QWEN3_VLLM_MODEL,
    "qwen3-asr-1.7b": DEFAULT_QWEN3_VLLM_MODEL,
    "qwen3-1.7b": DEFAULT_QWEN3_VLLM_MODEL,
    "1.7b": DEFAULT_QWEN3_VLLM_MODEL,
}

WHISPER_TO_QWEN3_LANGUAGE = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
QWEN3_TO_WHISPER_LANGUAGE = {v: k for k, v in WHISPER_TO_QWEN3_LANGUAGE.items()}

_ASR_TEXT_TAG = "<asr_text>"
_LANG_RE = re.compile(r"(?:^|\s)language\s+([A-Za-z][A-Za-z -]*)", re.IGNORECASE)
_EOT_RE = re.compile(r"<\|endoftext\|>$")
_VLLM_LIVE_PREFIX_MARKER = "\x1fwlk_live_prefix="


class _VLLMLiveCompactPromptEmbeds:
    """Absolute-position prompt_embeds view carrying only an uncomputed suffix."""

    def __init__(self, values, *, offset: int, full_shape: tuple[int, int]):
        self.values = values
        self.offset = int(offset)
        self._shape = tuple(int(v) for v in full_shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self):
        return getattr(self.values, "dtype", None)

    @property
    def device(self):
        return getattr(self.values, "device", None)

    def _zeros(self, length: int):
        import torch

        return torch.zeros(
            (int(length), self._shape[-1]),
            dtype=self.values.dtype,
            device=self.values.device,
        )

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row_index, *rest = index
            rows = self[row_index]
            return rows[(slice(None),) + tuple(rest)]
        if isinstance(index, slice):
            start, stop, step = index.indices(self._shape[0])
            if step != 1:
                return self._materialize()[index]
            if stop <= start:
                return self.values[:0]
            value_start = max(start, self.offset) - self.offset
            value_stop = max(stop, self.offset) - self.offset
            if start >= self.offset:
                return self.values[value_start:value_stop]
            if stop <= self.offset:
                return self._zeros(stop - start)
            return __import__("torch").cat(
                [
                    self._zeros(self.offset - start),
                    self.values[:value_stop],
                ],
                dim=0,
            )
        if isinstance(index, int):
            if index < 0:
                index += self._shape[0]
            if index < self.offset:
                return self._zeros(1)[0]
            return self.values[index - self.offset]
        return self._materialize()[index]

    def _materialize(self):
        if self.offset <= 0:
            return self.values
        return __import__("torch").cat(
            [self._zeros(self.offset), self.values],
            dim=0,
        )

    def __array__(self, dtype=None):
        array = self._materialize().detach().cpu().numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array


@dataclass
class _AlignedWord:
    text: str
    start: float
    end: float


def _missing_dependency_error(reason: str) -> ImportError:
    return ImportError(
        "qwen3-vllm requires vLLM with Qwen3-ASR ForcedAligner support. "
        "Install it on a CUDA/Linux host with `uv sync --extra qwen3-vllm` "
        "in an environment separate from `cu129`, or with: "
        "pip install 'qwen3-asr-causal[vllm]'. "
        f"Details: {reason}"
    )


def _load_vllm_runtime():
    try:
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.model_executor.models.qwen3_asr_forced_aligner import (
            Qwen3ASRForcedAlignerForTokenClassification,  # noqa: F401
        )
    except ImportError as exc:
        raise _missing_dependency_error(str(exc)) from exc
    return LLM, SamplingParams, TokensPrompt, AutoConfig


def _patch_vllm_mrope_prompt_embeds() -> None:
    """Allow Qwen3-ASR causal prompt embeds through vLLM V1.

    vLLM 0.23 accepts prompt_embeds, but its V1 Qwen3-ASR runner asserts that
    prompt_token_ids are present before initializing M-RoPE positions. Our
    causal path bypasses vLLM's audio placeholder and supplies the exact
    expanded embeddings; with no multimodal features, Qwen3-ASR's own M-RoPE
    implementation falls back to linear positions. Mirror that fallback here.

    Qwen3-ASR is also a multimodal model, so the V1 runner checks the generic
    multimodal branch before the prompt-embeds branch and can silently ignore
    direct EmbedsPrompt rows. When prompt embeds are present, let the dedicated
    prompt-embeds branch run and keep input_ids available for Qwen3-ASR.
    """
    try:
        import inspect
        import textwrap

        import torch
        from vllm.v1.worker import gpu_model_runner
    except Exception as exc:  # pragma: no cover - version-dependent fallback
        logger.debug("[qwen3-vllm] could not patch vLLM prompt_embeds M-RoPE: %s", exc)
        return

    runner_cls = getattr(gpu_model_runner, "GPUModelRunner", None)
    if runner_cls is None:
        return

    if not getattr(runner_cls, "_wlk_prompt_embeds_mrope_patch", False):
        original = runner_cls._init_mrope_positions

        def patched_mrope(self, req_state):
            prompt_embeds = getattr(req_state, "prompt_embeds", None)
            mm_features = getattr(req_state, "mm_features", None)
            if (
                req_state.prompt_token_ids is None
                and prompt_embeds is not None
                and not mm_features
            ):
                seq_len = int(prompt_embeds.shape[-2])
                positions = torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(3, -1)
                req_state.mrope_positions = positions.clone()
                req_state.mrope_position_delta = 0
                return
            return original(self, req_state)

        runner_cls._init_mrope_positions = patched_mrope
        runner_cls._wlk_prompt_embeds_mrope_patch = True

    if getattr(runner_cls, "_wlk_prompt_embeds_runner_patch", False):
        return

    old_condition = "    if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:\n"
    new_condition = (
        "    if (\n"
        "        self.supports_mm_inputs\n"
        "        and is_first_rank\n"
        "        and not is_encoder_decoder\n"
        "        and not self.input_batch.req_prompt_embeds\n"
        "    ):\n"
    )
    old_input_ids = (
        "        inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]\n"
        "        model_kwargs = self._init_model_kwargs()\n"
        "        input_ids = None\n"
    )
    new_input_ids = (
        "        inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]\n"
        "        model_kwargs = self._init_model_kwargs()\n"
        "        input_ids = self.input_ids.gpu[:num_input_tokens]\n"
    )
    for method_name in ("_preprocess", "execute_model"):
        method = getattr(runner_cls, method_name, None)
        if method is None:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(method))
            source_file = inspect.getsourcefile(method) or "<vllm>"
        except Exception as exc:  # pragma: no cover - version-dependent fallback
            logger.debug("[qwen3-vllm] could not inspect vLLM %s: %s", method_name, exc)
            continue
        if old_condition not in source or old_input_ids not in source:
            continue

        source = source.replace(old_condition, new_condition, 1).replace(
            old_input_ids, new_input_ids, 1
        )
        namespace = {}
        try:
            exec(compile(source, source_file, "exec"), gpu_model_runner.__dict__, namespace)
        except Exception as exc:  # pragma: no cover - version-dependent fallback
            logger.debug("[qwen3-vllm] could not patch vLLM %s: %s", method_name, exc)
            continue
        setattr(runner_cls, method_name, namespace[method_name])
        runner_cls._wlk_prompt_embeds_runner_patch = True
        return

    logger.debug("[qwen3-vllm] vLLM prompt_embeds runner patch not applied")


def _qwen3_audio_feature_length_for_output_steps(output_steps: int) -> int:
    """Synthetic pre-CNN length whose Qwen3-ASR output length is output_steps."""
    steps = int(output_steps)
    if steps <= 0:
        raise ValueError("output_steps must be positive")
    full_groups = (steps - 1) // 13
    remainder = steps - full_groups * 13
    return full_groups * 100 + 1 + (remainder - 1) * 8


def _patch_vllm_qwen3_asr_audio_embeds() -> bool:
    """Teach vLLM's Qwen3-ASR model to accept precomputed audio embeddings.

    vLLM's Qwen2 audio backend already supports an ``audio_embeds`` dict path.
    Qwen3-ASR only exposes raw/processed audio features, so the causal CUDA path
    otherwise has to use direct ``prompt_embeds`` and falls into V1/eager. This
    runtime patch adds the same multimodal embedding contract to Qwen3-ASR:

    ``multi_modal_data={"audio": {"audio_embeds": ..., "audio_embed_lengths": ...}}``

    The request stays token/multimodal based, so vLLM can use its normal compiled
    multimodal runner and prefix cache.
    """
    patched_any = False
    module_names = ("vllm.model_executor.models.qwen3_asr",)
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional package/version
            logger.debug("[qwen3-vllm] could not import %s: %s", module_name, exc)
            continue
        if getattr(module, "_wlk_audio_embeds_patch", False):
            patched_any = True
            continue

        parser_cls = getattr(module, "Qwen3ASRMultiModalDataParser", None)
        processor_cls = getattr(module, "Qwen3ASRMultiModalProcessor", None)
        model_cls = getattr(module, "Qwen3ASRForConditionalGeneration", None)
        field_config_fn = getattr(module, "_qwen3asr_field_config", None)
        dict_embedding_items = getattr(module, "DictEmbeddingItems", None)
        field_config_cls = getattr(module, "MultiModalFieldConfig", None)
        batch_feature_cls = getattr(module, "BatchFeature", None)
        if not all(
            (
                parser_cls,
                processor_cls,
                model_cls,
                field_config_fn,
                dict_embedding_items,
                field_config_cls,
                batch_feature_cls,
            )
        ):
            continue

        def patched_field_config(hf_inputs, _orig=field_config_fn, _field=field_config_cls):
            fields = dict(_orig(hf_inputs))
            if "audio_embeds" in hf_inputs:
                audio_embed_lengths = hf_inputs.get("audio_embed_lengths")
                if audio_embed_lengths is not None:
                    fields["audio_embeds"] = _field.flat_from_sizes(
                        "audio",
                        audio_embed_lengths,
                        dim=0,
                    )
                    fields["audio_embed_lengths"] = _field.batched("audio")
                else:
                    fields["audio_embeds"] = _field.batched("audio")
            return fields

        module._qwen3asr_field_config = patched_field_config

        original_parse_audio_data = parser_cls._parse_audio_data

        def patched_parse_audio_data(
            self,
            data,
            _orig=original_parse_audio_data,
            _module=module,
            _items=dict_embedding_items,
        ):
            if isinstance(data, dict) and "audio_embeds" in data:
                required_fields = {"audio_embeds"}
                if "audio_embed_lengths" in data:
                    required_fields.add("audio_embed_lengths")
                if "audio_feature_lengths" in data:
                    required_fields.add("audio_feature_lengths")
                return _items(
                    data,
                    modality="audio",
                    required_fields=required_fields,
                    fields_factory=_module._qwen3asr_field_config,
                )
            return _orig(self, data)

        parser_cls._parse_audio_data = patched_parse_audio_data

        original_call_hf_processor = processor_cls._call_hf_processor

        def patched_call_hf_processor(
            self,
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
            _orig=original_call_hf_processor,
            _batch_feature=batch_feature_cls,
        ):
            # Qwen3-ASR's HF processor expands audio tokens only when raw audio
            # is present. For precomputed embeddings, tokenize the prompt text
            # directly and let vLLM apply prompt replacements from passthrough
            # audio_embed/audio_length tensors, matching Qwen2-audio behavior.
            if not mm_data.get("audio") and not mm_data.get("audios"):
                prompt_ids = self.info.get_tokenizer().encode(prompt)
                prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
                return _batch_feature(dict(input_ids=[prompt_ids]), tensor_type="pt")
            return _orig(self, prompt, mm_data, mm_kwargs, tok_kwargs)

        processor_cls._call_hf_processor = patched_call_hf_processor

        original_parse_audio_input = model_cls._parse_and_validate_audio_input

        def patched_parse_audio_input(self, **kwargs):
            audio_embeds = kwargs.pop("audio_embeds", None)
            if audio_embeds is not None:
                audio_embed_lengths = kwargs.pop("audio_embed_lengths", None)
                kwargs.pop("audio_feature_lengths", None)
                kwargs.pop("feature_attention_mask", None)
                return {
                    "type": "audio_embeds",
                    "audio_embeds": audio_embeds,
                    "audio_embed_lengths": audio_embed_lengths,
                }
            return original_parse_audio_input(self, **kwargs)

        model_cls._parse_and_validate_audio_input = patched_parse_audio_input

        original_parse_mm_inputs = model_cls._parse_and_validate_multimodal_inputs

        def patched_parse_mm_inputs(self, **kwargs):
            if kwargs.get("audio_embeds") is not None:
                return {"audio": self._parse_and_validate_audio_input(**kwargs)}
            return original_parse_mm_inputs(self, **kwargs)

        model_cls._parse_and_validate_multimodal_inputs = patched_parse_mm_inputs

        original_process_audio_input = model_cls._process_audio_input

        def patched_process_audio_input(self, audio_input, *args, **kwargs):
            if audio_input and audio_input.get("type") == "audio_embeds":
                audio_embeds = audio_input["audio_embeds"]
                audio_embed_lengths = audio_input.get("audio_embed_lengths")
                if (
                    audio_embed_lengths is not None
                    and hasattr(audio_embeds, "ndim")
                    and int(audio_embeds.ndim) == 2
                ):
                    lengths = [int(length) for length in audio_embed_lengths.tolist()]
                    return tuple(audio_embeds.split(lengths, dim=0))
                if hasattr(audio_embeds, "ndim") and int(audio_embeds.ndim) == 2:
                    return (audio_embeds,)
                return tuple(audio_embeds)
            return original_process_audio_input(self, audio_input, *args, **kwargs)

        model_cls._process_audio_input = patched_process_audio_input
        module._wlk_audio_embeds_patch = True
        patched_any = True

    return patched_any


def _vllm_prompt_embeds_prefix_cache_supported() -> bool:
    """Return whether this vLLM build hashes prompt embeds for prefix cache."""
    try:
        import inspect

        from vllm.v1.core import kv_cache_utils
    except Exception as exc:  # pragma: no cover - version-dependent fallback
        logger.debug("[qwen3-vllm] could not inspect vLLM prefix cache support: %s", exc)
        return False

    if not hasattr(kv_cache_utils, "_gen_prompt_embeds_extra_hash_keys"):
        return False
    try:
        source = inspect.getsource(kv_cache_utils.generate_block_hash_extra_keys)
    except Exception:  # pragma: no cover - source may be unavailable
        return True
    return "prompt_embeds" in source or "prompt_embeds_keys" in source


def _encode_vllm_live_cache_salt(
    cache_salt: str | None,
    *,
    prefix_keep_len: int,
    prompt_head_len: int | None = None,
    audio_steps: int | None = None,
) -> str:
    """Attach private live-append metadata without changing the base salt."""
    if prefix_keep_len < 0:
        raise ValueError("prefix_keep_len must be non-negative")
    suffix = str(int(prefix_keep_len))
    if prompt_head_len is not None and audio_steps is not None:
        suffix += f";h={int(prompt_head_len)};a={int(audio_steps)}"
    return f"{cache_salt or ''}{_VLLM_LIVE_PREFIX_MARKER}{suffix}"


def _decode_vllm_live_cache_salt(cache_salt: str | None) -> tuple[str | None, int | None]:
    """Return (clean_cache_salt, prefix_keep_len) for live-append prompts."""
    if not cache_salt or _VLLM_LIVE_PREFIX_MARKER not in cache_salt:
        return cache_salt, None
    base, raw_value = cache_salt.rsplit(_VLLM_LIVE_PREFIX_MARKER, 1)
    raw_prefix = raw_value.split(";", 1)[0]
    try:
        prefix_keep_len = int(raw_prefix)
    except ValueError:
        return cache_salt, None
    if prefix_keep_len < 0:
        return cache_salt, None
    return (base or None), prefix_keep_len


def _mark_vllm_live_prompt_embeds_request(request, cache_salt: str | None) -> None:
    if cache_salt and _VLLM_LIVE_PREFIX_MARKER in cache_salt:
        try:
            request._wlk_live_prompt_embeds_streaming = True
        except Exception:
            pass


def _decode_vllm_live_prompt_mask_metadata(
    cache_salt: str | None,
) -> tuple[int, int] | None:
    if not cache_salt or _VLLM_LIVE_PREFIX_MARKER not in cache_salt:
        return None
    _base, raw_value = cache_salt.rsplit(_VLLM_LIVE_PREFIX_MARKER, 1)
    values: dict[str, int] = {}
    for piece in raw_value.split(";")[1:]:
        if "=" not in piece:
            continue
        key, raw_int = piece.split("=", 1)
        try:
            values[key] = int(raw_int)
        except ValueError:
            return None
    head_len = values.get("h")
    audio_steps = values.get("a")
    if head_len is None or audio_steps is None:
        return None
    if head_len < 0 or audio_steps < 0:
        return None
    return head_len, audio_steps


def _vllm_live_prompt_is_token_ids_from_cache_salt(
    cache_salt: str | None,
    *,
    prompt_len: int,
) -> list[bool] | None:
    metadata = _decode_vllm_live_prompt_mask_metadata(cache_salt)
    if metadata is None:
        return None
    head_len, audio_steps = metadata
    if head_len + audio_steps > prompt_len:
        return None
    tail_len = prompt_len - head_len - audio_steps
    return ([True] * head_len) + ([False] * audio_steps) + ([True] * tail_len)


def _vllm_live_prompt_is_token_ids_from_repeated_placeholder(
    prompt_token_ids,
    *,
    prompt_len: int,
    compact_mask_len: int,
) -> list[bool] | None:
    if prompt_token_ids is None or len(prompt_token_ids) != prompt_len:
        return None
    missing_len = prompt_len - int(compact_mask_len)
    if missing_len <= 0:
        return None
    best_start = None
    best_len = 0
    run_start = 0
    previous_token = object()
    for index, token_id in enumerate(list(prompt_token_ids) + [object()]):
        if index == 0:
            previous_token = token_id
            continue
        if token_id != previous_token:
            run_len = index - run_start
            if run_len > best_len:
                best_start = run_start
                best_len = run_len
            run_start = index
            previous_token = token_id
    if best_start is None or best_len < missing_len:
        return None
    if best_len != missing_len:
        return None
    return (
        ([True] * best_start)
        + ([False] * missing_len)
        + ([True] * (prompt_len - best_start - missing_len))
    )


def _vllm_live_prompt_is_token_ids_from_prompt_embeds(prompt_embeds) -> list[bool] | None:
    if prompt_embeds is None or not hasattr(prompt_embeds, "shape"):
        return None
    if len(prompt_embeds.shape) != 2:
        return None
    try:
        if hasattr(prompt_embeds, "detach"):
            row_energy = prompt_embeds.detach().abs().sum(dim=-1).cpu().tolist()
        else:
            row_energy = np.abs(np.asarray(prompt_embeds)).sum(axis=-1).tolist()
    except Exception:
        return None
    return [float(value) == 0.0 for value in row_energy]


def _vllm_live_full_prompt_is_token_ids(
    prompt_is_token_ids,
    *,
    cache_salt: str | None,
    prompt_len: int,
    prompt_token_ids=None,
    prompt_embeds=None,
) -> list[bool] | None:
    if prompt_is_token_ids is not None and len(prompt_is_token_ids) == prompt_len:
        return list(prompt_is_token_ids)
    restored_mask = _vllm_live_prompt_is_token_ids_from_cache_salt(
        cache_salt,
        prompt_len=prompt_len,
    )
    if restored_mask is not None:
        return restored_mask
    if prompt_is_token_ids is not None:
        restored_mask = _vllm_live_prompt_is_token_ids_from_repeated_placeholder(
            prompt_token_ids,
            prompt_len=prompt_len,
            compact_mask_len=len(prompt_is_token_ids),
        )
        if restored_mask is not None:
            return restored_mask
    restored_mask = _vllm_live_prompt_is_token_ids_from_prompt_embeds(prompt_embeds)
    if restored_mask is not None and len(restored_mask) == prompt_len:
        return restored_mask
    return None


def _vllm_live_prompt_input(
    prompt_input: dict,
    *,
    prefix_keep_len: int,
    prompt_head_len: int | None = None,
    audio_steps: int | None = None,
) -> dict:
    """Copy a prompt input and tag it for request-local vLLM append."""
    tagged = dict(prompt_input)
    tagged["cache_salt"] = _encode_vllm_live_cache_salt(
        tagged.get("cache_salt"),
        prefix_keep_len=prefix_keep_len,
        prompt_head_len=prompt_head_len,
        audio_steps=audio_steps,
    )
    return tagged


def _compact_vllm_live_prompt_embeds(prompt_embeds, cache_salt: str | None):
    if os.environ.get("WLK_VLLM_LIVE_COMPACT_PROMPT_EMBEDS", "0") != "1":
        return prompt_embeds
    _clean_salt, prefix_keep_len = _decode_vllm_live_cache_salt(cache_salt)
    if prefix_keep_len is None or int(prefix_keep_len) <= 0:
        return prompt_embeds
    if prompt_embeds is None or not hasattr(prompt_embeds, "shape"):
        return prompt_embeds
    try:
        if len(prompt_embeds.shape) != 2:
            return prompt_embeds
        full_len = int(prompt_embeds.shape[-2])
        hidden_size = int(prompt_embeds.shape[-1])
        lookback = max(
            0,
            int(os.environ.get("WLK_VLLM_LIVE_COMPACT_LOOKBACK_TOKENS", "64") or 0),
        )
        offset = max(0, min(max(0, int(prefix_keep_len)) - lookback, full_len))
        if offset <= 0:
            return prompt_embeds
        suffix = prompt_embeds[offset:]
        if hasattr(suffix, "contiguous"):
            suffix = suffix.contiguous()
        return _VLLMLiveCompactPromptEmbeds(
            suffix,
            offset=offset,
            full_shape=(full_len, hidden_size),
        )
    except Exception:
        return prompt_embeds


def _apply_vllm_live_request_metadata_from_prompt(request, prompt) -> None:
    """Restore bookkeeping fields that vLLM drops for EmbedsPrompt inputs."""
    if not isinstance(prompt, dict) or "prompt_embeds" not in prompt:
        return
    _mark_vllm_live_prompt_embeds_request(request, prompt.get("cache_salt"))
    prompt_token_ids = prompt.get("prompt_token_ids")
    if prompt_token_ids is not None and getattr(request, "prompt_embeds", None) is not None:
        request.prompt_token_ids = list(prompt_token_ids)
    prompt_is_token_ids = prompt.get("prompt_is_token_ids")
    if prompt_is_token_ids is None:
        prompt_is_token_ids = prompt.get("is_token_ids")
    if hasattr(request, "prompt_is_token_ids"):
        prompt_len = None
        prompt_embeds = getattr(request, "prompt_embeds", None)
        if prompt_embeds is not None and hasattr(prompt_embeds, "shape"):
            prompt_len = int(prompt_embeds.shape[-2])
        elif prompt_token_ids is not None:
            prompt_len = len(prompt_token_ids)
        if prompt_len is None:
            restored_mask = (
                list(prompt_is_token_ids) if prompt_is_token_ids is not None else None
            )
        else:
            restored_mask = _vllm_live_full_prompt_is_token_ids(
                prompt_is_token_ids,
                cache_salt=prompt.get("cache_salt"),
                prompt_len=prompt_len,
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
            )
        if restored_mask is not None:
            request.prompt_is_token_ids = restored_mask
    compact_embeds = _compact_vllm_live_prompt_embeds(
        getattr(request, "prompt_embeds", None),
        prompt.get("cache_salt"),
    )
    if compact_embeds is not getattr(request, "prompt_embeds", None):
        request.prompt_embeds = compact_embeds


def _restore_vllm_live_request_prompt_mask(request) -> None:
    if not hasattr(request, "prompt_is_token_ids"):
        return
    prompt_embeds = getattr(request, "prompt_embeds", None)
    if prompt_embeds is None:
        return
    prompt_len = int(prompt_embeds.shape[-2])
    prompt_is_token_ids = getattr(request, "prompt_is_token_ids", None)
    restored_mask = _vllm_live_full_prompt_is_token_ids(
        prompt_is_token_ids,
        cache_salt=getattr(request, "cache_salt", None),
        prompt_len=prompt_len,
        prompt_token_ids=getattr(request, "prompt_token_ids", None),
        prompt_embeds=prompt_embeds,
    )
    if os.environ.get("WLK_DEBUG_VLLM_LIVE_MASK"):
        logger.warning(
            "[qwen3-vllm] live mask restore prompt_len=%s token_len=%s "
            "mask_len=%s restored_len=%s has_live_salt=%s",
            prompt_len,
            len(getattr(request, "prompt_token_ids", None) or []),
            len(prompt_is_token_ids or []),
            len(restored_mask or []),
            bool(
                getattr(request, "cache_salt", None)
                and _VLLM_LIVE_PREFIX_MARKER in getattr(request, "cache_salt", "")
            ),
        )
    if restored_mask is not None:
        request.prompt_is_token_ids = restored_mask


def _patch_vllm_live_prompt_embeds_streaming() -> bool:
    """Patch vLLM V1 streaming so prompt_embeds can be updated append-style.

    This does not expose a public backend by itself. It installs the minimal
    hooks needed for a future live vLLM decoder session:

    - allow prompt_embeds in AsyncLLM streaming inputs;
    - carry prompt_embeds/cache_salt through StreamingUpdate;
    - when cache_salt carries ``wlk_live_prefix=N``, replace the prompt but keep
      only the first N already-computed KV positions.
    """
    try:
        import inspect
        import textwrap

        from vllm.v1.core.sched import output as sched_output_module
        from vllm.v1.core.sched import async_scheduler as async_sched_module
        from vllm.v1.core.sched import scheduler as sched_module
        from vllm.v1.engine import async_llm as async_llm_module
        from vllm.v1.engine import input_processor as input_processor_module
        from vllm.v1.engine import output_processor as output_processor_module
        from vllm.v1.worker import gpu_input_batch as gpu_input_batch_module
        from vllm.v1 import request as request_module
    except Exception as exc:  # pragma: no cover - version-dependent fallback
        logger.debug("[qwen3-vllm] could not import vLLM live streaming hooks: %s", exc)
        return False

    patched = True

    def reset_output_state_from_request(req_state, request, tokenizer) -> None:
        sampling_params = getattr(request, "sampling_params", None)
        if sampling_params is not None:
            if hasattr(req_state, "output_kind"):
                req_state.output_kind = sampling_params.output_kind
            if hasattr(req_state, "max_tokens_param"):
                req_state.max_tokens_param = sampling_params.max_tokens
            if hasattr(req_state, "top_p"):
                req_state.top_p = sampling_params.top_p
            if hasattr(req_state, "n"):
                req_state.n = sampling_params.n
            if hasattr(req_state, "temperature"):
                req_state.temperature = sampling_params.temperature
            tokenizer_for_request = (
                None
                if not getattr(sampling_params, "detokenize", True)
                else tokenizer
            )
            logprobs_cls = getattr(output_processor_module, "LogprobsProcessor", None)
            detokenizer_cls = getattr(
                output_processor_module,
                "IncrementalDetokenizer",
                None,
            )
            if logprobs_cls is not None and hasattr(req_state, "logprobs_processor"):
                req_state.logprobs_processor = logprobs_cls.from_new_request(
                    tokenizer=tokenizer_for_request,
                    request=request,
                )
            if detokenizer_cls is not None and hasattr(req_state, "detokenizer"):
                req_state.detokenizer = detokenizer_cls.from_new_request(
                    tokenizer=tokenizer_for_request,
                    request=request,
                )
        if hasattr(req_state, "sent_tokens_offset"):
            req_state.sent_tokens_offset = 0
        routed_experts = getattr(req_state, "routed_experts_chunks", None)
        if routed_experts is not None:
            routed_experts.clear()
        if hasattr(req_state, "num_cached_tokens"):
            req_state.num_cached_tokens = 0

    def attach_output_state_reset(update, request, tokenizer) -> None:
        sampling_params = getattr(request, "sampling_params", None)
        if sampling_params is None:
            return
        tokenizer_for_request = (
            None
            if not getattr(sampling_params, "detokenize", True)
            else tokenizer
        )
        logprobs_cls = getattr(output_processor_module, "LogprobsProcessor", None)
        detokenizer_cls = getattr(
            output_processor_module,
            "IncrementalDetokenizer",
            None,
        )
        if logprobs_cls is not None:
            update._wlk_logprobs_processor = logprobs_cls.from_new_request(
                tokenizer=tokenizer_for_request,
                request=request,
            )
        if detokenizer_cls is not None:
            update._wlk_detokenizer = detokenizer_cls.from_new_request(
                tokenizer=tokenizer_for_request,
                request=request,
            )
        update._wlk_sampling_params = sampling_params

    def apply_attached_output_state_reset(req_state, update) -> None:
        sampling_params = getattr(update, "_wlk_sampling_params", None)
        if sampling_params is not None:
            if hasattr(req_state, "output_kind"):
                req_state.output_kind = sampling_params.output_kind
            if hasattr(req_state, "max_tokens_param"):
                req_state.max_tokens_param = sampling_params.max_tokens
            if hasattr(req_state, "top_p"):
                req_state.top_p = sampling_params.top_p
            if hasattr(req_state, "n"):
                req_state.n = sampling_params.n
            if hasattr(req_state, "temperature"):
                req_state.temperature = sampling_params.temperature
        detokenizer = getattr(update, "_wlk_detokenizer", None)
        if detokenizer is not None and hasattr(req_state, "detokenizer"):
            req_state.detokenizer = detokenizer
        logprobs_processor = getattr(update, "_wlk_logprobs_processor", None)
        if (
            logprobs_processor is not None
            and hasattr(req_state, "logprobs_processor")
        ):
            req_state.logprobs_processor = logprobs_processor
        if hasattr(req_state, "sent_tokens_offset"):
            req_state.sent_tokens_offset = 0
        routed_experts = getattr(req_state, "routed_experts_chunks", None)
        if routed_experts is not None:
            routed_experts.clear()
        if hasattr(req_state, "num_cached_tokens"):
            req_state.num_cached_tokens = 0

    processor_cls = getattr(input_processor_module, "InputProcessor", None)
    if processor_cls is not None and not getattr(
        processor_cls,
        "_wlk_live_prompt_embeds_patch",
        False,
    ):
        original_process_inputs = processor_cls.process_inputs

        def patched_process_inputs(self, request_id, prompt, params, *args, **kwargs):
            request = original_process_inputs(
                self,
                request_id,
                prompt,
                params,
                *args,
                **kwargs,
            )
            _apply_vllm_live_request_metadata_from_prompt(request, prompt)
            return request

        processor_cls.process_inputs = patched_process_inputs
        processor_cls._wlk_live_prompt_embeds_patch = True

    request_cls = getattr(request_module, "Request", None)
    if request_cls is not None and not getattr(
        request_cls,
        "_wlk_live_prompt_mask_patch",
        False,
    ):
        original_from_engine_core_request = request_cls.from_engine_core_request

        def patched_from_engine_core_request(cls, request, block_hasher):
            parsed_request = original_from_engine_core_request(request, block_hasher)
            _restore_vllm_live_request_prompt_mask(parsed_request)
            _mark_vllm_live_prompt_embeds_request(
                parsed_request,
                getattr(parsed_request, "cache_salt", None),
            )
            return parsed_request

        request_cls.from_engine_core_request = classmethod(
            patched_from_engine_core_request
        )
        request_cls._wlk_live_prompt_mask_patch = True

    new_request_data_cls = getattr(sched_output_module, "NewRequestData", None)
    if new_request_data_cls is not None and not getattr(
        new_request_data_cls,
        "_wlk_live_prompt_mask_patch",
        False,
    ):
        original_new_request_data_from_request = new_request_data_cls.from_request

        def patched_new_request_data_from_request(
            cls,
            request,
            block_ids,
            prefill_token_ids=None,
        ):
            data = original_new_request_data_from_request(
                request,
                block_ids,
                prefill_token_ids,
            )
            prompt_embeds = getattr(data, "prompt_embeds", None)
            if prompt_embeds is not None:
                restored_mask = _vllm_live_full_prompt_is_token_ids(
                    getattr(data, "prompt_is_token_ids", None),
                    cache_salt=getattr(request, "cache_salt", None),
                    prompt_len=int(prompt_embeds.shape[-2]),
                    prompt_token_ids=getattr(data, "prompt_token_ids", None),
                    prompt_embeds=prompt_embeds,
                )
                if restored_mask is not None:
                    data.prompt_is_token_ids = restored_mask
            return data

        new_request_data_cls.from_request = classmethod(
            patched_new_request_data_from_request
        )
        new_request_data_cls._wlk_live_prompt_mask_patch = True

    streaming_update_cls = getattr(request_module, "StreamingUpdate", None)
    if streaming_update_cls is not None and not getattr(
        streaming_update_cls,
        "_wlk_live_prompt_embeds_patch",
        False,
    ):

        def patched_from_request(cls, request):
            if not request.resumable:
                return None
            update = cls(
                mm_features=request.mm_features,
                prompt_token_ids=request.prompt_token_ids,
                max_tokens=request.max_tokens,
                arrival_time=request.arrival_time,
                sampling_params=request.sampling_params,
            )
            update.prompt_embeds = getattr(request, "prompt_embeds", None)
            update.prompt_is_token_ids = getattr(request, "prompt_is_token_ids", None)
            update.cache_salt = getattr(request, "cache_salt", None)
            return update

        streaming_update_cls.from_request = classmethod(patched_from_request)
        streaming_update_cls._wlk_live_prompt_embeds_patch = True

    output_update_cls = getattr(output_processor_module, "StreamingUpdate", None)
    output_state_cls = getattr(output_processor_module, "RequestState", None)
    output_processor_cls = getattr(output_processor_module, "OutputProcessor", None)
    if output_processor_cls is not None and not getattr(
        output_processor_cls,
        "_wlk_live_prompt_embeds_patch",
        False,
    ):
        original_update_state = output_processor_cls._update_streaming_request_state

        def patched_update_streaming_request_state(self, req_state, request, prompt):
            _restore_vllm_live_request_prompt_mask(request)
            previous_queue = getattr(req_state, "input_chunk_queue", None)
            result = original_update_state(self, req_state, request, prompt)
            prompt_embeds = getattr(request, "prompt_embeds", None)
            if prompt_embeds is None:
                return result
            if previous_queue is None:
                req_state.prompt_token_ids = list(
                    getattr(request, "prompt_token_ids", None) or []
                )
                req_state.prompt_embeds = prompt_embeds
                req_state.prompt_len = int(prompt_embeds.shape[-2])
                reset_output_state_from_request(
                    req_state,
                    request,
                    getattr(self, "tokenizer", None),
                )
                if hasattr(req_state, "prompt_is_token_ids"):
                    req_state.prompt_is_token_ids = getattr(
                        request,
                        "prompt_is_token_ids",
                        None,
                    )
                req_state.is_prefilling = True
                if getattr(req_state, "stats", None) is not None:
                    req_state.stats.arrival_time = getattr(
                        request,
                        "arrival_time",
                        req_state.stats.arrival_time,
                    )
            else:
                queue = getattr(req_state, "input_chunk_queue", None)
                if queue:
                    update = queue[-1]
                    update.cache_salt = getattr(request, "cache_salt", None)
                    update.prompt_embeds = prompt_embeds
                    update.prompt_is_token_ids = getattr(
                        request,
                        "prompt_is_token_ids",
                        None,
                    )
                    attach_output_state_reset(
                        update,
                        request,
                        getattr(self, "tokenizer", None),
                    )
            return result

        output_processor_cls._update_streaming_request_state = (
            patched_update_streaming_request_state
        )
        output_processor_cls._wlk_live_prompt_embeds_patch = True

    if output_state_cls is not None and output_update_cls is not None and not getattr(
        output_state_cls,
        "_wlk_live_prompt_embeds_patch",
        False,
    ):
        original_apply_streaming_update = output_state_cls.apply_streaming_update

        def patched_apply_streaming_update(self, update):
            _clean_salt, prefix_keep_len = _decode_vllm_live_cache_salt(
                getattr(update, "cache_salt", None)
            )
            if prefix_keep_len is None or getattr(update, "prompt_embeds", None) is None:
                return original_apply_streaming_update(self, update)
            self.streaming_input = not update.final
            self.prompt_token_ids = list(update.prompt_token_ids or [])
            self.prompt_embeds = getattr(update, "prompt_embeds", None)
            self.prompt_len = int(self.prompt_embeds.shape[-2])
            apply_attached_output_state_reset(self, update)
            if self.stats is not None:
                self.stats.arrival_time = update.arrival_time
            self.is_prefilling = True
            return None

        output_state_cls.apply_streaming_update = patched_apply_streaming_update
        output_state_cls._wlk_live_prompt_embeds_patch = True

    scheduler_cls = getattr(sched_module, "Scheduler", None)
    if scheduler_cls is not None and not getattr(
        scheduler_cls,
        "_wlk_live_prompt_embeds_patch",
        False,
    ):
        original_update_session = scheduler_cls._update_request_as_session

        def patched_update_request_as_session(self, session, update):
            clean_salt, prefix_keep_len = _decode_vllm_live_cache_salt(
                getattr(update, "cache_salt", None)
            )
            prompt_embeds = getattr(update, "prompt_embeds", None)
            if prefix_keep_len is None or prompt_embeds is None:
                return original_update_session(self, session, update)

            pending_output_placeholders = max(
                0,
                int(getattr(session, "num_output_placeholders", 0) or 0),
            )
            prompt_len = int(prompt_embeds.shape[-2])
            prefix_keep_len = min(
                int(prefix_keep_len),
                int(getattr(session, "num_computed_tokens", 0)),
                prompt_len,
            )
            prompt_token_ids = list(update.prompt_token_ids or ([0] * prompt_len))
            if len(prompt_token_ids) != prompt_len:
                raise ValueError(
                    "vLLM live prompt_embeds update requires prompt_token_ids "
                    "to have the same length as prompt_embeds"
                )

            session.prompt_embeds = prompt_embeds
            session._wlk_live_prompt_embeds_streaming = True
            session.prompt_token_ids = prompt_token_ids
            if hasattr(session, "prompt_is_token_ids"):
                session.prompt_is_token_ids = _vllm_live_full_prompt_is_token_ids(
                    getattr(update, "prompt_is_token_ids", None),
                    cache_salt=getattr(update, "cache_salt", None),
                    prompt_len=prompt_len,
                    prompt_token_ids=prompt_token_ids,
                    prompt_embeds=prompt_embeds,
                )
            session.cache_salt = clean_salt
            session.mm_features = list(update.mm_features or [])
            session._all_token_ids[:] = prompt_token_ids
            session._output_token_ids.clear()
            session.spec_token_ids.clear()
            if hasattr(session, "num_output_placeholders"):
                session.num_output_placeholders = 0
            if hasattr(session, "discard_latest_async_tokens"):
                session.discard_latest_async_tokens = pending_output_placeholders > 0
            session.num_computed_tokens = prefix_keep_len
            session.num_prompt_tokens = prompt_len

            block_size = max(1, int(getattr(self, "block_size", 1) or 1))
            del session.block_hashes[prefix_keep_len // block_size :]
            if hasattr(session, "_prompt_embeds_per_block_hashes"):
                session._prompt_embeds_per_block_hashes.clear()
            session.update_block_hashes()
            session.arrival_time = update.arrival_time
            session.sampling_params = update.sampling_params
            if session.status == sched_module.RequestStatus.WAITING_FOR_STREAMING_REQ:
                self.num_waiting_for_streaming_input -= 1
            session.status = sched_module.RequestStatus.WAITING
            if self.log_stats:
                session.record_event(sched_module.EngineCoreEventType.QUEUED)
            return None

        scheduler_cls._update_request_as_session = patched_update_request_as_session
        scheduler_cls._wlk_live_prompt_embeds_patch = True

    async_scheduler_cls = getattr(async_sched_module, "AsyncScheduler", None)
    if async_scheduler_cls is not None and not async_scheduler_cls.__dict__.get(
        "_wlk_live_async_prompt_embeds_patch",
        False,
    ):
        original_async_update_request_with_output = (
            async_scheduler_cls._update_request_with_output
        )

        def patched_async_update_request_with_output(
            self,
            request,
            new_token_ids,
        ):
            if not getattr(request, "_wlk_live_prompt_embeds_streaming", False):
                return original_async_update_request_with_output(
                    self,
                    request,
                    new_token_ids,
                )

            if getattr(request, "discard_latest_async_tokens", False):
                request.discard_latest_async_tokens = False
                if hasattr(request, "num_output_placeholders"):
                    request.num_output_placeholders = max(
                        0,
                        int(getattr(request, "num_output_placeholders", 0) or 0)
                        - len(new_token_ids or []),
                    )
                return [], False

            status_before_update = getattr(request, "status", None)
            new_token_ids, stopped = sched_module.Scheduler._update_request_with_output(
                self,
                request,
                new_token_ids,
            )

            placeholders = max(
                0,
                int(getattr(request, "num_output_placeholders", 0) or 0),
            )
            if len(new_token_ids) >= placeholders:
                request.num_output_placeholders = 0
            else:
                request.num_output_placeholders = placeholders - len(new_token_ids)

            if (
                status_before_update == sched_module.RequestStatus.RUNNING
                and hasattr(self, "kv_cache_manager")
            ):
                self.kv_cache_manager.cache_blocks(
                    request,
                    max(
                        0,
                        int(getattr(request, "num_computed_tokens", 0) or 0)
                        - int(getattr(request, "num_output_placeholders", 0) or 0),
                    ),
                )
            return new_token_ids, stopped

        async_scheduler_cls._update_request_with_output = (
            patched_async_update_request_with_output
        )
        async_scheduler_cls._wlk_live_async_prompt_embeds_patch = True

    async_llm_cls = getattr(async_llm_module, "AsyncLLM", None)
    if async_llm_cls is not None and not getattr(
        async_llm_cls,
        "_wlk_live_prompt_embeds_patch",
        False,
    ):
        method = async_llm_cls._add_streaming_input_request
        try:
            source = textwrap.dedent(inspect.getsource(method))
            source_file = inspect.getsourcefile(method) or "<vllm>"
        except Exception as exc:  # pragma: no cover - version-dependent fallback
            logger.debug("[qwen3-vllm] could not inspect vLLM AsyncLLM: %s", exc)
            patched = False
        else:
            rejection = re.compile(
                r"(?P<indent> +)if req\.prompt_embeds is not None:\n"
                r"(?P=indent)    raise ValueError\(\n"
                r"(?P=indent)        \"prompt_embeds not supported for streaming inputs\"\n"
                r"(?P=indent)    \)\n"
            )
            match = rejection.search(source)
            if match is not None:
                indent = match.group("indent")
                async_llm_module.__dict__[
                    "_wlk_restore_vllm_live_request_prompt_mask"
                ] = _restore_vllm_live_request_prompt_mask
                replacement = (
                    f"{indent}if req.prompt_embeds is not None:\n"
                    f"{indent}    _wlk_restore_vllm_live_request_prompt_mask(req)\n"
                )
                namespace = {}
                source = rejection.sub(replacement, source, count=1)
                try:
                    exec(
                        compile(source, source_file, "exec"),
                        async_llm_module.__dict__,
                        namespace,
                    )
                except Exception as exc:  # pragma: no cover - version-dependent fallback
                    logger.debug("[qwen3-vllm] could not patch vLLM AsyncLLM: %s", exc)
                    patched = False
                else:
                    async_llm_cls._add_streaming_input_request = namespace[
                        "_add_streaming_input_request"
                    ]
                    async_llm_cls._wlk_live_prompt_embeds_patch = True
            else:
                patched = False

    if async_llm_cls is not None and not getattr(
        async_llm_cls,
        "_wlk_live_add_request_prompt_mask_patch",
        False,
    ):
        original_add_request = async_llm_cls._add_request

        async def patched_add_request(self, request, prompt, parent_req, index, queue):
            _restore_vllm_live_request_prompt_mask(request)
            return await original_add_request(
                self,
                request,
                prompt,
                parent_req,
                index,
                queue,
            )

        async_llm_cls._add_request = patched_add_request
        async_llm_cls._wlk_live_add_request_prompt_mask_patch = True

    input_batch_cls = getattr(gpu_input_batch_module, "InputBatch", None)
    if input_batch_cls is not None and not getattr(
        input_batch_cls,
        "_wlk_live_prompt_mask_patch",
        False,
    ):
        original_input_batch_add_request = input_batch_cls.add_request

        def patched_input_batch_add_request(self, request):
            prompt_embeds = getattr(request, "prompt_embeds", None)
            prompt_is_token_ids = getattr(request, "prompt_is_token_ids", None)
            if prompt_embeds is not None and prompt_is_token_ids is not None:
                prompt_len = int(prompt_embeds.shape[-2])
                if len(prompt_is_token_ids) != prompt_len:
                    restored_mask = _vllm_live_full_prompt_is_token_ids(
                        prompt_is_token_ids,
                        cache_salt=None,
                        prompt_len=prompt_len,
                        prompt_token_ids=getattr(request, "prompt_token_ids", None),
                        prompt_embeds=prompt_embeds,
                    )
                    if restored_mask is not None:
                        request.prompt_is_token_ids = restored_mask
            return original_input_batch_add_request(self, request)

        input_batch_cls.add_request = patched_input_batch_add_request
        input_batch_cls._wlk_live_prompt_mask_patch = True

    return patched


class _VLLMLiveTextDecoderSession:
    """One request-local vLLM streaming decoder session."""

    def __init__(self, decoder: "_VLLMLiveTextDecoder"):
        import queue
        import uuid as uuid_module

        self._decoder = decoder
        self._request_id = f"wlk-qwen3-vllm-live-{uuid_module.uuid4().hex}"
        self._result_queue: queue.Queue = queue.Queue()
        self._async_input_queue = None
        self._consume_task = None
        self._started = False
        self._closed = False
        self._last_returned_on_idle = False
        self._last_decode_wall_sec = 0.0
        self._last_decode_outputs = 0
        self._last_decode_time_to_first_output_sec = 0.0
        self._last_decode_idle_tail_sec = 0.0
        self._last_delta_output = False
        self._last_delta_token_count = 0

    def _ensure_started(self, sampling_params) -> None:
        if self._started:
            return
        self._decoder._start_session(self, sampling_params)
        self._started = True

    @staticmethod
    def _is_chunk_final(output) -> bool:
        if getattr(output, "finished", False):
            return True
        outputs = getattr(output, "outputs", None) or []
        if not outputs:
            return False
        return getattr(outputs[0], "finish_reason", None) is not None

    def decode(
        self,
        prompt_input: dict,
        sampling_params,
        *,
        timeout: float = 120.0,
        idle_timeout: float = 0.25,
    ):
        import queue
        import time

        if self._closed:
            raise RuntimeError("vLLM live decoder session is closed")
        self._ensure_started(sampling_params)
        self._last_returned_on_idle = False
        self._last_delta_output = _sampling_params_output_kind_name(
            sampling_params
        ) == "DELTA"
        self._last_delta_token_count = 0
        delta_token_ids: list[int] = []
        delta_text_parts: list[str] = []
        prompt_logprobs = None
        decode_started_at = time.perf_counter()
        first_output_at = None
        last_output_at_perf = None
        output_count = 0

        def finish(item):
            finished_at = time.perf_counter()
            self._last_delta_token_count = len(delta_token_ids)
            self._last_decode_wall_sec = finished_at - decode_started_at
            self._last_decode_outputs = output_count
            self._last_decode_time_to_first_output_sec = (
                first_output_at - decode_started_at
                if first_output_at is not None
                else 0.0
            )
            self._last_decode_idle_tail_sec = (
                finished_at - last_output_at_perf
                if last_output_at_perf is not None
                else 0.0
            )
            if self._last_delta_output and item is not None:
                return _clone_vllm_output_with_completion_token_ids(
                    item,
                    delta_token_ids,
                    text="".join(delta_text_parts),
                    prompt_logprobs=prompt_logprobs,
                )
            return item

        self._decoder._put_session_input(self, prompt_input, sampling_params)
        deadline = time.monotonic() + float(timeout)
        last_output = None
        last_output_at = None
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("timed out waiting for vLLM live decoder output")
            wait_timeout = min(0.25, max(0.0, deadline - now))
            if last_output is not None and last_output_at is not None:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, float(idle_timeout) - (now - last_output_at)),
                )
            try:
                item = self._result_queue.get(timeout=wait_timeout)
            except queue.Empty:
                if (
                    last_output is not None
                    and last_output_at is not None
                    and time.monotonic() - last_output_at >= float(idle_timeout)
                ):
                    self._last_returned_on_idle = True
                    return finish(last_output)
                continue
            if isinstance(item, BaseException):
                raise item
            if item is None:
                if last_output is not None:
                    return finish(last_output)
                raise RuntimeError("vLLM live decoder session ended before output")
            if getattr(item, "outputs", None):
                output_count += 1
                if self._last_delta_output:
                    completion = item.outputs[0]
                    token_ids = getattr(completion, "token_ids", None) or []
                    delta_token_ids.extend(int(token_id) for token_id in token_ids)
                    text = getattr(completion, "text", "") or ""
                    if text:
                        delta_text_parts.append(str(text))
                if prompt_logprobs is None:
                    prompt_logprobs = getattr(item, "prompt_logprobs", None)
                now_perf = time.perf_counter()
                if first_output_at is None:
                    first_output_at = now_perf
                last_output = item
                last_output_at = time.monotonic()
                last_output_at_perf = now_perf
            if self._is_chunk_final(item):
                return finish(item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._decoder._close_session(self)


class _VLLMLiveTextDecoder:
    """Shared AsyncLLM engine with one streaming request per audio stream."""

    def __init__(
        self,
        model_path: str,
        *,
        engine_kwargs: dict,
        startup_timeout: float = 180.0,
    ):
        import asyncio
        import threading

        self.model_path = model_path
        self.engine_kwargs = dict(engine_kwargs)
        self._startup_timeout = float(startup_timeout)
        self._loop = None
        self._engine = None
        self._StreamingInput = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="qwen3-vllm-live-decoder",
            daemon=True,
        )
        self._asyncio = asyncio
        self._thread.start()
        if not self._ready.wait(self._startup_timeout):
            raise RuntimeError("timed out starting vLLM live decoder")
        if self._startup_error is not None:
            raise RuntimeError("failed to start vLLM live decoder") from self._startup_error

    def _thread_main(self) -> None:
        asyncio = self._asyncio
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_engine())
        except BaseException as exc:  # pragma: no cover - startup is environment-specific
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        loop.run_forever()

    async def _start_engine(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.protocol import StreamingInput
        from vllm.v1.engine.async_llm import AsyncLLM

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        if os.environ.get("WLK_VLLM_LIVE_COMPACT_PROMPT_EMBEDS", "0") == "1":
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            runner="generate",
            enable_prompt_embeds=True,
            enable_prefix_caching=True,
            **self.engine_kwargs,
        )
        self._engine = AsyncLLM.from_engine_args(engine_args)
        self._StreamingInput = StreamingInput

    def new_session(self) -> _VLLMLiveTextDecoderSession:
        return _VLLMLiveTextDecoderSession(self)

    def _start_session(
        self,
        session: _VLLMLiveTextDecoderSession,
        sampling_params,
    ) -> None:
        if self._loop is None:
            raise RuntimeError("vLLM live decoder loop is not running")
        future = self._asyncio.run_coroutine_threadsafe(
            self._start_session_async(session, sampling_params),
            self._loop,
        )
        future.result(timeout=self._startup_timeout)

    async def _start_session_async(self, session, sampling_params) -> None:
        assert self._engine is not None
        input_queue = self._asyncio.Queue()
        session._async_input_queue = input_queue

        async def input_stream():
            while True:
                item = await input_queue.get()
                if item is None:
                    break
                prompt_input, params = item
                yield self._StreamingInput(prompt=prompt_input, sampling_params=params)

        async def consume_outputs():
            try:
                async for output in self._engine.generate(
                    input_stream(),
                    sampling_params,
                    session._request_id,
                ):
                    session._result_queue.put(output)
            except BaseException as exc:
                session._result_queue.put(exc)
            finally:
                session._result_queue.put(None)

        session._consume_task = self._asyncio.create_task(consume_outputs())

    def _put_session_input(self, session, prompt_input: dict, sampling_params) -> None:
        if self._loop is None or session._async_input_queue is None:
            raise RuntimeError("vLLM live decoder session is not started")
        future = self._asyncio.run_coroutine_threadsafe(
            session._async_input_queue.put((prompt_input, sampling_params)),
            self._loop,
        )
        future.result(timeout=self._startup_timeout)

    def _close_session(self, session) -> None:
        if self._loop is None or session._async_input_queue is None:
            return
        future = self._asyncio.run_coroutine_threadsafe(
            self._close_session_async(session),
            self._loop,
        )
        future.result(timeout=self._startup_timeout)

    async def _close_session_async(self, session) -> None:
        task = getattr(session, "_consume_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except self._asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.debug("[qwen3-vllm] vLLM live session cancel failed: %s", exc)
        queue = getattr(session, "_async_input_queue", None)
        if queue is not None:
            try:
                queue.put_nowait(None)
            except Exception:
                pass


def _resolve_model_path(kwargs: dict) -> str:
    model_path = kwargs.get("vllm_model") or kwargs.get("model_dir") or kwargs.get("model_path")
    if model_path:
        return model_path

    model_size = (kwargs.get("model_size") or "").strip()
    if not model_size:
        return DEFAULT_QWEN3_VLLM_MODEL
    lowered = model_size.lower()
    if "/" in model_size or model_size.startswith((".", "/")):
        return model_size
    return QWEN3_VLLM_MODEL_MAPPING.get(lowered, model_size)


def _resolve_audio_backend(kwargs: dict) -> str:
    backend = str(kwargs.get("qwen3_vllm_audio_backend", "standard") or "standard")
    if backend not in {"standard", "causal"}:
        raise ValueError(
            "qwen3_vllm_audio_backend must be 'standard' or 'causal', "
            f"got {backend!r}"
        )
    return backend


def _resolve_causal_decoder_backend(kwargs: dict, audio_backend: str) -> str:
    backend = str(
        kwargs.get("qwen3_vllm_causal_decoder_backend", "vllm-text") or "vllm-text"
    )
    if backend == "auto":
        backend = "vllm-text"
    if backend not in {"append-kv", "rolling", "vllm", "vllm-live", "vllm-text"}:
        raise ValueError(
            "qwen3_vllm_causal_decoder_backend must be 'append-kv', "
            "'rolling', 'vllm', 'vllm-live', or 'vllm-text', "
            f"got {backend!r}"
        )
    if audio_backend != "causal":
        return "vllm"
    return backend


def _safe_model_cache_name(model_id: str) -> str:
    import hashlib

    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip(".-")
    return f"{slug[:80]}--{digest}"


def _ensure_qwen3_asr_text_decoder_model(
    model_id: str,
    *,
    output_dir: str | None = None,
) -> str:
    """Export Qwen3-ASR's text decoder as a plain Qwen3ForCausalLM repo.

    vLLM's Qwen3-ASR runner is multimodal and currently forces the slow
    prompt-embeds path for causal audio embeddings. The text decoder itself is
    just Qwen3, so this helper materializes ``thinker.model`` + ``lm_head`` as
    a local text-only model that vLLM can run with its normal CUDA graph path.
    """
    if output_dir:
        out = Path(output_dir).expanduser()
    else:
        out = (
            Path.home()
            / ".cache"
            / "qwen3-asr-causal"
            / "qwen3_asr_text_decoder"
            / _safe_model_cache_name(model_id)
        )
    marker = out / "wlk_text_decoder_export.json"
    config_path = out / "config.json"
    if marker.exists() and config_path.exists():
        return str(out)

    import json

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from .model import _register_qwen3_asr_transformers

    logger.info("[qwen3-vllm] exporting Qwen3-ASR text decoder from '%s' ...", model_id)
    _register_qwen3_asr_transformers()
    hf_config = AutoConfig.from_pretrained(model_id)
    text_config = hf_config.thinker_config.text_config
    cfg_dict = text_config.to_dict()
    for key in ("architectures", "auto_map", "model_type", "rope_scaling"):
        cfg_dict.pop(key, None)
    qwen3_config = Qwen3Config(**cfg_dict)
    text_lm = Qwen3ForCausalLM(qwen3_config)

    qwen_model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    thinker = qwen_model.thinker
    state = {f"model.{key}": value for key, value in thinker.model.state_dict().items()}
    state.update(
        {f"lm_head.{key}": value for key, value in thinker.lm_head.state_dict().items()}
    )
    missing, unexpected = text_lm.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "could not export Qwen3-ASR text decoder cleanly: "
            f"missing={missing[:8]!r} unexpected={unexpected[:8]!r}"
        )

    out.mkdir(parents=True, exist_ok=True)
    text_lm.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)

    # Qwen3-ASR text config carries M-RoPE metadata. The standalone text model
    # must use plain linear positions, otherwise vLLM tries to initialize
    # multimodal M-RoPE for a Qwen3ForCausalLM and aborts.
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        saved_config.pop("rope_scaling", None)
        config_path.write_text(json.dumps(saved_config, indent=2) + "\n", encoding="utf-8")
    marker.write_text(
        json.dumps({"source_model": model_id, "format": "qwen3-asr-text-decoder"}, indent=2),
        encoding="utf-8",
    )
    logger.info("[qwen3-vllm] exported text decoder to '%s'", out)
    return str(out)


def _streaming_dtype_from_vllm_dtype(dtype: str) -> str:
    lowered = str(dtype or "auto").lower()
    if lowered in {"auto", "bfloat16", "bf16"}:
        return "bfloat16"
    if lowered in {"float16", "fp16", "half"}:
        return "float16"
    if lowered in {"float32", "fp32"}:
        return "float32"
    return lowered


def _qwen3_language(language: Optional[str]) -> Optional[str]:
    if not language or language == "auto":
        return None
    return WHISPER_TO_QWEN3_LANGUAGE.get(language, language)


def _clean_asr_text(text: str) -> str:
    text = _EOT_RE.sub("", text or "").strip()
    if _ASR_TEXT_TAG in text:
        _, text = text.rsplit(_ASR_TEXT_TAG, 1)
    return _EOT_RE.sub("", text).strip()


def _join_asr_text_prefix(prefix: str, suffix: str) -> str:
    prefix = (prefix or "").strip()
    suffix = (suffix or "").strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if prefix.endswith((" ", "\n", "\t")) or suffix.startswith((" ", "\n", "\t")):
        return f"{prefix}{suffix}".strip()
    return f"{prefix} {suffix}".strip()


def _detect_qwen3_language(text: str) -> Optional[str]:
    match = _LANG_RE.search(text or "")
    if not match:
        return None
    language = match.group(1).strip()
    if _ASR_TEXT_TAG in language:
        language = language.split(_ASR_TEXT_TAG, 1)[0].strip()
    return language or None


def _token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer could not encode required token {token!r}")
    return int(token_ids[0])


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _clean_align_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _normalize_align_word(token: str) -> str:
    return _clean_align_token(token).casefold()


def _split_align_words(text: str) -> list[str]:
    words: list[str] = []
    for segment in text.split():
        cleaned = _clean_align_token(segment)
        if not cleaned:
            continue
        buf: list[str] = []
        for ch in cleaned:
            if _is_cjk_char(ch):
                if buf:
                    words.append("".join(buf))
                    buf = []
                words.append(ch)
            else:
                buf.append(ch)
        if buf:
            words.append("".join(buf))
    return words


def _fix_timestamps(values) -> list[float]:
    data = [float(v) for v in values]
    n = len(data)
    if n <= 1:
        return data

    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_idx = dp.index(max(dp))
    lis_indices = []
    while max_idx != -1:
        lis_indices.append(max_idx)
        max_idx = parent[max_idx]
    lis_indices.reverse()

    normal = [False] * n
    for idx in lis_indices:
        normal[idx] = True

    result = data.copy()
    i = 0
    while i < n:
        if normal[i]:
            i += 1
            continue

        j = i
        while j < n and not normal[j]:
            j += 1

        count = j - i
        left = next((result[k] for k in range(i - 1, -1, -1) if normal[k]), None)
        right = next((result[k] for k in range(j, n) if normal[k]), None)

        if count <= 2:
            for k in range(i, j):
                if left is None:
                    result[k] = right if right is not None else result[k]
                elif right is None:
                    result[k] = left
                else:
                    result[k] = left if (k - (i - 1)) <= (j - k) else right
        elif left is not None and right is not None:
            step = (right - left) / (count + 1)
            for k in range(i, j):
                result[k] = left + step * (k - i + 1)
        elif left is not None:
            for k in range(i, j):
                result[k] = left
        elif right is not None:
            for k in range(i, j):
                result[k] = right
        i = j

    return result


def _normalize_timestamp_segment_time(value) -> float:
    segment_time = float(value)
    if segment_time > 1.0:
        segment_time /= 1000.0
    return segment_time


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "cpu"):
        return data.cpu().numpy()
    return np.asarray(data)


def _prompt_input_token_count(prompt_input) -> int:
    if not prompt_input:
        return 0
    prompt_embeds = prompt_input.get("prompt_embeds")
    if prompt_embeds is not None:
        return int(prompt_embeds.shape[-2])
    prompt_token_ids = prompt_input.get("prompt_token_ids") or []
    return len(prompt_token_ids)


def _request_output_num_cached_tokens(output) -> Optional[int]:
    if output is None:
        return None
    value = getattr(output, "num_cached_tokens", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vllm_request_output_kind(name: str):
    try:
        from vllm.sampling_params import RequestOutputKind
    except Exception:
        return None
    return getattr(RequestOutputKind, str(name).upper(), None)


def _sampling_params_output_kind_name(sampling_params) -> str:
    value = getattr(sampling_params, "output_kind", None)
    if value is None:
        kwargs = getattr(sampling_params, "kwargs", None)
        if isinstance(kwargs, dict):
            value = kwargs.get("output_kind")
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).upper()
    return str(value or "").upper()


def _clone_vllm_output_with_completion_token_ids(
    output,
    token_ids: Sequence[int],
    *,
    text: str = "",
    prompt_logprobs=None,
):
    if output is None or not getattr(output, "outputs", None):
        return output
    try:
        cloned_output = copy(output)
        cloned_completion = copy(output.outputs[0])
        cloned_completion.token_ids = list(token_ids)
        cloned_completion.text = text
        cloned_output.outputs = [cloned_completion] + list(output.outputs[1:])
        if prompt_logprobs is not None:
            cloned_output.prompt_logprobs = prompt_logprobs
        return cloned_output
    except Exception:
        return output


def _trim_token_ids_at_stop(
    token_ids: Sequence[int],
    stop_token_ids: Sequence[int | None] | None = None,
) -> list[int]:
    stop_ids = {
        int(token_id)
        for token_id in (stop_token_ids or ())
        if token_id is not None
    }
    out: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in stop_ids:
            break
        out.append(token_id)
    return out


def _vllm_prompt_logprob_rank(prompt_logprobs, index: int, token_id: int) -> int | None:
    if prompt_logprobs is None:
        return None
    try:
        position = prompt_logprobs[int(index)]
    except (IndexError, KeyError, TypeError):
        return None
    if not position:
        return None
    try:
        entry = position.get(int(token_id))
    except AttributeError:
        return None
    if entry is None:
        return None
    rank = getattr(entry, "rank", None)
    if rank is None:
        return None
    try:
        return int(rank)
    except (TypeError, ValueError):
        return None


def _accepted_vllm_prompt_draft_tokens(
    prompt_logprobs,
    *,
    draft_start: int,
    draft_token_ids: Sequence[int],
    prompt_len: int | None = None,
) -> int | None:
    """Return the greedy-verified draft prefix length, or None if unverifiable."""
    if prompt_logprobs is None:
        return None
    index_base = int(draft_start)
    try:
        logprob_len = len(prompt_logprobs)
    except TypeError:
        logprob_len = None
    if prompt_len is not None and logprob_len == int(prompt_len) - 1:
        # vLLM V1 stores prompt logprobs shifted: entry i scores prompt token i+1.
        index_base -= 1
    elif logprob_len is not None and logprob_len > 0:
        try:
            first = prompt_logprobs[0]
        except (IndexError, KeyError, TypeError):
            first = None
        if first is not None and int(draft_start) > 0:
            index_base -= 1
    accepted = 0
    for offset, token_id in enumerate(draft_token_ids):
        rank = _vllm_prompt_logprob_rank(
            prompt_logprobs,
            index_base + offset,
            int(token_id),
        )
        if rank is None:
            return None
        if rank != 1:
            break
        accepted += 1
    return accepted


class Qwen3VLLMASR:
    """Model holder for Qwen3-ASR + Qwen3-ForcedAligner through vLLM."""

    sep = ""
    SAMPLING_RATE = 16_000
    backend_choice = "qwen3-vllm"

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.audio_backend = _resolve_audio_backend(kwargs)
        self.causal_decoder_backend = _resolve_causal_decoder_backend(
            kwargs,
            self.audio_backend,
        )
        if self.audio_backend == "causal" and (
            self.causal_decoder_backend not in {"vllm-live", "vllm-text"}
            or (
                self.causal_decoder_backend == "vllm-live"
                and os.environ.get("WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING", "0") != "1"
            )
        ):
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        LLM, SamplingParams, TokensPrompt, AutoConfig = _load_vllm_runtime()

        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if kwargs.get("lan", "auto") == "auto" else kwargs.get("lan")
        if self.audio_backend == "causal" and not self.original_language:
            raise ValueError(
                "qwen3-vllm causal requires an explicit transcription language "
                "(e.g. --language en)."
            )
        self.model_path = _resolve_model_path(kwargs)
        self.aligner_model_path = kwargs.get("vllm_aligner_model") or DEFAULT_QWEN3_VLLM_ALIGNER_MODEL
        self.max_decode_tokens = int(kwargs.get("max_tokens") or 256)
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self.causal_asr = None
        self.asr_llm = None
        self.causal_audio_embeds_mm_enabled = False
        self.causal_attn_implementation = str(
            kwargs.get("qwen3_vllm_causal_attn_implementation", "auto") or "auto"
        )
        self.causal_text_decoder_model = str(
            kwargs.get("qwen3_vllm_text_decoder_model", "") or ""
        )
        self.causal_segment_min_sec = float(
            kwargs.get("qwen3_vllm_segment_min_sec", 0.0) or 0.0
        )
        self.causal_vllm_audio_embed_block_steps = int(
            kwargs.get("qwen3_vllm_audio_embed_block_steps") or 16
        )
        self.causal_vllm_live_idle_timeout_sec = max(
            0.0,
            float(kwargs.get("qwen3_vllm_live_idle_timeout_ms", 50.0) or 0.0)
            / 1000.0,
        )
        self.causal_decode_committed_prefix = (
            os.environ.get("WLK_QWEN3_VLLM_DECODE_COMMITTED_PREFIX", "0") == "1"
        )
        self.causal_decode_prefix_overlap_words = max(
            0,
            int(os.environ.get("WLK_QWEN3_VLLM_DECODE_PREFIX_OVERLAP_WORDS", "3") or 0),
        )
        self.causal_vllm_text_draft_enabled = (
            os.environ.get("WLK_QWEN3_VLLM_TEXT_DRAFT", "0") == "1"
        )
        self.causal_vllm_live_draft_enabled = (
            os.environ.get("WLK_QWEN3_VLLM_LIVE_DRAFT", "0") == "1"
        )

        tensor_parallel_size = int(kwargs.get("vllm_tensor_parallel_size") or 1)
        gpu_memory_utilization = float(kwargs.get("vllm_gpu_memory_utilization") or 0.45)
        dtype = kwargs.get("vllm_dtype") or "auto"

        common_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }
        cache_block_size = int(kwargs.get("qwen3_vllm_cache_block_size") or 0)
        if cache_block_size > 0:
            common_kwargs["block_size"] = cache_block_size
        asr_extra_kwargs = {}
        self.causal_prefix_cache_enabled = False
        use_vllm_asr_decoder = not (
            self.audio_backend == "causal"
            and self.causal_decoder_backend
            in {"append-kv", "rolling", "vllm-live", "vllm-text"}
        )
        if self.audio_backend == "causal" and self.causal_decoder_backend == "vllm":
            self.causal_audio_embeds_mm_enabled = _patch_vllm_qwen3_asr_audio_embeds()
            if self.causal_audio_embeds_mm_enabled:
                self.causal_prefix_cache_enabled = True
                asr_extra_kwargs["enable_prefix_caching"] = True
                asr_extra_kwargs["enable_mm_embeds"] = True
                asr_extra_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}
                logger.info(
                    "[qwen3-vllm] causal CUDA will feed precomputed audio "
                    "embeddings through vLLM multimodal audio_embeds blocks."
                )
            else:
                _patch_vllm_mrope_prompt_embeds()
                asr_extra_kwargs["enforce_eager"] = True
                self.causal_prefix_cache_enabled = _vllm_prompt_embeds_prefix_cache_supported()
                asr_extra_kwargs["enable_prefix_caching"] = self.causal_prefix_cache_enabled
                logger.warning(
                    "[qwen3-vllm] vLLM Qwen3-ASR audio_embeds patch was not "
                    "available; causal CUDA will use prompt_embeds V1/eager."
                )
                if not self.causal_prefix_cache_enabled:
                    logger.warning(
                        "[qwen3-vllm] vLLM prompt-embeds prefix-cache hashing was not "
                        "detected; causal CUDA will fall back to full prompt prefill."
                    )
        elif self.audio_backend == "causal":
            if self.causal_decoder_backend == "vllm-text":
                logger.info(
                    "[qwen3-vllm] causal CUDA will use a standalone vLLM "
                    "Qwen3 text decoder fed by causal audio prompt embeddings."
                )
            elif self.causal_decoder_backend == "vllm-live":
                logger.info(
                    "[qwen3-vllm] causal CUDA will use a live vLLM "
                    "Qwen3 text decoder session with request-local prompt "
                    "embedding append."
                )
            elif self.causal_decoder_backend == "append-kv":
                logger.info(
                    "[qwen3-vllm] causal CUDA will use the append-KV decoder "
                    "backend and vLLM only for ForcedAligner timestamps."
                )
            else:
                logger.info(
                    "[qwen3-vllm] causal CUDA will use the rolling HF decoder KV "
                    "backend and vLLM only for ForcedAligner timestamps."
                )
        max_model_len = int(kwargs.get("vllm_max_model_len") or 0)
        if max_model_len > 0:
            common_kwargs["max_model_len"] = max_model_len

        if use_vllm_asr_decoder:
            logger.info("Loading Qwen3-ASR vLLM model '%s' ...", self.model_path)
            asr_kwargs = {**common_kwargs, **asr_extra_kwargs}
            if self.audio_backend == "causal" and not self.causal_audio_embeds_mm_enabled:
                asr_kwargs["enable_prompt_embeds"] = True
            self.asr_llm = LLM(model=self.model_path, runner="generate", **asr_kwargs)
            self.tokenizer = self.asr_llm.get_tokenizer()
        elif (
            self.audio_backend == "causal"
            and self.causal_decoder_backend in {"vllm-live", "vllm-text"}
        ):
            text_decoder_model = self.causal_text_decoder_model or _ensure_qwen3_asr_text_decoder_model(
                self.model_path
            )
            self.causal_text_decoder_model = text_decoder_model
            logger.info("Loading Qwen3-ASR text decoder vLLM model '%s' ...", text_decoder_model)
            text_decoder_kwargs = {
                "enable_prompt_embeds": True,
                "enable_prefix_caching": True,
            }
            if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0":
                # Two in-process vLLM engines can try to capture new CUDA graph
                # shapes at runtime. In multiprocess mode the engines are
                # isolated, so keep the faster compiled/cudagraph path.
                text_decoder_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}
            if self.causal_decoder_backend == "vllm-live":
                if os.environ.get("WLK_VLLM_LIVE_COMPACT_PROMPT_EMBEDS", "0") == "1":
                    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
                _patch_vllm_mrope_prompt_embeds()
                if not _patch_vllm_live_prompt_embeds_streaming():
                    raise RuntimeError(
                        "qwen3-vllm causal vllm-live requires vLLM V1 "
                        "streaming hooks compatible with prompt_embeds."
                    )
                live_engine_kwargs = dict(common_kwargs)
                if "compilation_config" in text_decoder_kwargs:
                    live_engine_kwargs["compilation_config"] = text_decoder_kwargs[
                        "compilation_config"
                    ]
                self.asr_llm = _VLLMLiveTextDecoder(
                    text_decoder_model,
                    engine_kwargs=live_engine_kwargs,
                )
                self.tokenizer = None
            else:
                self.asr_llm = LLM(
                    model=text_decoder_model,
                    runner="generate",
                    **text_decoder_kwargs,
                    **common_kwargs,
                )
                self.tokenizer = self.asr_llm.get_tokenizer()
        else:
            self.tokenizer = None

        logger.info("Loading Qwen3 ForcedAligner vLLM model '%s' ...", self.aligner_model_path)
        aligner_env_restore = None
        if (
            self.audio_backend == "causal"
            and self.causal_decoder_backend == "vllm-live"
            and os.environ.get("WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING", "0") == "1"
            and os.environ.get("WLK_QWEN3_VLLM_ALIGNER_MULTIPROCESSING", "0") != "1"
        ):
            aligner_env_restore = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        try:
            self.aligner_llm = LLM(
                model=self.aligner_model_path,
                runner="pooling",
                hf_overrides={
                    "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
                },
                **common_kwargs,
            )
        finally:
            if aligner_env_restore is not None:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = aligner_env_restore
            elif (
                self.audio_backend == "causal"
                and self.causal_decoder_backend == "vllm-live"
                and os.environ.get("WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING", "0") == "1"
                and os.environ.get("WLK_QWEN3_VLLM_ALIGNER_MULTIPROCESSING", "0") != "1"
            ):
                os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        self.aligner_tokenizer = self.aligner_llm.get_tokenizer()
        aligner_config = AutoConfig.from_pretrained(self.aligner_model_path)
        timestamp_token_id = getattr(aligner_config, "timestamp_token_id", None)
        if timestamp_token_id is None:
            timestamp_token_id = _token_id(self.aligner_tokenizer, "<timestamp>")
        self.timestamp_token_id = int(timestamp_token_id)
        self.timestamp_segment_time = _normalize_timestamp_segment_time(
            getattr(aligner_config, "timestamp_segment_time", 0.02)
        )
        if self.audio_backend == "causal":
            self._init_causal_audio_backend(
                kwargs,
                dtype=dtype,
                keep_decoder=self.causal_decoder_backend in {"append-kv", "rolling"},
            )
            if self.tokenizer is None and self.causal_asr is not None:
                self.tokenizer = self.causal_asr.qwen_tokenizer

    def _init_causal_audio_backend(
        self,
        kwargs: dict,
        *,
        dtype: str,
        keep_decoder: bool = False,
    ) -> None:
        from .asr import Qwen3StreamingASR

        tower_checkpoint = (
            str(kwargs.get("qwen3_vllm_tower_checkpoint") or "").strip()
            or DEFAULT_QWEN3_VLLM_CAUSAL_TOWER
        )
        left_context_sec = float(kwargs.get("qwen3_vllm_left_context_sec") or 15.0)
        block_frames = int(kwargs.get("qwen3_vllm_block_frames") or 192)
        segment_max_steps = int(
            kwargs.get("qwen3_vllm_segment_max_steps", 150)
            if kwargs.get("qwen3_vllm_segment_max_steps", 150) is not None
            else 150
        )
        prompt_context_words = int(
            kwargs.get("qwen3_vllm_prompt_context_words", 0)
            if kwargs.get("qwen3_vllm_prompt_context_words", 0) is not None
            else 0
        )
        streaming_dtype = _streaming_dtype_from_vllm_dtype(dtype)

        logger.info(
            "Loading Qwen3 causal audio tower '%s' for CUDA/vLLM %s ...",
            tower_checkpoint,
            "rolling decoder KV" if keep_decoder else "audio embeddings",
        )
        self.causal_asr = Qwen3StreamingASR(
            logfile=self.logfile,
            lan=self.original_language,
            model_size=self.model_path,
            qwen3_streaming_audio_backend="causal",
            qwen3_streaming_tower_checkpoint=tower_checkpoint,
            qwen3_streaming_left_context_sec=left_context_sec,
            qwen3_streaming_right_context_ms=0,
            qwen3_streaming_segment_max_steps=segment_max_steps,
            qwen3_streaming_prompt_context_words=prompt_context_words,
            qwen3_streaming_block_frames=block_frames,
            qwen3_streaming_device="cuda",
            qwen3_streaming_dtype=streaming_dtype,
            qwen3_streaming_attn_implementation=self.causal_attn_implementation,
        )
        if keep_decoder:
            self.causal_asr.decoder_rolling_kv = True
            self.causal_asr.speculative_draft = True
            return
        try:
            import torch

            # The CUDA/vLLM path only needs the causal audio tower, adapter, and
            # token embeddings. Drop the HF decoder blocks after loading to
            # leave memory for the two vLLM engines.
            model = self.causal_asr.model
            if hasattr(model, "text_model"):
                model.text_model = None
            if hasattr(model, "lm_head"):
                model.lm_head = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - best-effort memory trim
            logger.debug("[qwen3-vllm] could not drop unused HF decoder: %s", exc)

    def new_causal_session(
        self,
        language: Optional[str] = None,
        *,
        context: str = "",
    ):
        return _Qwen3VLLMCausalSession(self, language=language, context=context)

    def _build_asr_prompt(self, audio: np.ndarray):
        language = _qwen3_language(self.original_language)
        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        if language is None:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\nlanguage {language}{_ASR_TEXT_TAG}"
            )
        return self._TokensPrompt(
            prompt_token_ids=self.tokenizer.encode(prompt),
            multi_modal_data={"audio": audio.astype(np.float32)},
        )

    def transcribe_text(self, audio: np.ndarray) -> tuple[str, Optional[str]]:
        if self.asr_llm is None:
            raise RuntimeError(
                "qwen3-vllm causal append-KV/rolling is a streaming-only decoder path; "
                "use Qwen3VLLMCausalOnlineProcessor or set "
                "qwen3_vllm_causal_decoder_backend='vllm' for batch calls."
            )
        if len(audio) < 400:
            return "", None

        prompt = self._build_asr_prompt(audio)
        params = self._SamplingParams(temperature=0.0, max_tokens=self.max_decode_tokens)
        outputs = self.asr_llm.generate([prompt], params, use_tqdm=False)
        raw_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        language = _detect_qwen3_language(raw_text)
        text = _clean_asr_text(raw_text)
        return text, language

    def _build_aligner_prompt(self, words: list[str], audio: np.ndarray):
        text = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        prompt = "<|audio_start|><|audio_pad|><|audio_end|>" + text
        prompt_token_ids = self.aligner_tokenizer.encode(prompt, add_special_tokens=False)
        return self._TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": audio.astype(np.float32)},
        )

    def align_words(
        self,
        audio: np.ndarray,
        text: str,
        language: Optional[str],
    ) -> list[_AlignedWord]:
        words = _split_align_words(text)
        if not words:
            return []

        prompt = self._build_aligner_prompt(words, audio)
        outputs = self.aligner_llm.encode([prompt], pooling_task="token_classify", use_tqdm=False)
        if not outputs:
            return []

        output = outputs[0]
        data = _to_numpy(output.outputs.data)
        if data.ndim == 3:
            data = data[0]
        pred_ids = np.argmax(data, axis=-1).reshape(-1)
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        if prompt_token_ids is None:
            prompt_token_ids = prompt.get("prompt_token_ids")
        prompt_token_ids = list(prompt_token_ids or [])
        limit = min(len(prompt_token_ids), len(pred_ids))
        timestamp_values = [
            pred_ids[idx] * self.timestamp_segment_time
            for idx in range(limit)
            if int(prompt_token_ids[idx]) == self.timestamp_token_id
        ]
        timestamp_values = _fix_timestamps(timestamp_values)

        aligned = []
        for idx, word in enumerate(words):
            start_idx = idx * 2
            end_idx = start_idx + 1
            if end_idx >= len(timestamp_values):
                break
            aligned.append(
                _AlignedWord(
                    text=word,
                    start=round(float(timestamp_values[start_idx]), 3),
                    end=round(float(timestamp_values[end_idx]), 3),
                )
            )
        return aligned

    def transcribe_aligned(self, audio: np.ndarray) -> tuple[list[_AlignedWord], Optional[str]]:
        text, detected_language = self.transcribe_text(audio)
        if not text:
            return [], detected_language
        language = detected_language or _qwen3_language(self.original_language) or "English"
        return self.align_words(audio, text, language), detected_language

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        text, _ = self.transcribe_text(audio)
        return text

    def use_vad(self):
        return False


class _Qwen3VLLMCausalSession:
    """Per-stream causal audio state feeding Qwen3 vLLM."""

    def __init__(
        self,
        asr: Qwen3VLLMASR,
        language: Optional[str] = None,
        *,
        context: str = "",
    ):
        if asr.causal_asr is None:
            raise RuntimeError("qwen3-vllm causal audio backend is not loaded")
        self.asr = asr
        self.causal_asr = asr.causal_asr
        self.language = language if language and language != "auto" else asr.original_language
        self.qwen_language = self.causal_asr.qwen_language(self.language)
        self.mel = self.causal_asr.new_mel_extractor()
        self.state = self.causal_asr.model.init_cached_audio_decode_state()
        self.frame_hidden = None
        self.audio_samples_seen = 0
        self._flushed = False
        self.cache_salt = f"wlk-qwen3-vllm-causal-{uuid.uuid4().hex}"

        from .model import _split_prompt_template
        from .streamer import qwen_asr_prompt_text

        context_parts = []
        if str(self.causal_asr.base_context or "").strip():
            context_parts.append(str(self.causal_asr.base_context).strip())
        if str(context or "").strip():
            context_parts.append(
                "Previous transcript context:\n" + str(context).strip()
            )
        prompt_template = self.causal_asr.qwen_tokenizer.encode(
            qwen_asr_prompt_text(
                context="\n\n".join(context_parts),
                language=self.qwen_language,
            ),
            add_special_tokens=False,
        )
        self.prompt_template_ids = list(prompt_template)
        self.prompt_head_ids, self.prompt_tail_ids = _split_prompt_template(
            prompt_template,
            self.causal_asr.audio_placeholder_token_id,
        )
        self.last_hypothesis_tokens: list[int] = []
        self.last_generation_stats: dict = {}
        self.decode_prefix_text = ""
        self._vllm_live_prompt_audio_steps = 0
        if getattr(asr, "causal_decoder_backend", None) == "vllm-live":
            if asr.asr_llm is None or not hasattr(asr.asr_llm, "new_session"):
                raise RuntimeError("qwen3-vllm vllm-live decoder is not initialized")
            self.vllm_live_decoder_session = asr.asr_llm.new_session()
        else:
            self.vllm_live_decoder_session = None

    def cached_audio_steps(self) -> int:
        if self.frame_hidden is None:
            return 0
        return int(self.frame_hidden.shape[1])

    def append_audio(self, audio: np.ndarray) -> None:
        if self._flushed and audio.size:
            raise RuntimeError("cannot append audio after flushing a causal session")
        audio = audio.astype(np.float32, copy=False)
        self.audio_samples_seen += int(audio.size)
        frames = self.mel.append(audio)
        if frames is None or int(frames.shape[1]) == 0:
            return
        with self.causal_asr.decode_lock:
            self._append_frames_unlocked(frames)

    def flush_pending(self) -> None:
        if self._flushed:
            return
        with self.causal_asr.decode_lock:
            frames = self.mel.flush()
            if frames is not None and int(frames.shape[1]) > 0:
                self._append_frames_unlocked(frames)
            flush = getattr(self.causal_asr.model, "flush_audio_to_cache", None)
            if flush is not None:
                cached, _delta, self.state = flush(self.state)
                self.frame_hidden = cached
        self._flushed = True

    def _append_frames_unlocked(self, frames) -> None:
        frames = frames.to(self.causal_asr.device)
        cached, _delta, self.state = self.causal_asr.model.append_audio_to_cache(
            frames,
            self.state,
        )
        self.frame_hidden = cached

    def _decode_prefix_token_ids(self) -> list[int]:
        backend = getattr(self.asr, "causal_decoder_backend", "vllm")
        if backend not in {"vllm-live", "vllm-text"}:
            return []
        if not bool(getattr(self.asr, "causal_decode_committed_prefix", False)):
            return []
        text = str(getattr(self, "decode_prefix_text", "") or "").strip()
        if not text:
            return []
        return list(
            self.causal_asr.qwen_tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def _prompt_input(self, *, appended_token_ids: Sequence[int] | None = None):
        if self.frame_hidden is None or int(self.frame_hidden.shape[1]) == 0:
            return None
        import torch

        device = self.causal_asr.device
        with self.causal_asr.decode_lock, torch.no_grad():
            audio_embeds = self.frame_hidden.to(device=device)[0].detach().cpu()
        if getattr(self.asr, "causal_audio_embeds_mm_enabled", False):
            return self._audio_embeds_prompt_input(audio_embeds)

        audio_steps = int(audio_embeds.shape[0])
        decode_prefix_token_ids = self._decode_prefix_token_ids()
        appended_ids = [int(token_id) for token_id in (appended_token_ids or [])]
        prompt_token_ids = (
            list(self.prompt_head_ids)
            + ([self.causal_asr.audio_placeholder_token_id] * audio_steps)
            + list(self.prompt_tail_ids)
            + decode_prefix_token_ids
            + appended_ids
        )
        prompt_is_token_ids = (
            ([True] * len(self.prompt_head_ids))
            + ([False] * audio_steps)
            + ([True] * len(self.prompt_tail_ids))
            + ([True] * len(decode_prefix_token_ids))
            + ([True] * len(appended_ids))
        )
        prompt_embeds = torch.zeros(
            (len(prompt_token_ids), int(audio_embeds.shape[-1])),
            dtype=audio_embeds.dtype,
        )
        if audio_steps:
            start = len(self.prompt_head_ids)
            prompt_embeds[start : start + audio_steps] = audio_embeds
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_token_ids": prompt_token_ids,
            "prompt_is_token_ids": prompt_is_token_ids,
            "is_token_ids": prompt_is_token_ids,
            "cache_salt": self.cache_salt,
        }

    def _vllm_live_prompt_input(
        self,
        *,
        appended_token_ids: Sequence[int] | None = None,
    ):
        prompt_input = self._prompt_input(appended_token_ids=appended_token_ids)
        if prompt_input is None or "prompt_embeds" not in prompt_input:
            return prompt_input
        audio_steps = self.cached_audio_steps()
        previous_audio_steps = int(getattr(self, "_vllm_live_prompt_audio_steps", 0))
        if 0 < previous_audio_steps <= audio_steps:
            prefix_keep_len = len(self.prompt_head_ids) + previous_audio_steps
        else:
            prefix_keep_len = 0
        self._vllm_live_prompt_audio_steps = audio_steps
        tagged = _vllm_live_prompt_input(
            prompt_input,
            prefix_keep_len=prefix_keep_len,
            prompt_head_len=len(self.prompt_head_ids),
            audio_steps=audio_steps,
        )
        # vLLM 0.23 accepts already-processed decoder inputs via type="embeds".
        # This keeps the full per-position text/audio mask intact instead of
        # letting the chat preprocessor rebuild a compact text-only mask.
        tagged["type"] = "embeds"
        prompt_is_token_ids = tagged.get("prompt_is_token_ids")
        if prompt_is_token_ids is not None:
            tagged["is_token_ids"] = list(prompt_is_token_ids)
        return tagged

    def _audio_embeds_prompt_input(self, audio_embeds):
        import torch

        audio_steps = int(audio_embeds.shape[0])
        if audio_steps <= 0:
            return None
        block_steps = max(
            1,
            int(getattr(self.asr, "causal_vllm_audio_embed_block_steps", 16) or 16),
        )
        blocks = [
            audio_embeds[start : start + block_steps].contiguous()
            for start in range(0, audio_steps, block_steps)
        ]
        block_lengths = torch.tensor(
            [int(block.shape[0]) for block in blocks],
            dtype=torch.long,
        )
        fake_audio_feature_lengths = torch.tensor(
            [
                _qwen3_audio_feature_length_for_output_steps(int(length))
                for length in block_lengths.tolist()
            ],
            dtype=torch.long,
        )
        prompt_token_ids = (
            list(self.prompt_head_ids)
            + ([self.causal_asr.audio_placeholder_token_id] * len(blocks))
            + list(self.prompt_tail_ids)
        )
        return {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {
                "audio": {
                    "audio_embeds": torch.cat(blocks, dim=0),
                    "audio_embed_lengths": block_lengths,
                    "audio_feature_lengths": fake_audio_feature_lengths,
                }
            },
            "cache_salt": self.cache_salt,
        }

    def decode_text(self, *, flush: bool = False) -> tuple[str, Optional[str]]:
        if flush:
            self.flush_pending()
        if getattr(self.asr, "causal_decoder_backend", "vllm") in {
            "append-kv",
            "rolling",
        }:
            return self._decode_text_rolling()
        if getattr(self.asr, "causal_decoder_backend", "vllm") == "vllm-live":
            return self._decode_text_vllm_live()
        return self._decode_text_vllm_text()

    def _vllm_text_stop_token_ids(self) -> list[int]:
        eos_token_id = getattr(self.causal_asr, "eos_token_id", None)
        return [] if eos_token_id is None else [int(eos_token_id)]

    def _completion_token_ids(self, output) -> list[int]:
        if output is None or not getattr(output, "outputs", None):
            return []
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is None:
            return []
        return [int(token_id) for token_id in token_ids]

    def _decode_token_ids_to_text(self, token_ids: Sequence[int]) -> str:
        try:
            from .streamer import decode_clean_token_ids

            return decode_clean_token_ids(
                self.causal_asr.qwen_tokenizer,
                list(token_ids),
                wait_token_id=getattr(self.causal_asr, "wait_token_id", None),
                word_start_token_id=getattr(self.causal_asr, "word_start_token_id", None),
            )
        except Exception:
            return self.causal_asr.qwen_tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
            )

    def _vllm_text_sampling_params(
        self,
        *,
        max_tokens: int,
        prompt_logprobs: int | None = None,
    ):
        params_kwargs = {"temperature": 0.0, "max_tokens": max(1, int(max_tokens))}
        stop_token_ids = self._vllm_text_stop_token_ids()
        if stop_token_ids:
            params_kwargs["stop_token_ids"] = stop_token_ids
        if prompt_logprobs is not None:
            params_kwargs["prompt_logprobs"] = int(prompt_logprobs)
        return self.asr._SamplingParams(**params_kwargs)

    def _run_vllm_text_generate(
        self,
        prompt_input: dict,
        *,
        max_tokens: int,
        prompt_logprobs: int | None = None,
    ):
        params = self._vllm_text_sampling_params(
            max_tokens=max_tokens,
            prompt_logprobs=prompt_logprobs,
        )
        outputs = self.asr.asr_llm.generate(
            [prompt_input],
            params,
            use_tqdm=False,
        )
        return outputs[0] if outputs else None

    def _vllm_text_generation_stats(
        self,
        *,
        output,
        prompt_input: dict,
        decode_prefix_text: str,
        decode_prefix_tokens: int,
        extra: dict | None = None,
    ) -> dict:
        prompt_tokens = _prompt_input_token_count(prompt_input)
        cached_tokens = _request_output_num_cached_tokens(output)
        if cached_tokens is None:
            effective_prefill_tokens = None
        else:
            effective_prefill_tokens = max(0, prompt_tokens - cached_tokens)
        stats = {
            "decoder_path": getattr(self.asr, "causal_decoder_backend", "vllm"),
            "decoder_backend": getattr(self.asr, "causal_decoder_backend", "vllm"),
            "prompt_tokens": prompt_tokens,
            "vllm_cached_tokens": cached_tokens,
            "vllm_effective_prefill_tokens": effective_prefill_tokens,
            "audio_steps": self.cached_audio_steps(),
            "decode_prefix_tokens": decode_prefix_tokens,
            "decode_prefix_chars": len(decode_prefix_text),
        }
        if extra:
            stats.update(extra)
        return stats

    def _finish_vllm_text_output(
        self,
        *,
        output,
        decode_prefix_text: str,
        token_ids: Sequence[int] | None = None,
    ) -> tuple[str, Optional[str]]:
        raw_text = output.outputs[0].text if output and output.outputs else ""
        detected_language = _detect_qwen3_language(raw_text) or self.qwen_language
        stop_token_ids = self._vllm_text_stop_token_ids()
        generated_token_ids = _trim_token_ids_at_stop(
            list(token_ids) if token_ids is not None else self._completion_token_ids(output),
            stop_token_ids,
        )
        if generated_token_ids:
            self.last_hypothesis_tokens = generated_token_ids
        else:
            self.last_hypothesis_tokens = []
        if token_ids is not None:
            clean_text = self._decode_token_ids_to_text(generated_token_ids)
        else:
            clean_text = _clean_asr_text(raw_text)
        text = _join_asr_text_prefix(decode_prefix_text, clean_text)
        return text, detected_language

    def _decode_text_vllm_text_without_draft(
        self,
        *,
        prompt_input: dict,
        decode_prefix_text: str,
        decode_prefix_tokens: int,
        extra_stats: dict | None = None,
    ) -> tuple[str, Optional[str]]:
        output = self._run_vllm_text_generate(
            prompt_input,
            max_tokens=self.asr.max_decode_tokens,
        )
        self.last_generation_stats = self._vllm_text_generation_stats(
            output=output,
            prompt_input=prompt_input,
            decode_prefix_text=decode_prefix_text,
            decode_prefix_tokens=decode_prefix_tokens,
            extra=extra_stats,
        )
        return self._finish_vllm_text_output(
            output=output,
            decode_prefix_text=decode_prefix_text,
        )

    def _decode_text_vllm_text_with_draft(
        self,
        *,
        draft_token_ids: Sequence[int],
        decode_prefix_text: str,
        decode_prefix_tokens: int,
    ) -> tuple[str, Optional[str]] | None:
        draft = _trim_token_ids_at_stop(
            draft_token_ids,
            self._vllm_text_stop_token_ids(),
        )
        draft = draft[: max(0, int(self.asr.max_decode_tokens))]
        if not draft:
            return None

        prompt_input = self._prompt_input(appended_token_ids=draft)
        if prompt_input is None:
            return "", self.qwen_language
        base_prompt_tokens = _prompt_input_token_count(prompt_input) - len(draft)
        remaining_tokens = max(1, int(self.asr.max_decode_tokens) - len(draft))
        output = self._run_vllm_text_generate(
            prompt_input,
            max_tokens=remaining_tokens,
            prompt_logprobs=1,
        )
        accepted = _accepted_vllm_prompt_draft_tokens(
            getattr(output, "prompt_logprobs", None),
            draft_start=base_prompt_tokens,
            draft_token_ids=draft,
            prompt_len=_prompt_input_token_count(prompt_input),
        )
        common_stats = {
            "draft_tokens": len(draft),
            "draft_accepted": accepted if accepted is not None else 0,
            "draft_unverifiable": int(accepted is None),
            "draft_all_accepted": accepted == len(draft) if accepted is not None else False,
            "draft_fallback": int(accepted is None or accepted < len(draft)),
            "prefill_positions": max(
                0,
                _prompt_input_token_count(prompt_input)
                - int(_request_output_num_cached_tokens(output) or 0),
            ),
        }
        if accepted is None or accepted < len(draft):
            fallback_prompt = self._prompt_input()
            if fallback_prompt is None:
                return "", self.qwen_language
            return self._decode_text_vllm_text_without_draft(
                prompt_input=fallback_prompt,
                decode_prefix_text=decode_prefix_text,
                decode_prefix_tokens=decode_prefix_tokens,
                extra_stats=common_stats,
            )

        completion_ids = self._completion_token_ids(output)
        combined_ids = (list(draft) + completion_ids)[: int(self.asr.max_decode_tokens)]
        self.last_generation_stats = self._vllm_text_generation_stats(
            output=output,
            prompt_input=prompt_input,
            decode_prefix_text=decode_prefix_text,
            decode_prefix_tokens=decode_prefix_tokens,
            extra=common_stats,
        )
        self.last_generation_stats["decoder_path"] = "vllm-text+draft"
        return self._finish_vllm_text_output(
            output=output,
            decode_prefix_text=decode_prefix_text,
            token_ids=combined_ids,
        )

    def _decode_text_vllm_text(self) -> tuple[str, Optional[str]]:
        prompt_input = self._prompt_input()
        if prompt_input is None:
            return "", self.qwen_language
        decode_prefix_text = str(getattr(self, "decode_prefix_text", "") or "").strip()
        decode_prefix_tokens = len(self._decode_prefix_token_ids())
        draft_enabled = bool(
            getattr(self.asr, "causal_vllm_text_draft_enabled", False)
        )
        if draft_enabled and not decode_prefix_tokens and self.last_hypothesis_tokens:
            drafted = self._decode_text_vllm_text_with_draft(
                draft_token_ids=self.last_hypothesis_tokens,
                decode_prefix_text=decode_prefix_text,
                decode_prefix_tokens=decode_prefix_tokens,
            )
            if drafted is not None:
                return drafted
        return self._decode_text_vllm_text_without_draft(
            prompt_input=prompt_input,
            decode_prefix_text=decode_prefix_text,
            decode_prefix_tokens=decode_prefix_tokens,
        )

    def _run_vllm_live_decode(
        self,
        *,
        appended_token_ids: Sequence[int] | None = None,
        max_tokens: int | None = None,
        prompt_logprobs: int | None = None,
    ):
        import time

        prompt_started_at = time.perf_counter()
        prompt_input = self._vllm_live_prompt_input(
            appended_token_ids=appended_token_ids,
        )
        prompt_build_ms = (time.perf_counter() - prompt_started_at) * 1000.0
        if prompt_input is None:
            return None, None, {}
        if self.vllm_live_decoder_session is None:
            raise RuntimeError("qwen3-vllm vllm-live decoder session is not initialized")

        params_kwargs = {
            "temperature": 0.0,
            "max_tokens": max(1, int(max_tokens or self.asr.max_decode_tokens)),
            "detokenize": False,
        }
        stop_token_ids = self._vllm_text_stop_token_ids()
        if stop_token_ids:
            params_kwargs["stop_token_ids"] = stop_token_ids
        if os.environ.get("WLK_QWEN3_VLLM_LIVE_DELTA_OUTPUT", "1") != "0":
            delta_output_kind = _vllm_request_output_kind("DELTA")
            if delta_output_kind is not None:
                params_kwargs["output_kind"] = delta_output_kind
        if prompt_logprobs is not None:
            params_kwargs["prompt_logprobs"] = int(prompt_logprobs)
        params = self.asr._SamplingParams(**params_kwargs)
        decode_started_at = time.perf_counter()
        output = self.vllm_live_decoder_session.decode(
            prompt_input,
            params,
            idle_timeout=self.asr.causal_vllm_live_idle_timeout_sec,
        )
        vllm_live_decode_wall_ms = (time.perf_counter() - decode_started_at) * 1000.0

        prompt_tokens = _prompt_input_token_count(prompt_input)
        _clean_salt, prefix_keep_len = _decode_vllm_live_cache_salt(
            prompt_input.get("cache_salt")
        )
        prefix_keep_len = max(0, int(prefix_keep_len or 0))
        effective_prefill_tokens = max(0, prompt_tokens - prefix_keep_len)
        audio_steps = self.cached_audio_steps()
        reused_audio_steps = max(
            0,
            min(audio_steps, prefix_keep_len - len(self.prompt_head_ids)),
        )
        stats = {
            "decoder_path": "vllm-live",
            "decoder_backend": "vllm-live",
            "prompt_tokens": prompt_tokens,
            "vllm_cached_tokens": prefix_keep_len,
            "vllm_effective_prefill_tokens": effective_prefill_tokens,
            "vllm_live_prefix_keep_tokens": prefix_keep_len,
            "vllm_live_suffix_prefill_tokens": effective_prefill_tokens,
            "audio_steps": audio_steps,
            "audio_delta_steps": max(0, audio_steps - reused_audio_steps),
            "reused_audio_steps": reused_audio_steps,
            "prompt_head_tokens": len(self.prompt_head_ids),
            "template_tail_tokens": len(self.prompt_tail_ids),
            "vllm_live_returned_on_idle": bool(
                getattr(self.vllm_live_decoder_session, "_last_returned_on_idle", False)
            ),
            "prompt_build_wall_ms": prompt_build_ms,
            "vllm_live_decode_wall_ms": vllm_live_decode_wall_ms,
            "vllm_live_session_wall_ms": float(
                getattr(self.vllm_live_decoder_session, "_last_decode_wall_sec", 0.0)
            )
            * 1000.0,
            "vllm_live_time_to_first_output_ms": float(
                getattr(
                    self.vllm_live_decoder_session,
                    "_last_decode_time_to_first_output_sec",
                    0.0,
                )
            )
            * 1000.0,
            "vllm_live_idle_tail_ms": float(
                getattr(
                    self.vllm_live_decoder_session,
                    "_last_decode_idle_tail_sec",
                    0.0,
                )
            )
            * 1000.0,
            "vllm_live_output_events": int(
                getattr(self.vllm_live_decoder_session, "_last_decode_outputs", 0)
            ),
            "vllm_live_delta_output": bool(
                getattr(self.vllm_live_decoder_session, "_last_delta_output", False)
            ),
            "vllm_live_delta_tokens": int(
                getattr(self.vllm_live_decoder_session, "_last_delta_token_count", 0)
            ),
        }
        return output, prompt_input, stats

    def _decode_text_vllm_live_without_draft(
        self,
        *,
        decode_prefix_text: str,
        decode_prefix_tokens: int,
        extra_stats: dict | None = None,
    ) -> tuple[str, Optional[str]]:
        output, _prompt_input, stats = self._run_vllm_live_decode()
        if output is None:
            return "", self.qwen_language
        stats["decode_prefix_tokens"] = decode_prefix_tokens
        stats["decode_prefix_chars"] = len(decode_prefix_text)
        if extra_stats:
            stats.update(extra_stats)
        self.last_generation_stats = stats
        completion_ids = _trim_token_ids_at_stop(
            self._completion_token_ids(output),
            self._vllm_text_stop_token_ids(),
        )
        if completion_ids and hasattr(self.causal_asr, "qwen_tokenizer"):
            return self._finish_vllm_text_output(
                output=output,
                decode_prefix_text=decode_prefix_text,
                token_ids=completion_ids,
            )
        raw_text = output.outputs[0].text if output and output.outputs else ""
        detected_language = _detect_qwen3_language(raw_text) or self.qwen_language
        self.last_hypothesis_tokens = completion_ids
        text = _join_asr_text_prefix(decode_prefix_text, _clean_asr_text(raw_text))
        return text, detected_language

    def _decode_text_vllm_live_with_draft(
        self,
        *,
        draft_token_ids: Sequence[int],
        decode_prefix_text: str,
        decode_prefix_tokens: int,
    ) -> tuple[str, Optional[str]] | None:
        draft = _trim_token_ids_at_stop(
            draft_token_ids,
            self._vllm_text_stop_token_ids(),
        )
        draft = draft[: max(0, int(self.asr.max_decode_tokens))]
        if not draft:
            return None

        remaining_tokens = max(1, int(self.asr.max_decode_tokens) - len(draft))
        output, prompt_input, stats = self._run_vllm_live_decode(
            appended_token_ids=draft,
            max_tokens=remaining_tokens,
            prompt_logprobs=1,
        )
        if output is None or prompt_input is None:
            return "", self.qwen_language
        base_prompt_tokens = _prompt_input_token_count(prompt_input) - len(draft)
        accepted = _accepted_vllm_prompt_draft_tokens(
            getattr(output, "prompt_logprobs", None),
            draft_start=base_prompt_tokens,
            draft_token_ids=draft,
            prompt_len=_prompt_input_token_count(prompt_input),
        )
        draft_stats = {
            "draft_tokens": len(draft),
            "draft_accepted": accepted if accepted is not None else 0,
            "draft_unverifiable": int(accepted is None),
            "draft_all_accepted": accepted == len(draft) if accepted is not None else False,
            "draft_fallback": int(accepted is None or accepted < len(draft)),
        }
        if accepted is None or accepted < len(draft):
            return self._decode_text_vllm_live_without_draft(
                decode_prefix_text=decode_prefix_text,
                decode_prefix_tokens=decode_prefix_tokens,
                extra_stats=draft_stats,
            )

        completion_ids = self._completion_token_ids(output)
        combined_ids = (list(draft) + completion_ids)[: int(self.asr.max_decode_tokens)]
        stats["decode_prefix_tokens"] = decode_prefix_tokens
        stats["decode_prefix_chars"] = len(decode_prefix_text)
        stats.update(draft_stats)
        stats["decoder_path"] = "vllm-live+draft"
        self.last_generation_stats = stats
        return self._finish_vllm_text_output(
            output=output,
            decode_prefix_text=decode_prefix_text,
            token_ids=combined_ids,
        )

    def _decode_text_vllm_live(self) -> tuple[str, Optional[str]]:
        decode_prefix_text = str(getattr(self, "decode_prefix_text", "") or "").strip()
        decode_prefix_tokens = len(self._decode_prefix_token_ids())
        draft_enabled = bool(
            getattr(self.asr, "causal_vllm_live_draft_enabled", False)
        )
        if draft_enabled and not decode_prefix_tokens and self.last_hypothesis_tokens:
            drafted = self._decode_text_vllm_live_with_draft(
                draft_token_ids=self.last_hypothesis_tokens,
                decode_prefix_text=decode_prefix_text,
                decode_prefix_tokens=decode_prefix_tokens,
            )
            if drafted is not None:
                return drafted
        return self._decode_text_vllm_live_without_draft(
            decode_prefix_text=decode_prefix_text,
            decode_prefix_tokens=decode_prefix_tokens,
        )

    def _decode_text_rolling(self) -> tuple[str, Optional[str]]:
        if self.frame_hidden is None or int(self.frame_hidden.shape[1]) == 0:
            return "", self.qwen_language

        import torch
        from .streamer import (
            _tensor_to_int_list,
            decode_clean_token_ids,
            trim_at_stop,
        )

        with self.causal_asr.decode_lock, torch.no_grad():
            generated, stats = self.causal_asr.model.generate_full_hypothesis_rolling(
                self.frame_hidden,
                state=self.state,
                template_token_ids=self.prompt_template_ids,
                audio_placeholder_token_id=self.causal_asr.audio_placeholder_token_id,
                draft_token_ids=self.last_hypothesis_tokens,
                max_new_tokens=self.asr.max_decode_tokens,
                eos_token_id=self.causal_asr.eos_token_id,
                suppress_token_ids=list(self.causal_asr.suppress_token_ids),
                repetition_penalty=self.causal_asr.repetition_penalty,
                no_repeat_ngram_size=self.causal_asr.no_repeat_ngram_size,
                max_consecutive_text_tokens=int(
                    getattr(self.causal_asr, "max_consecutive_text_tokens", 0)
                ),
            )
            token_ids = trim_at_stop(
                _tensor_to_int_list(generated),
                self.causal_asr.eos_token_id,
            )
        self.last_hypothesis_tokens = token_ids
        backend = getattr(self.asr, "causal_decoder_backend", "rolling")
        self.last_generation_stats = dict(stats or {})
        if backend == "append-kv":
            path = str(self.last_generation_stats.get("decoder_path") or "rolling")
            self.last_generation_stats["decoder_path"] = path.replace(
                "rolling",
                "append-kv",
                1,
            )
        self.last_generation_stats.setdefault("decoder_backend", backend)
        self.last_generation_stats.setdefault("audio_steps", self.cached_audio_steps())
        text = decode_clean_token_ids(
            self.causal_asr.qwen_tokenizer,
            token_ids,
            wait_token_id=self.causal_asr.wait_token_id,
            word_start_token_id=self.causal_asr.word_start_token_id,
        )
        return text, self.qwen_language

    def close(self) -> None:
        session = getattr(self, "vllm_live_decoder_session", None)
        if session is not None:
            session.close()
            self.vllm_live_decoder_session = None


class Qwen3VLLMOnlineProcessor:
    """Batch retranscription processor with ForcedAligner timestamp holdback."""

    SAMPLING_RATE = 16_000
    _HOLDBACK_SECONDS = 0.250
    _MIN_NEW_SECONDS = 1.0
    _MAX_BUFFER_SECONDS = 30.0
    _TRIM_BEFORE_COMMITTED_SECONDS = 2.0
    _COMMITTED_EPSILON = 0.05

    def __init__(self, asr: Qwen3VLLMASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._buffer_time_offset = 0.0
        self._last_committed_time = 0.0
        self._current_tokens: list[ASRToken] = []
        self._samples_since_last_inference = 0
        self._min_new_samples = int(self._MIN_NEW_SECONDS * self.SAMPLING_RATE)

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32))
        self._samples_since_last_inference += len(audio)

    def _audio_duration(self) -> float:
        return len(self.audio_buffer) / self.SAMPLING_RATE

    def _trim_buffer_if_needed(self):
        duration = self._audio_duration()
        if duration <= self._MAX_BUFFER_SECONDS:
            return

        trim_to_time = self._last_committed_time - self._TRIM_BEFORE_COMMITTED_SECONDS
        if trim_to_time <= self._buffer_time_offset:
            return

        cut_samples = int((trim_to_time - self._buffer_time_offset) * self.SAMPLING_RATE)
        if cut_samples <= 0:
            return

        self.audio_buffer = self.audio_buffer[cut_samples:]
        self._buffer_time_offset += cut_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(self._samples_since_last_inference, len(self.audio_buffer))
        self._current_tokens = []

    def _aligned_tokens(self, flush: bool = False) -> list[ASRToken]:
        aligned_words, detected_language = self.asr.transcribe_aligned(self.audio_buffer)
        tokens: list[ASRToken] = []
        for idx, word in enumerate(aligned_words):
            text = word.text if idx == 0 else " " + word.text
            tokens.append(
                ASRToken(
                    start=self._buffer_time_offset + word.start,
                    end=self._buffer_time_offset + word.end,
                    text=text,
                    detected_language=QWEN3_TO_WHISPER_LANGUAGE.get(detected_language, detected_language.lower())
                    if detected_language
                    else None,
                )
            )
        self._current_tokens = tokens
        return tokens

    def _commit_available(self, flush: bool = False) -> list[ASRToken]:
        if len(self.audio_buffer) < 400:
            return []

        self._trim_buffer_if_needed()
        cached_tokens = self._current_tokens
        tokens = self._aligned_tokens(flush=flush)
        if not tokens and flush and cached_tokens:
            tokens = cached_tokens
            self._current_tokens = cached_tokens
        if not tokens:
            return []

        cutoff = (
            self._buffer_time_offset + self._audio_duration()
            if flush
            else self._buffer_time_offset + self._audio_duration() - self._HOLDBACK_SECONDS
        )
        start_idx = 0
        while (
            start_idx < len(tokens)
            and tokens[start_idx].end <= self._last_committed_time + self._COMMITTED_EPSILON
        ):
            start_idx += 1

        end_idx = start_idx
        while end_idx < len(tokens) and tokens[end_idx].end <= cutoff:
            end_idx += 1

        committed = tokens[start_idx:end_idx]
        if committed:
            self._last_committed_time = committed[-1].end
        return committed

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            if not is_last and self._samples_since_last_inference < self._min_new_samples:
                return [], self.end
            self._samples_since_last_inference = 0
            return self._commit_available(flush=is_last), self.end
        except Exception as e:
            logger.warning("[qwen3-vllm] process_iter error: %s", e, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        tokens = [
            token
            for token in self._current_tokens
            if token.end > self._last_committed_time + self._COMMITTED_EPSILON
        ]
        return Transcript.from_tokens(tokens=tokens, sep="")

    def _reset_for_next_utterance(self):
        self._buffer_time_offset += self._audio_duration()
        self._last_committed_time = self._buffer_time_offset
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._current_tokens = []

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        tokens = self._commit_available(flush=True)
        logger.info("[qwen3-vllm] start_silence: flushed %d words", len(tokens))
        self._reset_for_next_utterance()
        return tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._buffer_time_offset += silence_duration
        self._last_committed_time += silence_duration
        self.end += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        return None

    def finish(self) -> Tuple[List[ASRToken], float]:
        tokens = self._commit_available(flush=True)
        logger.info("[qwen3-vllm] finish: flushed %d words", len(tokens))
        return tokens, self.end


class Qwen3VLLMCausalOnlineProcessor(Qwen3VLLMOnlineProcessor):
    """Append-only causal audio processor with vLLM text generation."""

    def __init__(self, asr: Qwen3VLLMASR, logfile=sys.stderr):
        super().__init__(asr, logfile=logfile)
        session_language = getattr(asr, "_session_language", None)
        self._language = session_language or asr.original_language
        self.session = asr.new_causal_session(self._language)
        self._committed_text = ""
        self._session_committed_text = ""
        causal_asr = getattr(asr, "causal_asr", None)
        self._segment_max_cached_steps = int(
            getattr(causal_asr, "segment_max_steps", 0) or 0
        )
        self._segment_min_sec = max(
            0.0,
            float(getattr(asr, "causal_segment_min_sec", 0.0) or 0.0),
        )
        self._segment_context_words = int(
            getattr(causal_asr, "prompt_context_words", 0) or 0
        )
        self._generation_stats: list[dict] = []

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        audio = audio.astype(np.float32, copy=False)
        super().insert_audio_chunk(audio, audio_stream_end_time)
        self.session.append_audio(audio)

    def _trim_buffer_if_needed(self):
        # The causal path first decodes the full append-only hypothesis, then
        # trims this aligner-only buffer only if the committed text prefix still
        # matches and can be dropped from the hypothesis.
        return None

    def _trim_aligner_buffer_to_committed_overlap(
        self,
        *,
        overlap_sec: Optional[float] = None,
    ):
        # The causal audio cache stays append-only in ``self.session``. This
        # buffer is only for ForcedAligner calls, so it may drop committed past.
        overlap = (
            self._TRIM_BEFORE_COMMITTED_SECONDS
            if overlap_sec is None
            else float(overlap_sec)
        )
        trim_to_time = self._last_committed_time - overlap
        if trim_to_time <= self._buffer_time_offset:
            return

        cut_samples = int(
            (trim_to_time - self._buffer_time_offset) * self.SAMPLING_RATE
        )
        if cut_samples <= 0:
            return

        self.audio_buffer = self.audio_buffer[cut_samples:]
        self._buffer_time_offset += cut_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(
            self._samples_since_last_inference,
            len(self.audio_buffer),
        )
        self._current_tokens = []

    def _text_suffix_after_committed(self, text: str) -> tuple[str, bool]:
        text = (text or "").strip()
        committed = (self._session_committed_text or "").strip()
        if not text or not committed:
            return text, False
        if text.startswith(committed):
            return text[len(committed) :].strip(), True

        committed_words = _split_align_words(committed)
        words = _split_align_words(text)
        if len(words) >= len(committed_words) and all(
            _normalize_align_word(left) == _normalize_align_word(right)
            for left, right in zip(committed_words, words, strict=False)
        ):
            return " ".join(words[len(committed_words) :]).strip(), True

        forced_prefix = str(getattr(self.session, "decode_prefix_text", "") or "").strip()
        if forced_prefix:
            forced_words = _split_align_words(forced_prefix)
            if len(words) >= len(forced_words) and all(
                _normalize_align_word(left) == _normalize_align_word(right)
                for left, right in zip(forced_words, words, strict=False)
            ):
                generated_words = words[len(forced_words) :]
                committed_tail = committed_words[len(forced_words) :]
                matched_tail = 0
                for expected, actual in zip(committed_tail, generated_words, strict=False):
                    if _normalize_align_word(expected) != _normalize_align_word(actual):
                        break
                    matched_tail += 1
                return " ".join(generated_words[matched_tail:]).strip(), True
        return text, False

    def _tokens_from_aligned_words(
        self,
        aligned_words: list[_AlignedWord],
        detected_language: Optional[str],
    ) -> list[ASRToken]:
        tokens: list[ASRToken] = []
        prefix_space = bool(self._committed_text.strip())
        for idx, word in enumerate(aligned_words):
            needs_space = idx > 0 or prefix_space
            word_text = (" " if needs_space else "") + word.text
            tokens.append(
                ASRToken(
                    start=self._buffer_time_offset + word.start,
                    end=self._buffer_time_offset + word.end,
                    text=word_text,
                    detected_language=QWEN3_TO_WHISPER_LANGUAGE.get(
                        detected_language,
                        detected_language.lower() if detected_language else None,
                    )
                    if detected_language
                    else None,
                )
            )
        self._current_tokens = tokens
        return tokens

    def _session_cached_steps(self) -> int:
        cached_steps = getattr(self.session, "cached_audio_steps", None)
        if callable(cached_steps):
            return int(cached_steps())
        frame_hidden = getattr(self.session, "frame_hidden", None)
        if frame_hidden is None:
            return 0
        return int(frame_hidden.shape[1])

    def _rollover_context_text(self) -> str:
        text = (self._committed_text or "").strip()
        if not text:
            return ""
        max_words = self._segment_context_words
        if max_words <= 0:
            return ""
        words = text.split()
        return " ".join(words[-max_words:])

    def _decode_forced_prefix_text(self) -> str:
        if not bool(getattr(self.asr, "causal_decode_committed_prefix", False)):
            return ""
        committed = (self._session_committed_text or "").strip()
        if not committed:
            return ""
        overlap_words = max(
            0,
            int(getattr(self.asr, "causal_decode_prefix_overlap_words", 3) or 0),
        )
        words = committed.split()
        if overlap_words <= 0:
            return committed
        if len(words) <= overlap_words:
            return ""
        return " ".join(words[:-overlap_words])

    def _roll_causal_session(self) -> None:
        self._trim_aligner_buffer_to_committed_overlap()
        replay_audio = self.audio_buffer.astype(np.float32, copy=True)
        context = self._rollover_context_text()
        close = getattr(self.session, "close", None)
        if callable(close):
            close()
        self.session = self.asr.new_causal_session(
            self._language,
            context=context,
        )
        self._session_committed_text = ""
        if replay_audio.size:
            self.session.append_audio(replay_audio)
        logger.info(
            "[qwen3-vllm] causal rollover: replayed %.2fs overlap, "
            "context_words=%d, committed_time=%.2fs",
            len(replay_audio) / self.SAMPLING_RATE,
            len(context.split()) if context else 0,
            self._last_committed_time,
        )

    def _maybe_roll_causal_session(self, *, flush: bool = False) -> None:
        if flush or self._segment_max_cached_steps <= 0:
            return
        if not self._committed_text.strip():
            return
        active_audio_sec = (
            float(getattr(self.session, "audio_samples_seen", 0))
            / self.SAMPLING_RATE
        )
        if active_audio_sec <= self._segment_min_sec:
            return
        if self._session_cached_steps() <= self._segment_max_cached_steps:
            return
        self._roll_causal_session()

    def _aligned_tokens(self, flush: bool = False) -> list[ASRToken]:
        import time

        if hasattr(self.session, "decode_prefix_text"):
            self.session.decode_prefix_text = self._decode_forced_prefix_text()
        decode_started_at = time.perf_counter()
        text, detected_language = self.session.decode_text(flush=flush)
        decode_wall_ms = (time.perf_counter() - decode_started_at) * 1000.0
        stats = dict(getattr(self.session, "last_generation_stats", None) or {})
        if stats:
            stats.setdefault("stream_decode_wall_ms", decode_wall_ms)
            stats["audio_buffer_sec"] = len(self.audio_buffer) / self.SAMPLING_RATE
            stats["decode_prefix_overlap_words"] = int(
                getattr(self.asr, "causal_decode_prefix_overlap_words", 0) or 0
            )
        if not text:
            if stats:
                self._generation_stats.append(stats)
            self._current_tokens = []
            return []
        text, dropped_prefix = self._text_suffix_after_committed(text)
        if dropped_prefix:
            self._trim_aligner_buffer_to_committed_overlap()
        if not text:
            if stats:
                stats["align_wall_ms"] = 0.0
                stats["aligned_words"] = 0
                self._generation_stats.append(stats)
            self._current_tokens = []
            return []
        language = detected_language or _qwen3_language(self.asr.original_language) or "English"
        align_started_at = time.perf_counter()
        aligned_words = self.asr.align_words(self.audio_buffer, text, language)
        if stats:
            stats["align_wall_ms"] = (time.perf_counter() - align_started_at) * 1000.0
            stats["aligned_words"] = len(aligned_words)
            self._generation_stats.append(stats)
        return self._tokens_from_aligned_words(aligned_words, detected_language)

    def _commit_available(self, flush: bool = False) -> list[ASRToken]:
        committed = super()._commit_available(flush=flush)
        if committed:
            committed_text = "".join(token.text for token in committed)
            self._committed_text = (self._committed_text + committed_text).strip()
            self._session_committed_text = (
                self._session_committed_text + committed_text
            ).strip()
            self._maybe_roll_causal_session(flush=flush)
        return committed

    def _reset_for_next_utterance(self):
        super()._reset_for_next_utterance()
        close = getattr(self.session, "close", None)
        if callable(close):
            close()
        self.session = self.asr.new_causal_session(self._language)
        self._committed_text = ""
        self._session_committed_text = ""
        self._generation_stats = []

    def finish(self) -> Tuple[List[ASRToken], float]:
        result = super().finish()
        close = getattr(self.session, "close", None)
        if callable(close):
            close()
        return result

    def generation_stats_summary(self) -> dict:
        rows = list(self._generation_stats)
        if not rows:
            return {"decode_calls": 0}

        def values(key: str) -> list[int]:
            return [
                int(row[key])
                for row in rows
                if row.get(key) is not None
            ]

        def float_values(key: str) -> list[float]:
            return [
                float(row[key])
                for row in rows
                if row.get(key) is not None
            ]

        prompt_tokens = values("prompt_tokens")
        cached_tokens = values("vllm_cached_tokens")
        effective_prefill = values("vllm_effective_prefill_tokens")
        audio_steps = values("audio_steps")

        def numeric_summary(key: str) -> dict[str, int | None]:
            vals = values(key)
            return {
                f"{key}_last": vals[-1] if vals else None,
                f"{key}_max": max(vals) if vals else None,
            }

        def float_summary(key: str) -> dict[str, float | None]:
            vals = float_values(key)
            return {
                f"{key}_last": vals[-1] if vals else None,
                f"{key}_max": max(vals) if vals else None,
                f"{key}_mean": sum(vals) / len(vals) if vals else None,
            }

        summary = {
            "decode_calls": len(rows),
            "decoder_path": rows[-1].get("decoder_path"),
            "decoder_backend": rows[-1].get("decoder_backend"),
            "prompt_tokens_last": prompt_tokens[-1] if prompt_tokens else None,
            "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
            "vllm_cached_tokens_last": cached_tokens[-1] if cached_tokens else None,
            "vllm_cached_tokens_max": max(cached_tokens) if cached_tokens else None,
            "vllm_effective_prefill_tokens_last": (
                effective_prefill[-1] if effective_prefill else None
            ),
            "vllm_effective_prefill_tokens_max": (
                max(effective_prefill) if effective_prefill else None
            ),
            "audio_steps_last": audio_steps[-1] if audio_steps else None,
            "audio_steps_max": max(audio_steps) if audio_steps else None,
        }
        for key in (
            "audio_delta_steps",
            "reused_audio_steps",
            "prefill_positions",
            "vllm_live_prefix_keep_tokens",
            "vllm_live_suffix_prefill_tokens",
            "draft_tokens",
            "draft_accepted",
            "draft_unverifiable",
            "draft_fallback",
            "decode_steps",
            "prompt_head_tokens",
            "template_tail_tokens",
            "decode_prefix_tokens",
            "decode_prefix_chars",
            "decode_prefix_overlap_words",
            "aligned_words",
            "vllm_live_output_events",
            "vllm_live_delta_tokens",
        ):
            summary.update(numeric_summary(key))
        for key in (
            "stream_decode_wall_ms",
            "prompt_build_wall_ms",
            "vllm_live_decode_wall_ms",
            "vllm_live_session_wall_ms",
            "vllm_live_time_to_first_output_ms",
            "vllm_live_idle_tail_ms",
            "align_wall_ms",
            "audio_buffer_sec",
        ):
            summary.update(float_summary(key))
        rebuilt = [
            bool(row["decoder_rebuilt"])
            for row in rows
            if "decoder_rebuilt" in row
        ]
        if rebuilt:
            summary["decoder_rebuilds"] = sum(1 for flag in rebuilt if flag)
        accepted = [
            bool(row["draft_all_accepted"])
            for row in rows
            if "draft_all_accepted" in row
        ]
        if accepted:
            summary["draft_all_accepted_calls"] = sum(1 for flag in accepted if flag)
        live_idle = [
            bool(row["vllm_live_returned_on_idle"])
            for row in rows
            if "vllm_live_returned_on_idle" in row
        ]
        if live_idle:
            summary["vllm_live_returned_on_idle_calls"] = sum(
                1 for flag in live_idle if flag
            )
        live_delta = [
            bool(row["vllm_live_delta_output"])
            for row in rows
            if "vllm_live_delta_output" in row
        ]
        if live_delta:
            summary["vllm_live_delta_output_calls"] = sum(
                1 for flag in live_delta if flag
            )
        if prompt_tokens and cached_tokens:
            denom = max(1, prompt_tokens[-1])
            summary["vllm_cached_token_ratio_last"] = cached_tokens[-1] / denom
        return summary
