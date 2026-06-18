#!/usr/bin/env python3
"""Measure Qwen3-ASR streaming RTF on an NVIDIA H100 through vLLM."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _resample_if_needed(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return audio
    try:
        import librosa
    except ImportError as exc:
        raise SystemExit(
            f"Audio {sample_rate} Hz must be resampled to {target_rate} Hz; install librosa."
        ) from exc
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate).astype(np.float32)


def _cuda_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark must run on an NVIDIA H100 host.")
    index = torch.cuda.current_device()
    return {
        "device_index": index,
        "name": torch.cuda.get_device_name(index),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _cuda_name_without_torch() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return names[0] if names else None


def _should_delay_cuda_init(args: argparse.Namespace) -> bool:
    return (
        args.mode == "causal"
        and args.qwen3_vllm_causal_decoder_backend == "vllm-live"
    )


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_vllm_version() -> str | None:
    try:
        import vllm
    except ImportError:
        return None
    return getattr(vllm, "__version__", None)


def _build_asr(args: argparse.Namespace):
    from qwen3_asr_causal.vllm import Qwen3VLLMASR

    audio_backend = "causal" if args.mode == "causal" else "standard"
    return Qwen3VLLMASR(
        lan=args.language,
        model_size=args.model_size,
        vllm_model=args.vllm_model or None,
        vllm_aligner_model=args.vllm_aligner_model or None,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        qwen3_vllm_audio_backend=audio_backend,
        qwen3_vllm_causal_decoder_backend=args.qwen3_vllm_causal_decoder_backend,
        qwen3_vllm_causal_attn_implementation=args.qwen3_vllm_causal_attn_implementation,
        qwen3_vllm_text_decoder_model=args.qwen3_vllm_text_decoder_model,
        qwen3_vllm_live_idle_timeout_ms=args.qwen3_vllm_live_idle_timeout_ms,
        qwen3_vllm_tower_checkpoint=args.qwen3_vllm_tower_checkpoint,
        qwen3_vllm_left_context_sec=args.qwen3_vllm_left_context_sec,
        qwen3_vllm_block_frames=args.qwen3_vllm_block_frames,
        qwen3_vllm_cache_block_size=args.qwen3_vllm_cache_block_size,
        qwen3_vllm_segment_max_steps=args.qwen3_vllm_segment_max_steps,
        qwen3_vllm_segment_min_sec=args.qwen3_vllm_segment_min_sec,
        qwen3_vllm_prompt_context_words=args.qwen3_vllm_prompt_context_words,
        max_tokens=args.max_tokens,
    )


def _build_processor(args: argparse.Namespace, asr):
    from qwen3_asr_causal.vllm import (
        Qwen3VLLMCausalOnlineProcessor,
        Qwen3VLLMOnlineProcessor,
    )

    if args.mode == "causal":
        return Qwen3VLLMCausalOnlineProcessor(asr)
    return Qwen3VLLMOnlineProcessor(asr)


def _row_audio_path(row: dict[str, Any]) -> str:
    path = row.get("audio") or row.get("wav") or row.get("path")
    if not path:
        raise ValueError(f"Manifest row has no audio/wav/path field: {row}")
    return str(path)


def _transcribe_streaming(
    args: argparse.Namespace,
    asr,
    audio: np.ndarray,
) -> tuple[str, float, dict[str, Any]]:
    processor = _build_processor(args, asr)
    sample_rate = int(getattr(processor, "SAMPLING_RATE", 16_000))
    chunk_samples = max(1, int(args.chunk_sec * sample_rate))

    _sync_cuda()
    start = time.perf_counter()
    stream_time = 0.0
    committed_text: list[str] = []
    for offset in range(0, len(audio), chunk_samples):
        chunk = audio[offset : offset + chunk_samples]
        stream_time += len(chunk) / sample_rate
        processor.insert_audio_chunk(chunk, stream_time)
        tokens, _ = processor.process_iter(is_last=False)
        committed_text.extend(token.text for token in tokens)

    tokens, _ = processor.finish()
    committed_text.extend(token.text for token in tokens)
    _sync_cuda()
    latency_sec = time.perf_counter() - start
    generation_summary = {}
    summarize = getattr(processor, "generation_stats_summary", None)
    if callable(summarize):
        generation_summary = summarize()
    return "".join(committed_text).strip(), latency_sec, generation_summary


def _summarize_generation_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

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

    keys = (
        "decode_calls",
        "prompt_tokens_last",
        "prompt_tokens_max",
        "vllm_cached_tokens_last",
        "vllm_cached_tokens_max",
        "vllm_effective_prefill_tokens_last",
        "vllm_effective_prefill_tokens_max",
        "audio_steps_last",
        "audio_steps_max",
        "audio_delta_steps_last",
        "audio_delta_steps_max",
        "reused_audio_steps_last",
        "reused_audio_steps_max",
        "vllm_live_prefix_keep_tokens_last",
        "vllm_live_prefix_keep_tokens_max",
        "vllm_live_suffix_prefill_tokens_last",
        "vllm_live_suffix_prefill_tokens_max",
        "prefill_positions_last",
        "prefill_positions_max",
        "draft_tokens_last",
        "draft_tokens_max",
        "draft_accepted_last",
        "draft_accepted_max",
        "draft_unverifiable_last",
        "draft_unverifiable_max",
        "draft_fallback_last",
        "draft_fallback_max",
        "decode_steps_last",
        "decode_steps_max",
        "prompt_head_tokens_last",
        "prompt_head_tokens_max",
        "template_tail_tokens_last",
        "template_tail_tokens_max",
        "decode_prefix_tokens_last",
        "decode_prefix_tokens_max",
        "decode_prefix_chars_last",
        "decode_prefix_chars_max",
        "decode_prefix_overlap_words_last",
        "decode_prefix_overlap_words_max",
        "aligned_words_last",
        "aligned_words_max",
        "vllm_live_output_events_last",
        "vllm_live_output_events_max",
        "vllm_live_delta_tokens_last",
        "vllm_live_delta_tokens_max",
        "decoder_rebuilds",
        "draft_all_accepted_calls",
        "vllm_live_returned_on_idle_calls",
        "vllm_live_delta_output_calls",
    )
    out: dict[str, Any] = {}
    for key in keys:
        vals = values(key)
        if vals:
            out[f"generation_{key}_max"] = max(vals)
            out[f"generation_{key}_mean"] = sum(vals) / len(vals)
    float_keys = (
        "stream_decode_wall_ms_last",
        "stream_decode_wall_ms_max",
        "stream_decode_wall_ms_mean",
        "prompt_build_wall_ms_last",
        "prompt_build_wall_ms_max",
        "prompt_build_wall_ms_mean",
        "vllm_live_decode_wall_ms_last",
        "vllm_live_decode_wall_ms_max",
        "vllm_live_decode_wall_ms_mean",
        "vllm_live_session_wall_ms_last",
        "vllm_live_session_wall_ms_max",
        "vllm_live_session_wall_ms_mean",
        "vllm_live_time_to_first_output_ms_last",
        "vllm_live_time_to_first_output_ms_max",
        "vllm_live_time_to_first_output_ms_mean",
        "vllm_live_idle_tail_ms_last",
        "vllm_live_idle_tail_ms_max",
        "vllm_live_idle_tail_ms_mean",
        "align_wall_ms_last",
        "align_wall_ms_max",
        "align_wall_ms_mean",
        "audio_buffer_sec_last",
        "audio_buffer_sec_max",
        "audio_buffer_sec_mean",
    )
    for key in float_keys:
        vals = float_values(key)
        if vals:
            out[f"generation_{key}_max"] = max(vals)
            out[f"generation_{key}_mean"] = sum(vals) / len(vals)
    ratios = [
        float(row["vllm_cached_token_ratio_last"])
        for row in rows
        if row.get("vllm_cached_token_ratio_last") is not None
    ]
    if ratios:
        out["generation_vllm_cached_token_ratio_last_mean"] = sum(ratios) / len(ratios)
        out["generation_vllm_cached_token_ratio_last_min"] = min(ratios)
    decoder_paths = sorted({str(row.get("decoder_path")) for row in rows if row.get("decoder_path")})
    if decoder_paths:
        out["generation_decoder_paths"] = decoder_paths
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "causal"), required=True)
    parser.add_argument("--model-size", default="0.6b")
    parser.add_argument("--language", default="en")
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--vllm-model", default="")
    parser.add_argument("--vllm-aligner-model", default="")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=0)
    parser.add_argument(
        "--qwen3-vllm-causal-decoder-backend",
        choices=("append-kv", "rolling", "vllm", "vllm-live", "vllm-text"),
        default="vllm-text",
    )
    parser.add_argument(
        "--qwen3-vllm-causal-attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--qwen3-vllm-tower-checkpoint", default="")
    parser.add_argument("--qwen3-vllm-text-decoder-model", default="")
    parser.add_argument("--qwen3-vllm-live-idle-timeout-ms", type=float, default=50.0)
    parser.add_argument("--qwen3-vllm-left-context-sec", type=float, default=15.0)
    parser.add_argument("--qwen3-vllm-block-frames", type=int, default=192)
    parser.add_argument("--qwen3-vllm-cache-block-size", type=int, default=0)
    parser.add_argument("--qwen3-vllm-segment-max-steps", type=int, default=150)
    parser.add_argument("--qwen3-vllm-segment-min-sec", type=float, default=0.0)
    parser.add_argument("--qwen3-vllm-prompt-context-words", type=int, default=0)
    parser.add_argument("--allow-non-h100", action="store_true")
    args = parser.parse_args()

    cuda = None
    if _should_delay_cuda_init(args):
        cuda_name = _cuda_name_without_torch()
        if cuda_name is None:
            raise SystemExit("CUDA is not visible through nvidia-smi.")
        if not args.allow_non_h100 and "H100" not in cuda_name.upper():
            raise SystemExit(
                f"Expected an H100, got {cuda_name!r}. "
                "Use --allow-non-h100 for smoke tests."
            )
    else:
        cuda = _cuda_info()
        if not args.allow_non_h100 and "H100" not in cuda["name"].upper():
            raise SystemExit(
                f"Expected an H100, got {cuda['name']!r}. "
                "Use --allow-non-h100 for smoke tests."
            )

    rows = [
        json.loads(line)
        for line in args.manifest_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{args.manifest_jsonl} is empty")

    asr = _build_asr(args)
    if cuda is None:
        cuda = _cuda_info()
    sample_rate = int(getattr(asr, "SAMPLING_RATE", 16_000))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    latencies: list[float] = []
    generation_rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            path = _row_audio_path(row)
            audio, audio_sample_rate = _load_audio(path)
            audio = _resample_if_needed(audio, audio_sample_rate, sample_rate)
            audio_sec = len(audio) / sample_rate
            text, latency_sec, generation_summary = _transcribe_streaming(args, asr, audio)
            rtf = latency_sec / audio_sec if audio_sec > 0.0 else None
            durations.append(audio_sec)
            latencies.append(latency_sec)
            if generation_summary:
                generation_rows.append(generation_summary)

            result = {
                "audio": path,
                "mode": args.mode,
                "text": text,
                "audio_sec": audio_sec,
                "latency_sec": latency_sec,
                "realtime_factor": rtf,
            }
            if generation_summary:
                result["generation_stats"] = generation_summary
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{idx}/{len(rows)}] mode={args.mode} "
                f"rtf={rtf:.4f} audio={audio_sec:.2f}s latency={latency_sec:.2f}s {Path(path).name}",
                flush=True,
            )

    total_audio = sum(durations)
    total_latency = sum(latencies)
    if args.mode == "causal" and args.qwen3_vllm_causal_decoder_backend == "append-kv":
        measurement_stack = "qwen3-vllm CUDA: append-KV ASR + vLLM ForcedAligner"
    elif args.mode == "causal" and args.qwen3_vllm_causal_decoder_backend == "rolling":
        measurement_stack = "qwen3-vllm CUDA: HF rolling ASR + vLLM ForcedAligner"
    elif args.mode == "causal" and args.qwen3_vllm_causal_decoder_backend == "vllm-text":
        measurement_stack = "qwen3-vllm CUDA: vLLM text decoder + HF causal audio + vLLM ForcedAligner"
    elif args.mode == "causal" and args.qwen3_vllm_causal_decoder_backend == "vllm-live":
        measurement_stack = "qwen3-vllm CUDA: live vLLM text decoder append + HF causal audio + vLLM ForcedAligner"
    else:
        measurement_stack = "vLLM in-process CUDA"

    summary = {
        "mode": args.mode,
        "measurement_stack": measurement_stack,
        "measurement_protocol": "streaming no-past-rewrite with 250 ms holdback",
        "hardware": cuda["name"],
        "platform": platform.platform(),
        "torch_version": cuda["torch_version"],
        "cuda_version": cuda["cuda_version"],
        "vllm_version": _load_vllm_version(),
        "model_size": args.model_size,
        "language": args.language,
        "chunk_sec": args.chunk_sec,
        "vllm_max_model_len": args.vllm_max_model_len,
        "qwen3_vllm_audio_backend": "causal" if args.mode == "causal" else "standard",
        "qwen3_vllm_causal_decoder_backend": (
            args.qwen3_vllm_causal_decoder_backend
            if args.mode == "causal"
            else None
        ),
        "qwen3_vllm_causal_attn_implementation": (
            args.qwen3_vllm_causal_attn_implementation
            if args.mode == "causal"
            else None
        ),
        "qwen3_vllm_text_decoder_model": (
            args.qwen3_vllm_text_decoder_model
            if args.mode == "causal"
            else None
        ),
        "qwen3_vllm_live_idle_timeout_ms": (
            args.qwen3_vllm_live_idle_timeout_ms
            if args.mode == "causal"
            and args.qwen3_vllm_causal_decoder_backend == "vllm-live"
            else None
        ),
        "qwen3_vllm_tower_checkpoint": args.qwen3_vllm_tower_checkpoint,
        "qwen3_vllm_left_context_sec": args.qwen3_vllm_left_context_sec,
        "qwen3_vllm_block_frames": args.qwen3_vllm_block_frames,
        "qwen3_vllm_cache_block_size": args.qwen3_vllm_cache_block_size,
        "qwen3_vllm_segment_max_steps": args.qwen3_vllm_segment_max_steps,
        "qwen3_vllm_segment_min_sec": args.qwen3_vllm_segment_min_sec,
        "qwen3_vllm_prompt_context_words": args.qwen3_vllm_prompt_context_words,
        "count": len(rows),
        "audio_duration_total_sec": total_audio,
        "latency_total_sec": total_latency,
        "realtime_factor_total": total_latency / total_audio if total_audio > 0.0 else None,
        "realtime_factor_mean": (
            sum(lat / dur for lat, dur in zip(latencies, durations, strict=True)) / len(rows)
            if rows
            else None
        ),
    }
    summary.update(_summarize_generation_stats(generation_rows))
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
