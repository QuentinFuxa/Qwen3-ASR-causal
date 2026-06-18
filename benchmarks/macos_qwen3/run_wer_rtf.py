#!/usr/bin/env python3
"""WER/RTF benchmark for Qwen3 streaming variants on macOS.

The benchmark compares:

* qwen3-vllm-metal: the normal Qwen3-ASR vllm-metal streaming backend.
* qwen3-vllm-metal causal: vllm-metal decoder with the fine-tuned causal MLX
  audio tower and rolling decoder KV.
* qwen3-streaming windowed: the normal HF/MPS bounded-recompute backend.
* qwen3-streaming causal: the fine-tuned causal audio tower backend.

It reports the final append-only stream transcript WER and two speed metrics:

* inference_rtf: sum of ASR process_iter calls divided by audio duration.
* wall_rtf: end-to-end feed/drain/finish time divided by audio duration.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TOWER_CHECKPOINT = "qfuxa/qwen3-asr-0.6b-streaming"
DEFAULT_LONG_SAMPLES = (
    Path.home() / ".cache" / "whisperlivekit" / "benchmark_data" / "long_samples.json"
)


@dataclass(frozen=True)
class BenchSample:
    name: str
    path: str
    reference: str
    duration: float
    language: str
    category: str = "unknown"
    source: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SystemSpec:
    key: str
    label: str
    backend: str
    model_size: str
    plot_label: str
    color: str
    note: str


SYSTEM_SPECS = {
    "qwen3-metal": SystemSpec(
        key="qwen3-metal",
        label="Qwen3-ASR 0.6B normal streaming (vLLM Metal)",
        backend="qwen3-vllm-metal",
        model_size="0.6b",
        plot_label="Qwen3 normal\nvLLM Metal",
        color="#2563eb",
        note=(
            "Normal Qwen3-ASR on vllm-metal. The current Metal backend streams by "
            "re-decoding the current audio buffer and holding back trailing words."
        ),
    ),
    "qwen3-metal-causal": SystemSpec(
        key="qwen3-metal-causal",
        label="Qwen3-ASR 0.6B causal audio tower (vLLM Metal)",
        backend="qwen3-vllm-metal",
        model_size="0.6b",
        plot_label="Qwen3 causal\nvLLM Metal",
        color="#7c3aed",
        note=(
            "Experimental vllm-metal path with the fine-tuned append-only MLX "
            "causal audio tower plus rolling decoder KV. Audio blocks are encoded "
            "once; the decoder keeps the [prompt + audio] prefix KV and verifies "
            "the previous hypothesis as a draft."
        ),
    ),
    "qwen3-causal": SystemSpec(
        key="qwen3-causal",
        label="Qwen3-ASR 0.6B causal streaming tower",
        backend="qwen3-streaming",
        model_size="0.6b",
        plot_label="Qwen3 causal\nHF/MPS",
        color="#16a34a",
        note=(
            "Fine-tuned causal-KV audio tower. Each audio block is encoded once; "
            "the stream output is append-only."
        ),
    ),
    "qwen3-windowed": SystemSpec(
        key="qwen3-windowed",
        label="Qwen3-ASR 0.6B normal windowed streaming",
        backend="qwen3-streaming",
        model_size="0.6b",
        plot_label="Qwen3 normal\nwindowed HF/MPS",
        color="#f97316",
        note=(
            "Normal Qwen3-ASR through the HF streaming backend. It re-encodes a "
            "bounded audio window at each update, unlike the causal tower."
        ),
    ),
}


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def _safe_check_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def get_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    cpu = _safe_check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu:
        info["cpu"] = cpu
    mem = _safe_check_output(["sysctl", "-n", "hw.memsize"])
    if mem:
        info["ram_gb"] = round(int(mem) / (1024**3), 1)

    versions: dict[str, str] = {}
    for module_name, label in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("mlx", "mlx"),
        ("vllm_metal", "vllm-metal"),
    ]:
        try:
            module = __import__(module_name)
            versions[label] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            continue
    info["versions"] = versions
    return info


def load_samples_from_json(path: Path) -> list[BenchSample]:
    raw = json.loads(path.expanduser().read_text())
    if isinstance(raw, dict):
        raw_samples = raw.get("samples", [])
    else:
        raw_samples = raw

    samples = []
    for idx, item in enumerate(raw_samples):
        sample_path = item.get("path") or item.get("file")
        reference = item.get("reference") or item.get("text") or item.get("transcript")
        if not sample_path or reference is None:
            raise ValueError(f"sample {idx} in {path} is missing path/file or reference")
        sample_path_obj = Path(sample_path).expanduser()
        if not sample_path_obj.is_absolute():
            sample_path_obj = path.expanduser().parent / sample_path_obj
        sample_path = str(sample_path_obj)
        samples.append(
            BenchSample(
                name=str(item.get("name") or Path(sample_path).stem),
                path=sample_path,
                reference=str(reference),
                duration=float(item.get("duration") or item.get("duration_s") or 0.0),
                language=str(item.get("language") or item.get("lang") or "en"),
                category=str(item.get("category") or "long"),
                source=str(item.get("source") or path.name),
                tags=list(item.get("tags") or []),
            )
        )
    return samples


def load_benchmark_samples(args: argparse.Namespace) -> list[BenchSample]:
    samples_json = args.samples_json
    if args.long_samples and not samples_json:
        samples_json = str(DEFAULT_LONG_SAMPLES)

    if samples_json:
        samples = load_samples_from_json(Path(samples_json))
    else:
        from whisperlivekit.benchmark.datasets import get_benchmark_samples

        loaded = get_benchmark_samples(
            languages=_split_csv(args.languages),
            categories=_split_csv(args.categories),
            quick=args.quick,
            force=args.force_download,
        )
        samples = [
            BenchSample(
                name=s.name,
                path=s.path,
                reference=s.reference,
                duration=s.duration,
                language=s.language,
                category=s.category,
                source=s.source,
                tags=sorted(s.tags),
            )
            for s in loaded
        ]

    languages = set(_split_csv(args.languages) or [])
    categories = set(_split_csv(args.categories) or [])
    if languages:
        samples = [s for s in samples if s.language in languages]
    if categories and samples_json:
        samples = [s for s in samples if s.category in categories]
    sample_names = set(_split_csv(args.sample_names) or [])
    if sample_names:
        samples = [s for s in samples if s.name in sample_names]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no benchmark samples selected")
    return samples


def selected_systems(value: str) -> list[SystemSpec]:
    keys = _split_csv(value) or []
    unknown = [key for key in keys if key not in SYSTEM_SPECS]
    if unknown:
        valid = ", ".join(sorted(SYSTEM_SPECS))
        raise ValueError(f"unknown system(s): {', '.join(unknown)}. Valid: {valid}")
    return [SYSTEM_SPECS[key] for key in keys]


def harness_kwargs(spec: SystemSpec, sample: BenchSample, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "backend": spec.backend,
        "model_size": args.model_size or spec.model_size,
        "lan": sample.language,
        "pcm_input": True,
        "vac": args.vac,
        "vad": args.vad,
        "diarization": False,
    }
    if spec.key == "qwen3-metal":
        kwargs.update(
            {
                "model_size": args.metal_model_size or args.model_size or spec.model_size,
                "holdback_words": args.metal_holdback_words,
                "min_chunk_size": args.metal_min_chunk_size,
                "vllm_dtype": args.metal_dtype,
                "trim_sentence_buffer": args.metal_trim_sentence_buffer,
            }
        )
    elif spec.key == "qwen3-metal-causal":
        kwargs.update(
            {
                "model_size": args.metal_model_size or args.model_size or spec.model_size,
                "holdback_words": args.metal_holdback_words,
                "min_chunk_size": args.metal_min_chunk_size,
                "vllm_dtype": args.metal_dtype,
                "trim_sentence_buffer": args.metal_trim_sentence_buffer,
                "qwen3_vllm_metal_audio_backend": "causal",
                "qwen3_vllm_metal_tower_checkpoint": args.causal_tower_checkpoint,
                "qwen3_vllm_metal_left_context_sec": args.causal_left_context_sec,
                "qwen3_vllm_metal_block_frames": args.causal_block_frames,
            }
        )
    elif spec.key == "qwen3-causal":
        kwargs.update(
            {
                "model_size": args.causal_model_size or args.model_size or spec.model_size,
                "qwen3_streaming_audio_backend": "causal",
                "qwen3_streaming_tower_checkpoint": args.causal_tower_checkpoint,
                "qwen3_streaming_device": args.causal_device,
                "qwen3_streaming_dtype": args.causal_dtype,
                "qwen3_streaming_chunk_sec": args.causal_chunk_sec,
                "qwen3_streaming_left_context_sec": args.causal_left_context_sec,
                "qwen3_streaming_block_frames": args.causal_block_frames,
            }
        )
    elif spec.key == "qwen3-windowed":
        kwargs.update(
            {
                "model_size": args.windowed_model_size or args.model_size or spec.model_size,
                "qwen3_streaming_audio_backend": "windowed",
                "qwen3_streaming_device": args.windowed_device,
                "qwen3_streaming_dtype": args.windowed_dtype,
                "qwen3_streaming_chunk_sec": args.windowed_chunk_sec,
                "qwen3_streaming_left_context_sec": args.windowed_left_context_sec,
                "qwen3_streaming_right_context_ms": args.windowed_right_context_ms,
            }
        )
    return kwargs


def _hypothesis_from_state(state) -> str:
    return (state.committed_text or state.text or "").strip()


async def run_sample(
    spec: SystemSpec,
    sample: BenchSample,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from whisperlivekit.metrics import compute_wer
    from whisperlivekit.test_harness import TestHarness

    kwargs = harness_kwargs(spec, sample, args)
    setup_start = time.perf_counter()
    async with TestHarness(**kwargs) as harness:
        setup_time = time.perf_counter() - setup_start
        wall_start = time.perf_counter()
        feed_start = time.perf_counter()
        await harness.feed(
            sample.path,
            speed=args.speed,
            chunk_duration=args.chunk_duration,
        )
        feed_time = time.perf_counter() - feed_start
        drain_seconds = max(args.min_drain, sample.duration * args.drain_factor)
        if drain_seconds > 0:
            await harness.drain(drain_seconds)
        state = await harness.finish(timeout=args.finish_timeout)
        metrics = harness.metrics
    wall_time = time.perf_counter() - wall_start

    hypothesis = _hypothesis_from_state(state)
    wer = compute_wer(sample.reference, hypothesis)
    transcription_durations = list(getattr(metrics, "transcription_durations", []) or [])
    inference_time = sum(transcription_durations)
    duration = sample.duration if sample.duration > 0 else max(
        getattr(metrics, "total_audio_duration_s", 0.0),
        1e-9,
    )

    return {
        "system": spec.key,
        "label": spec.label,
        "backend": spec.backend,
        "model_size": kwargs.get("model_size", ""),
        "sample": sample.name,
        "path": sample.path,
        "language": sample.language,
        "category": sample.category,
        "source": sample.source,
        "duration_s": round(duration, 3),
        "stream_wer": round(float(wer["wer"]), 6),
        "wer_details": {
            "substitutions": int(wer["substitutions"]),
            "insertions": int(wer["insertions"]),
            "deletions": int(wer["deletions"]),
            "ref_words": int(wer["ref_words"]),
            "hyp_words": int(wer["hyp_words"]),
        },
        "inference_time_s": round(inference_time, 3),
        "inference_rtf": round(inference_time / duration, 6),
        "wall_time_s": round(wall_time, 3),
        "wall_rtf": round(wall_time / duration, 6),
        "setup_time_s": round(setup_time, 3),
        "feed_time_s": round(feed_time, 3),
        "avg_call_ms": round(getattr(metrics, "avg_latency_ms", 0.0), 3),
        "p95_call_ms": round(getattr(metrics, "p95_latency_ms", 0.0), 3),
        "n_transcription_calls": int(getattr(metrics, "n_transcription_calls", 0)),
        "n_chunks_received": int(getattr(metrics, "n_chunks_received", 0)),
        "n_tokens_produced": int(getattr(metrics, "n_tokens_produced", 0)),
        "n_lines": len(state.speech_lines),
        "timing_valid": bool(state.timing_valid),
        "timing_monotonic": bool(state.timing_monotonic),
        "final_buffer": state.buffer_transcription,
        "hypothesis": hypothesis,
        "reference": sample.reference,
    }


async def run_system(
    spec: SystemSpec,
    samples: list[BenchSample],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from whisperlivekit.core import TranscriptionEngine
    from whisperlivekit.test_harness import _engine_cache

    TranscriptionEngine.reset()
    _engine_cache.clear()
    gc.collect()

    results: list[dict[str, Any]] = []
    try:
        for idx, sample in enumerate(samples, start=1):
            print(
                f"[{spec.key}] {idx}/{len(samples)} {sample.name} "
                f"({sample.duration:.1f}s, {sample.language})",
                flush=True,
            )
            try:
                result = await run_sample(spec, sample, args)
            except Exception as exc:
                if not args.keep_going:
                    raise
                result = {
                    "system": spec.key,
                    "label": spec.label,
                    "backend": spec.backend,
                    "sample": sample.name,
                    "path": sample.path,
                    "language": sample.language,
                    "category": sample.category,
                    "duration_s": sample.duration,
                    "error": repr(exc),
                }
            results.append(result)
            if "error" in result:
                print(f"  ERROR {result['error']}", flush=True)
            else:
                print(
                    "  "
                    f"WER={result['stream_wer'] * 100:.1f}% "
                    f"inference_RTF={result['inference_rtf']:.3f} "
                    f"wall_RTF={result['wall_rtf']:.3f} "
                    f"calls={result['n_transcription_calls']}",
                    flush=True,
                )
    finally:
        TranscriptionEngine.reset()
        _engine_cache.clear()
        gc.collect()
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for result in results:
        if result.get("error"):
            failures.append(result)
            continue
        grouped.setdefault(result["system"], []).append(result)

    systems: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        if not rows:
            continue
        total_duration = sum(float(r["duration_s"]) for r in rows)
        total_inference = sum(float(r["inference_time_s"]) for r in rows)
        total_wall = sum(float(r["wall_time_s"]) for r in rows)
        total_ref = sum(int(r["wer_details"]["ref_words"]) for r in rows)
        total_errors = sum(
            int(r["wer_details"]["substitutions"])
            + int(r["wer_details"]["insertions"])
            + int(r["wer_details"]["deletions"])
            for r in rows
        )
        calls = sum(int(r.get("n_transcription_calls", 0)) for r in rows)
        systems[key] = {
            "system": key,
            "label": rows[0]["label"],
            "backend": rows[0]["backend"],
            "model_size": rows[0].get("model_size", ""),
            "plot_label": SYSTEM_SPECS.get(key, SYSTEM_SPECS["qwen3-metal"]).plot_label,
            "color": SYSTEM_SPECS.get(key, SYSTEM_SPECS["qwen3-metal"]).color,
            "n_samples": len(rows),
            "total_audio_s": round(total_duration, 3),
            "weighted_stream_wer": round(total_errors / max(total_ref, 1), 6),
            "avg_stream_wer": round(
                sum(float(r["stream_wer"]) for r in rows) / len(rows),
                6,
            ),
            "total_inference_time_s": round(total_inference, 3),
            "inference_rtf": round(total_inference / max(total_duration, 1e-9), 6),
            "total_wall_time_s": round(total_wall, 3),
            "wall_rtf": round(total_wall / max(total_duration, 1e-9), 6),
            "avg_call_ms": round(
                sum(float(r.get("avg_call_ms", 0.0)) for r in rows) / len(rows),
                3,
            ),
            "p95_call_ms_avg": round(
                sum(float(r.get("p95_call_ms", 0.0)) for r in rows) / len(rows),
                3,
            ),
            "calls_per_audio_min": round(calls / max(total_duration / 60.0, 1e-9), 3),
            "n_transcription_calls": calls,
            "timing_valid": all(bool(r.get("timing_valid", True)) for r in rows),
            "timing_monotonic": all(bool(r.get("timing_monotonic", True)) for r in rows),
        }

    metal = systems.get("qwen3-metal")
    if metal:
        base_rtf = float(metal["inference_rtf"])
        base_wer = float(metal["weighted_stream_wer"])
        for row in systems.values():
            rtf = float(row["inference_rtf"])
            row["speedup_vs_qwen3_metal"] = round(base_rtf / rtf, 3) if rtf > 0 else None
            row["wer_delta_vs_qwen3_metal"] = round(
                float(row["weighted_stream_wer"]) - base_wer,
                6,
            )
    else:
        for row in systems.values():
            row["speedup_vs_qwen3_metal"] = None
            row["wer_delta_vs_qwen3_metal"] = None

    windowed = systems.get("qwen3-windowed")
    if windowed:
        base_rtf = float(windowed["inference_rtf"])
        base_wer = float(windowed["weighted_stream_wer"])
        for row in systems.values():
            rtf = float(row["inference_rtf"])
            row["speedup_vs_qwen3_windowed"] = (
                round(base_rtf / rtf, 3) if rtf > 0 else None
            )
            row["wer_delta_vs_qwen3_windowed"] = round(
                float(row["weighted_stream_wer"]) - base_wer,
                6,
            )
    else:
        for row in systems.values():
            row["speedup_vs_qwen3_windowed"] = None
            row["wer_delta_vs_qwen3_windowed"] = None

    return {
        "systems": systems,
        "failures": failures,
        "n_failures": len(failures),
    }


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows = list(summary["systems"].values())
    fieldnames = [
        "system",
        "label",
        "n_samples",
        "total_audio_s",
        "weighted_stream_wer",
        "avg_stream_wer",
        "inference_rtf",
        "wall_rtf",
        "speedup_vs_qwen3_metal",
        "speedup_vs_qwen3_windowed",
        "avg_call_ms",
        "calls_per_audio_min",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_report(
    summary: dict[str, Any],
    system_info: dict[str, Any],
    config: dict[str, Any],
) -> str:
    rows = list(summary["systems"].values())
    lines = [
        "# Qwen3 macOS streaming WER/RTF",
        "",
        "WER is computed on the final append-only transcript emitted by the streaming "
        "pipeline. `inference_rtf` is ASR compute time divided by audio duration; "
        "`wall_rtf` includes feed, drain and finish time.",
        "",
        f"- Platform: {system_info.get('platform', 'unknown')}",
        f"- CPU: {system_info.get('cpu', 'unknown')}",
        f"- RAM: {system_info.get('ram_gb', 'unknown')} GB",
        f"- Feed speed: {config.get('speed')}",
        f"- Chunk duration: {config.get('chunk_duration')} s",
        "",
        "| System | Samples | Audio | Stream WER | Inference RTF | Wall RTF | Speedup vs Metal | Speedup vs Windowed | Avg call | Calls/min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = row.get("speedup_vs_qwen3_metal")
        speedup_text = "" if speedup is None else f"{speedup:.2f}x"
        windowed_speedup = row.get("speedup_vs_qwen3_windowed")
        windowed_speedup_text = (
            "" if windowed_speedup is None else f"{windowed_speedup:.2f}x"
        )
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['n_samples']} | "
            f"{row['total_audio_s']:.1f}s | "
            f"{row['weighted_stream_wer'] * 100:.2f}% | "
            f"{row['inference_rtf']:.3f} | "
            f"{row['wall_rtf']:.3f} | "
            f"{speedup_text} | "
            f"{windowed_speedup_text} | "
            f"{row['avg_call_ms']:.0f} ms | "
            f"{row['calls_per_audio_min']:.1f} |"
        )

    lines.extend(["", "## Notes", ""])
    for spec in SYSTEM_SPECS.values():
        if spec.key in summary["systems"]:
            lines.append(f"- {spec.label}: {spec.note}")
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- {failure.get('system')} / {failure.get('sample')}: "
                f"{failure.get('error')}"
            )
    lines.append("")
    return "\n".join(lines)


def generate_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(summary["systems"].values())
    if not rows:
        raise RuntimeError("no successful systems to plot")

    labels = [row["plot_label"] for row in rows]
    colors = [row["color"] for row in rows]
    rtfs = [float(row["inference_rtf"]) for row in rows]
    wers = [float(row["weighted_stream_wer"]) * 100.0 for row in rows]

    fig, (ax_rtf, ax_wer) = plt.subplots(1, 2, figsize=(10.5, 4.8), facecolor="white")
    fig.suptitle("Qwen3 streaming on macOS: WER / RTF", fontsize=14, fontweight="bold")

    rtf_bars = ax_rtf.bar(labels, rtfs, color=colors, edgecolor="white", linewidth=1.0)
    ax_rtf.axhline(1.0, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.7)
    ax_rtf.text(
        0.98,
        1.0,
        "real-time boundary",
        transform=ax_rtf.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#ef4444",
    )
    ax_rtf.set_ylabel("Inference RTF (lower is faster)")
    ax_rtf.grid(axis="y", alpha=0.15)
    for bar, value in zip(rtf_bars, rtfs):
        ax_rtf.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    wer_bars = ax_wer.bar(labels, wers, color=colors, edgecolor="white", linewidth=1.0)
    ax_wer.set_ylabel("Stream WER % (lower is better)")
    ax_wer.grid(axis="y", alpha=0.15)
    for bar, value in zip(wer_bars, wers):
        ax_wer.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    for ax in (ax_rtf, ax_wer):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_benchmark_samples(args)
    specs = selected_systems(args.systems)
    print(
        f"Selected {len(samples)} samples, systems={','.join(spec.key for spec in specs)}",
        flush=True,
    )

    all_results: list[dict[str, Any]] = []
    for spec in specs:
        all_results.extend(await run_system(spec, samples, args))

    summary = summarize_results(all_results)
    return {
        "benchmark": "macos_qwen3_streaming_wer_rtf",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system_info": get_system_info(),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "func"
        },
        "samples": [sample.__dict__ for sample in samples],
        "summary": summary,
        "results": all_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default="qwen3-metal,qwen3-windowed,qwen3-causal",
        help=(
            "Comma-separated systems: qwen3-metal,qwen3-metal-causal,"
            "qwen3-windowed,qwen3-causal"
        ),
    )
    parser.add_argument("--output-dir", default="benchmarks/macos_qwen3/latest")
    parser.add_argument("--samples-json", default=None)
    parser.add_argument(
        "--long-samples",
        action="store_true",
        help=f"Use {DEFAULT_LONG_SAMPLES}",
    )
    parser.add_argument("--languages", default="en")
    parser.add_argument("--categories", default="")
    parser.add_argument("--sample-names", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--chunk-duration", type=float, default=0.5)
    parser.add_argument("--min-drain", type=float, default=5.0)
    parser.add_argument("--drain-factor", type=float, default=0.5)
    parser.add_argument("--finish-timeout", type=float, default=240.0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--model-size", default="0.6b")
    parser.add_argument("--vac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vad", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--metal-model-size", default="0.6b")
    parser.add_argument("--metal-holdback-words", type=int, default=2)
    parser.add_argument("--metal-min-chunk-size", type=float, default=1.0)
    parser.add_argument("--metal-dtype", default="auto")
    parser.add_argument(
        "--metal-trim-sentence-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--causal-model-size", default="0.6b")
    parser.add_argument("--causal-tower-checkpoint", default=DEFAULT_TOWER_CHECKPOINT)
    parser.add_argument("--causal-device", default="auto")
    parser.add_argument("--causal-dtype", default="auto")
    parser.add_argument("--causal-chunk-sec", type=float, default=2.0)
    parser.add_argument("--causal-left-context-sec", type=float, default=15.0)
    parser.add_argument("--causal-block-frames", type=int, default=192)

    parser.add_argument("--windowed-model-size", default="0.6b")
    parser.add_argument("--windowed-device", default="auto")
    parser.add_argument("--windowed-dtype", default="auto")
    parser.add_argument("--windowed-chunk-sec", type=float, default=2.0)
    parser.add_argument("--windowed-left-context-sec", type=float, default=12.0)
    parser.add_argument("--windowed-right-context-ms", type=int, default=640)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(run_benchmark(args))
    summary = report["summary"]

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_csv(summary, output_dir / "summary.csv")
    (output_dir / "summary.md").write_text(
        markdown_report(summary, report["system_info"], report["config"])
    )
    generate_plot(summary, output_dir / "wer_rtf.png")

    print(f"Wrote {results_path}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'wer_rtf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
