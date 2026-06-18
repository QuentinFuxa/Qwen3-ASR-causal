import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "macos_qwen3" / "run_wer_rtf.py"


def load_module():
    spec = importlib.util.spec_from_file_location("macos_qwen3_run_wer_rtf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_results_weighted_wer_and_speedup():
    mod = load_module()
    results = [
        {
            "system": "qwen3-metal",
            "label": "Qwen3 metal",
            "backend": "qwen3-vllm-metal",
            "model_size": "0.6b",
            "duration_s": 10.0,
            "stream_wer": 0.2,
            "wer_details": {
                "substitutions": 1,
                "insertions": 1,
                "deletions": 0,
                "ref_words": 10,
                "hyp_words": 10,
            },
            "inference_time_s": 5.0,
            "wall_time_s": 12.0,
            "avg_call_ms": 500.0,
            "p95_call_ms": 800.0,
            "n_transcription_calls": 4,
            "timing_valid": True,
            "timing_monotonic": True,
        },
        {
            "system": "qwen3-causal",
            "label": "Qwen3 causal",
            "backend": "qwen3-streaming",
            "model_size": "0.6b",
            "duration_s": 10.0,
            "stream_wer": 0.1,
            "wer_details": {
                "substitutions": 1,
                "insertions": 0,
                "deletions": 0,
                "ref_words": 10,
                "hyp_words": 10,
            },
            "inference_time_s": 2.0,
            "wall_time_s": 11.0,
            "avg_call_ms": 200.0,
            "p95_call_ms": 300.0,
            "n_transcription_calls": 5,
            "timing_valid": True,
            "timing_monotonic": True,
        },
        {
            "system": "qwen3-causal",
            "label": "Qwen3 causal",
            "backend": "qwen3-streaming",
            "sample": "bad",
            "error": "boom",
        },
    ]

    summary = mod.summarize_results(results)

    assert summary["n_failures"] == 1
    assert summary["systems"]["qwen3-metal"]["weighted_stream_wer"] == 0.2
    assert summary["systems"]["qwen3-metal"]["inference_rtf"] == 0.5
    assert summary["systems"]["qwen3-causal"]["weighted_stream_wer"] == 0.1
    assert summary["systems"]["qwen3-causal"]["inference_rtf"] == 0.2
    assert summary["systems"]["qwen3-causal"]["speedup_vs_qwen3_metal"] == 2.5


def test_markdown_report_mentions_stream_wer_and_rtf():
    mod = load_module()
    summary = {
        "systems": {
            "qwen3-metal": {
                "label": "Qwen3 metal",
                "n_samples": 1,
                "total_audio_s": 10.0,
                "weighted_stream_wer": 0.2,
                "avg_stream_wer": 0.2,
                "inference_rtf": 0.5,
                "wall_rtf": 1.2,
                "speedup_vs_qwen3_metal": 1.0,
                "avg_call_ms": 500.0,
                "calls_per_audio_min": 24.0,
            }
        },
        "failures": [],
    }

    text = mod.markdown_report(
        summary,
        {"platform": "Darwin", "cpu": "Apple M5", "ram_gb": 32},
        {"speed": 1.0, "chunk_duration": 0.5},
    )

    assert "Stream WER" in text
    assert "Inference RTF" in text
    assert "20.00%" in text
    assert "0.500" in text


def test_harness_kwargs_supports_qwen3_metal_causal():
    mod = load_module()
    sample = mod.BenchSample(
        name="short",
        path="/tmp/audio.wav",
        reference="hello world",
        duration=1.0,
        language="en",
    )
    args = mod.parse_args(
        [
            "--systems",
            "qwen3-metal-causal",
            "--metal-model-size",
            "0.6b",
            "--metal-holdback-words",
            "3",
            "--metal-min-chunk-size",
            "0",
            "--metal-dtype",
            "float16",
            "--causal-tower-checkpoint",
            "tower",
            "--causal-left-context-sec",
            "15",
            "--causal-block-frames",
            "192",
        ]
    )

    kwargs = mod.harness_kwargs(
        mod.SYSTEM_SPECS["qwen3-metal-causal"],
        sample,
        args,
    )

    assert kwargs["backend"] == "qwen3-vllm-metal"
    assert kwargs["qwen3_vllm_metal_audio_backend"] == "causal"
    assert kwargs["qwen3_vllm_metal_tower_checkpoint"] == "tower"
    assert kwargs["qwen3_vllm_metal_left_context_sec"] == 15.0
    assert kwargs["qwen3_vllm_metal_block_frames"] == 192
    assert kwargs["holdback_words"] == 3
    assert kwargs["min_chunk_size"] == 0


def test_load_samples_from_json_accepts_long_sample_shape(tmp_path):
    mod = load_module()
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not-real-audio")
    manifest = tmp_path / "samples.json"
    manifest.write_text(
        """
        [
          {
            "name": "long_en",
            "path": "%s",
            "reference": "hello world",
            "duration": 12.5,
            "language": "en",
            "category": "long"
          }
        ]
        """
        % audio_path
    )

    samples = mod.load_samples_from_json(manifest)

    assert len(samples) == 1
    assert samples[0].name == "long_en"
    assert samples[0].reference == "hello world"
    assert samples[0].duration == 12.5
