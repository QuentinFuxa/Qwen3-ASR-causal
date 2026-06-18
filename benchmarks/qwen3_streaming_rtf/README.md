# Qwen3 Streaming RTF

This folder keeps the RTF numbers used for the README / Hugging Face model-card
barplot.

Protocol: streaming ASR generation, excluding model load. The real-streaming WER replay cuts the last 250 ms of live prediction text and preserves the already published prefix; that changes the displayed/scored text, not the model compute time.

These H100 CUDA numbers are HF Transformers runs, not vLLM. The combined public
SVG also includes the vLLM CUDA numbers from `benchmarks/qwen3_vllm_h100/`.

| Hardware / stack | System | RTF | Source |
|---|---:|---:|---|
| Apple M5 Metal | Qwen3-ASR normal | 0.262 | `benchmarks/macos_qwen3/runs/metal_decoder_rolling_en30/results.json` |
| Apple M5 Metal | Qwen3 causal tower | 0.156 | `benchmarks/macos_qwen3/runs/metal_decoder_rolling_en30/results.json` |
| NVIDIA H100 HF Transformers | Qwen3-ASR normal | 0.293 | `experiments/qwen3-causal/runs/jl_20260610/surgery_left12_seg200_chunk2000.summary.json` |
| NVIDIA H100 HF Transformers | Qwen3 causal tower | 0.160 | `/Users/quentin/Downloads/qwen3_checkpoints/ws4_bench_artifacts.tgz:ws4_bench/mcif21_prod_causal.jsonl` |
| NVIDIA A100 vLLM CUDA | Qwen3-ASR normal | 0.146 | `benchmarks/qwen3_vllm_h100/results/a100_normal_jfk_x2.summary.json` |
| NVIDIA A100 vLLM CUDA | Qwen3 causal `vllm-live` | 0.099 | `benchmarks/qwen3_vllm_h100/results/a100_causal_vllm_live_delta_mp_asyncpatch2_jfk_x2.summary.json` |

Regenerate:

```bash
python benchmarks/qwen3_streaming_rtf/generate_combined_rtf_svg.py
```
