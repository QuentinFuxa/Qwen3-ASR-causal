# Qwen3 macOS streaming WER/RTF

WER is computed on the final append-only transcript emitted by the streaming pipeline. `inference_rtf` is ASR compute time divided by audio duration; `wall_rtf` includes feed, drain and finish time.

- Platform: macOS-26.5.1-arm64-arm-64bit
- CPU: Apple M5
- RAM: 32.0 GB
- Feed speed: 4.0
- Chunk duration: 0.5 s

| System | Samples | Audio | Stream WER | Inference RTF | Wall RTF | Speedup vs Metal | Speedup vs Windowed | Avg call | Calls/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR 0.6B normal streaming (vLLM Metal) | 1 | 30.0s | 7.69% | 0.357 | 0.422 | 1.00x |  | 714 ms | 30.0 |
| Qwen3-ASR 0.6B causal streaming tower | 1 | 30.0s | 15.38% | 0.222 | 0.300 | 1.61x |  | 196 ms | 68.0 |

## Notes

- Qwen3-ASR 0.6B normal streaming (vLLM Metal): Normal Qwen3-ASR on vllm-metal. The current Metal backend streams by re-decoding the current audio buffer and holding back trailing words.
- Qwen3-ASR 0.6B causal streaming tower: Fine-tuned causal-KV audio tower. Each audio block is encoded once; the stream output is append-only.
