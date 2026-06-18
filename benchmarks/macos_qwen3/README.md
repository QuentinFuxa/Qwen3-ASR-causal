# Qwen3 macOS Streaming WER/RTF

This folder contains a small benchmark harness for comparing Qwen3 streaming
variants on Apple Silicon:

- `qwen3-metal`: normal Qwen3-ASR through `qwen3-vllm-metal`.
- `qwen3-metal-causal`: experimental `qwen3-vllm-metal` path with the
  fine-tuned causal MLX audio tower plus rolling decoder KV.
- `qwen3-windowed`: normal HF/MPS Qwen3 streaming with bounded window re-encode.
- `qwen3-causal`: fine-tuned causal audio tower, append-only audio encoding.

The benchmark reports:

- `stream_wer`: WER on the final append-only transcript emitted by the streaming
  pipeline.
- `inference_rtf`: sum of ASR calls divided by audio duration.
- `wall_rtf`: feed, drain and finish wall time divided by audio duration.

## Reproduce the saved run

The checked-in `en30_speed4_no_vad/` run intentionally disables VAC/VAD and
disables Metal sentence trimming. That exposes the normal streaming cost where
the Metal backend re-decodes a growing prefix, while the causal tower keeps
audio encoding append-only.

```bash
.venv/bin/python benchmarks/macos_qwen3/run_wer_rtf.py \
  --systems qwen3-metal,qwen3-causal \
  --long-samples \
  --sample-names en_30s \
  --languages en \
  --speed 4 \
  --no-vac --no-vad \
  --no-metal-trim-sentence-buffer \
  --min-drain 0 --drain-factor 0 \
  --finish-timeout 600 \
  --keep-going \
  --output-dir benchmarks/macos_qwen3/en30_speed4_no_vad
```

Current saved result on Apple M5:

| System | Stream WER | Inference RTF | Speedup vs Metal |
|---|---:|---:|---:|
| Qwen3 normal vLLM Metal | 7.69% | 0.357 | 1.00x |
| Qwen3 causal HF/MPS | 15.38% | 0.222 | 1.61x |

`qwen3-metal-causal` should be measured with sentence trimming enabled, because
the causal tower is trained for bounded sentence-like segments. A local
`en_30s` smoke with the same no-VAC/no-VAD feed and default Metal trimming,
after adding rolling decoder KV, scored 17.95% WER at 0.156 inference RTF
versus 8.97% WER at 0.262 inference RTF for normal `qwen3-metal` (1.68x faster).

The default production path with VAC/VAD and sentence trimming can be faster
for the Metal backend on silence-rich audio; keep both protocols separate when
reading the numbers.
