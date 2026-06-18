# Qwen3 vLLM NVIDIA RTF

This benchmark tracks the real `qwen3-vllm` normal-vs-causal comparison for the
README/model card. It must not reuse the HF Transformers/CUDA causal number from
`benchmarks/qwen3_streaming_rtf`: useful number, wrong stack.

## Current Result

Measured on JarvisLabs `NVIDIA A100-PCIE-40GB` on 2026-06-14 with vLLM 0.23.0,
Torch 2.11.0+cu129, CUDA 12.9, `Qwen/Qwen3-ASR-0.6B`, 1.0 s streaming chunks,
and a 22 s repeated JFK clip (`data/jfk_x2.wav`). Model load time is excluded.

| Hardware / stack | Mode | RTF | Notes |
|---|---:|---:|---|
| Apple M5 vLLM Metal | normal | 0.262 | `en_30s` local smoke |
| Apple M5 vLLM Metal | causal | 0.156 | 1.68x faster |
| NVIDIA A100 vLLM CUDA | normal | 0.146 | real streaming, ForcedAligner timestamps |
| NVIDIA A100 vLLM CUDA | causal `vllm-live` | 0.099 | 1.48x faster; live vLLM prompt-embeds append + async/CUDA graphs |
| NVIDIA A100 vLLM CUDA | causal `vllm-text` | 0.113 | 1.30x faster; vLLM text decoder prefix-cache |
| NVIDIA A100 vLLM CUDA | causal append-KV | 0.150 | persistent decoder KV + vLLM ForcedAligner |
| NVIDIA A100 vLLM CUDA | causal rolling | 0.160 | old name/path before append-KV instrumentation |
| NVIDIA A100 vLLM CUDA | causal vLLM `audio_embeds` | 0.412 | legacy pure-vLLM causal path |

![Qwen3 combined streaming RTF](../../assets/rtf_combined.svg)

The fastest CUDA causal path is now `vllm-live` with
`WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1`: the causal audio tower encodes new
blocks incrementally, one live vLLM text-decoder request keeps the audio KV
prefix, and streaming updates append only the new audio suffix plus prompt
tail. The async scheduler patch handles stale/excess output placeholders, so
the decoder can use vLLM's multiprocessing/CUDA graph path. The forced aligner
is kept in-process by default for this mode to avoid running two independent
vLLM EngineCore processes. On the A100/JFK x2 smoke this measured 0.099 RTF,
versus 0.146 for normal `qwen3-vllm`.

The `vllm-text` backend remains a conservative fallback: it uses prompt
embeddings and vLLM prefix caching in one request per chunk, but not a live
request-local append. It measured 0.113 RTF on the same 22 s smoke and 0.104 RTF
on a repeated 110 s JFK smoke.

The `append-kv` backend is the explicit name for the old `rolling` KV path and
measured 0.150 RTF on the 22 s smoke: it keeps decoder KV over
`[prompt head + audio]` and forwards only new audio steps plus the shifted
prompt tail and previous draft. On the final decode of that smoke,
`reused_audio_steps=120`, `audio_delta_steps=16`, and `prefill_positions=47`,
so it is not refilling the full 136-step audio prefix. The legacy pure-vLLM
`audio_embeds` path
measured 0.412 RTF and still needs `cudagraph_mode=NONE` in vLLM 0.23. These
are speed smokes, not WER claims; use the long-form WER runs in the main README
for quality numbers.

If a plot shows causal CUDA slower than normal, it is almost certainly plotting
one of these legacy/probe paths instead of the current `vllm-live` or
`vllm-text` result. The old `audio_embeds`/V1 eager shim exercises vLLM in a
much less optimized request pattern, so it is a useful diagnostic but not the
CUDA causal speed number.

Cache instrumentation added on 2026-06-14 confirms what vLLM is actually doing:
on the 22 s JFK smoke, the final decode prompt had 160 tokens, 128 were vLLM
prefix-cache hits, and only 32 were scheduled as effective prefill; the maximum
effective prefill over the stream was 66 tokens. On the repeated 110 s smoke,
the same maxima held (`prompt_tokens_max=186`,
`vllm_effective_prefill_tokens_max=66`) and RTF measured 0.104. So the current
`vllm-text` path is no longer refilling the whole `[prompt + audio embeddings]`
prefix on every chunk, but it is still not a true live append: vLLM recomputes
the last partial block/tail at cache-block granularity.

A follow-up cache-block test found that `--qwen3-vllm-cache-block-size 8` is not
supported by vLLM 0.23 CUDA attention backends, and `block_size=16` gives the
same effective-prefill counters as the default (`last=32`, `max=66`). So the
default CUDA path is already at the usable vLLM block granularity; further
speedup needs a real append/KV path rather than a smaller cache block.

An opt-in `WLK_QWEN3_VLLM_TEXT_DRAFT=1` experiment verifies the previous
hypothesis with vLLM `prompt_logprobs` before reusing it as a draft. It was
lossless on the JFK x2 smoke but slower (`0.174` RTF) because rejected drafts
need a fallback request and `prompt_logprobs` adds enough overhead to erase the
token savings. It stays disabled by default.

The closer live-request experiment,
`WLK_QWEN3_VLLM_LIVE_DRAFT=1 --qwen3-vllm-causal-decoder-backend vllm-live`,
does crop a single vLLM streaming request back to the audio KV prefix
(`vllm_effective_prefill_tokens_last=9`), but the previous full-hypothesis draft
was unverifiable/rejected on every JFK x2 chunk and the eager live runner
measured `0.352` RTF after the prompt-logprob index fix. It is a useful
implementation probe, not a publishable speed path.

The non-draft `vllm-live` path uses vLLM `RequestOutputKind.DELTA` with local
token reconstruction, so every chunk avoids cumulative detokenized output
traffic (`generation_vllm_live_delta_output_calls=22`). In eager/in-process mode
it measured 0.280 RTF. With `WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1`, the live
decoder reached `cudagraph_mode=FULL_AND_PIECEWISE` and the async scheduler
patch removed the old `num_output_placeholders >= 0` crash; the same JFK x2
stream measured 0.099 RTF.

## Protocol

The runner uses the live streaming processor, not offline full-file decoding:

- feed audio in fixed chunks;
- run inference incrementally;
- commit only words ending outside the last 250 ms;
- flush the final 250 ms only at end of stream;
- never rewrite already committed text.

RTF is total ASR wall time divided by total audio duration, with CUDA
synchronized before and after each file. Model load time is excluded.

## Commands

Normal vLLM:

```bash
python benchmarks/qwen3_vllm_h100/run_vllm_h100.py \
  --manifest-jsonl data/manifest_jfk_x2.jsonl \
  --output-jsonl benchmarks/qwen3_vllm_h100/results/a100_normal_jfk_x2.jsonl \
  --mode normal \
  --model-size 0.6b \
  --language en \
  --chunk-sec 1.0 \
  --max-tokens 256 \
  --vllm-gpu-memory-utilization 0.32 \
  --vllm-max-model-len 8192 \
  --allow-non-h100
```

Causal vLLM:

```bash
WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1 \
python benchmarks/qwen3_vllm_h100/run_vllm_h100.py \
  --manifest-jsonl data/manifest_jfk_x2.jsonl \
  --output-jsonl benchmarks/qwen3_vllm_h100/results/a100_causal_vllm_live_delta_mp_asyncpatch2_jfk_x2.jsonl \
  --mode causal \
  --model-size 0.6b \
  --language en \
  --chunk-sec 1.0 \
  --max-tokens 256 \
  --vllm-gpu-memory-utilization 0.32 \
  --vllm-max-model-len 2048 \
  --qwen3-vllm-causal-decoder-backend vllm-live \
  --qwen3-vllm-live-idle-timeout-ms 20 \
  --qwen3-vllm-tower-checkpoint qfuxa/qwen3-asr-0.6b-streaming \
  --qwen3-vllm-segment-max-steps 150 \
  --qwen3-vllm-segment-min-sec 0 \
  --qwen3-vllm-prompt-context-words 0 \
  --allow-non-h100
```

Generate the combined Metal/NVIDIA figure:

```bash
python benchmarks/qwen3_streaming_rtf/generate_combined_rtf_svg.py
```
