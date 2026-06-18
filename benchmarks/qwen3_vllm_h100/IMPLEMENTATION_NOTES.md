# CUDA/vLLM Causal Implementation Notes

The CUDA/vLLM causal path now exists for `qwen3-vllm` and was benchmarked on an
NVIDIA A100. It is intentionally still labeled experimental.

## What Counts

The plotted causal bar is valid only when the measured stack satisfies all of
this:

- runs on an NVIDIA GPU;
- uses vLLM in the `qwen3-vllm` backend for ForcedAligner timestamps;
- uses the trained causal Qwen3 audio tower;
- uses the `vllm-live` decoder path for the plotted fast CUDA causal result,
  or `vllm-text` / `append-kv` / `rolling` / legacy vLLM `audio_embeds` paths
  for development comparisons;
- feeds audio incrementally without future context;
- preserves committed prefix text and never rewrites past output;
- hides the last 250 ms of non-final predictions and flushes only at stream end.

The 2026-06-14 A100 run satisfies these constraints. It is not the older HF
Transformers/CUDA causal production number from `benchmarks/qwen3_streaming_rtf`.

## Current CUDA/vLLM Path

`qwen3_asr_causal.vllm` now supports:

```bash
wlk --backend qwen3-vllm --language en \
  --qwen3-vllm-audio-backend causal \
  --qwen3-vllm-causal-decoder-backend vllm-live \
  --qwen3-vllm-tower-checkpoint qfuxa/qwen3-asr-0.6b-streaming
```

Fast `vllm-live` implementation shape:

1. Reuse the causal audio encoder from `qwen3_asr_causal.causal`.
2. Encode new audio blocks incrementally with causal audio state.
3. Export the Qwen3-ASR text decoder as a standalone vLLM CausalLM.
4. Keep one vLLM streaming request alive and update its prompt embeddings with
   request-local `prefix_keep_len` metadata.
5. With `WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1`, run that live decoder through
   vLLM's async multiprocessing/CUDA graph path.
6. Bound stream length with causal segment rollover: once enough text is
   committed, create a fresh session, replay only the uncommitted audio overlap,
   and preserve the published prefix outside the decoder.
7. Keep the ForcedAligner timestamp path in vLLM for the 250 ms no-rewrite
   streaming policy.

This is a patched vLLM V1 streaming path rather than a public upstream
prompt-embeds append API, but it does keep the request-local audio KV prefix and
avoids the full prompt+audio re-prefill failure mode.

On A100/vLLM 0.23, the fast `vllm-live` JFK x2 smoke measured 0.099 RTF. The
final decode prompt had `prompt_tokens=160`, `vllm_live_prefix_keep_tokens=129`,
and `vllm_live_suffix_prefill_tokens=31`; the maximum suffix prefill over the
stream was 66 tokens. Mean live decoder wall time was 34.0 ms.

Runtime cache counters are recorded from `RequestOutput.num_cached_tokens`.
For the conservative `vllm-text` fallback, the 22 s JFK smoke ended with `prompt_tokens=160`,
`vllm_cached_tokens=128`, and `vllm_effective_prefill_tokens=32`; the maximum
effective prefill over the stream was 66 tokens while measuring 0.113 RTF. The
repeated 110 s smoke kept the same maximum (`prompt_tokens_max=186`,
`vllm_effective_prefill_tokens_max=66`) while measuring 0.104 RTF. This proves
the current path is not doing a full prompt+audio prefill per chunk. The
remaining gap to the requested ideal is the cache-block/tail recompute, which
requires either an actual request-local append API or a KV connector/custom
backend that can expose already-computed partial blocks as external KV.

Changing the vLLM cache block size does not fix that gap on CUDA today:
`block_size=8` fails during engine init in vLLM 0.23 because the available CUDA
attention backends do not support it, while `block_size=16` reproduces the
default counters (`vllm_effective_prefill_tokens_last=32`,
`vllm_effective_prefill_tokens_max=66`).

Experimental `--qwen3-vllm-causal-decoder-backend append-kv` implementation
shape (`rolling` remains as a compatibility alias):

1. Reuse the causal audio encoder from `qwen3_asr_causal.causal`.
2. Maintain append-only causal audio state per stream.
3. Generate ASR text with `generate_full_hypothesis_rolling`, which keeps
   decoder KV over `[prompt head + audio]` and forwards only new audio, the
   prompt tail, and the previous hypothesis draft.
4. Keep the ForcedAligner timestamp path in vLLM for the 250 ms no-rewrite
   streaming policy.
5. Trim only the aligner audio/text suffix after committed prefixes, and roll
   the ASR session when the causal segment exceeds the configured bound.

Runtime stats for this path expose `audio_delta_steps`, `reused_audio_steps`,
and `prefill_positions`. For a healthy append-KV stream after the first call,
`reused_audio_steps` should grow with the stable audio prefix while
`audio_delta_steps` tracks only the newly appended audio and
`prefill_positions` covers the delta plus tail/draft rather than the full
`[prompt + audio]` prefix.

On the 2026-06-14 A100/JFK x2 smoke, `append-kv` measured 0.150 RTF. Its final
decode had `audio_steps=136`, `reused_audio_steps=120`,
`audio_delta_steps=16`, and `prefill_positions=47`; the maximum effective
append prefill was 66 positions. This proves the algorithmic append/KV contract
on real CUDA, but it is still slower than `vllm-text` because the text decoder
runtime is HF Transformers rather than a vLLM model runner.

The legacy `--qwen3-vllm-causal-decoder-backend vllm` implementation shape:

1. Split the accumulated causal audio embeddings into stable multimodal
   `audio_embeds` blocks.
2. Compose a Qwen ASR token prompt with one audio placeholder per block.
3. Generate text through vLLM with `enable_mm_embeds=True` and
   `enable_prefix_caching=True`.
4. Keep the ForcedAligner timestamp path for the 250 ms no-rewrite streaming
   policy.

vLLM 0.23 needs a runtime shim for the legacy `vllm` decoder backend:

- Qwen3-ASR's vLLM model accepts raw/processed audio features but not
  precomputed audio embeddings.
- The shim adds the same `audio_embeds` contract used by Qwen2-audio.
- The shim bypasses the HF audio processor for embedding-only inputs and
  splits concatenated embeddings back into one tensor per multimodal item.
- V1 multiprocessing is disabled so the monkey patches are visible to the
  worker.
- `cudagraph_mode=NONE` is currently required for the ASR engine because vLLM
  0.23's CUDA graph replay path crashes on this embedding input shape.

## Why The Old Causal Bar Was Slower

The old measured A100 CUDA causal rolling RTF was 0.160 versus 0.146 for
standard `qwen3-vllm` on the same 22 s clip. The old pure-vLLM `audio_embeds`
causal path measured 0.412. Profiling the rolling path on the same clip gave
roughly 2.42 s in HF `decode_text` and 0.75 s in vLLM `align_words` for a
3.62 s streaming loop, so the remaining bottleneck was the HF rolling decoder,
not the ForcedAligner.

The `vllm-text` path moves the text decoder back into vLLM, keeps prefix
caching enabled, and bounds each causal audio segment. It measured 0.113 RTF
on the same 22 s smoke and 0.104 RTF on a 110 s repeated JFK smoke. The causal
checkpoint still needs a separate quality gate on real long-form WER; these
numbers are speed smokes under the live no-rewrite protocol.

A verified-draft experiment (`WLK_QWEN3_VLLM_TEXT_DRAFT=1`) appends the
previous hypothesis to the vLLM prompt and checks it with `prompt_logprobs`.
This keeps correctness by falling back to the default request when the draft is
not fully greedy-accepted. On JFK x2 it was lossless but slower (0.174 RTF):
only 10 of 22 calls fully accepted the draft, rejected chunks paid a second
request, and the prompt-logprob pass raised mean stream decode time from about
52 ms to 93 ms. It is therefore opt-in and not the default path.

An even closer live-append experiment (`WLK_QWEN3_VLLM_LIVE_DRAFT=1` with
`--qwen3-vllm-causal-decoder-backend vllm-live`) keeps one vLLM streaming
request alive, crops it back to the `[prompt head + audio]` KV prefix, then
prefills only `new_audio + prompt_tail + previous_hypothesis_draft` before
generating the continuation. It also verifies the draft with `prompt_logprobs`
and falls back by recropping to the current full audio prefix when verification
fails. On A100/JFK x2 this reached `vllm_effective_prefill_tokens_last=9`, so
the request-local append is doing the intended KV crop, but every draft was
rejected (`draft_all_accepted_calls=0`) and the eager live runner measured
0.333 RTF. This confirms the scheduler-level append machinery works, while the
remaining performant path still needs a vLLM custom runner/KV connector rather
than the public streaming request shim.

The non-draft `vllm-live` path now avoids cumulative output traffic by setting
`SamplingParams(output_kind=DELTA, detokenize=False)` when vLLM exposes
`RequestOutputKind.DELTA`, then reconstructing the cumulative token list locally.
The A100/JFK x2 smoke recorded `generation_vllm_live_delta_output_calls=22` and
improved the live probe from 0.325 RTF to 0.280 RTF. Adding stop token ids was
neutral/slower at 0.289 RTF: the request still returned on the 20 ms idle guard
for all chunks, so EOS was not the bottleneck.

Letting `vllm-live` use vLLM's default V1 multiprocessing/CUDA graph path now
works with `WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1`. The missing piece was an
async-scheduler patch scoped to WLK live prompt-embeds requests: when a streaming
update replaces the prompt, stale async outputs are discarded and excess output
tokens no longer drive `num_output_placeholders` negative. A direct synthetic
A100 smoke with prompt lengths 32 -> 40 -> 48 verified live prompt-embeds
updates under `cudagraph_mode=FULL_AND_PIECEWISE`.

The full JFK x2 benchmark also needs the ForcedAligner. Running both the live
decoder and aligner as independent vLLM multiprocessing engines can hang during
the second engine initialization, so the live CUDA path now keeps the aligner
in-process by default when `WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1` (override:
`WLK_QWEN3_VLLM_ALIGNER_MULTIPROCESSING=1`). With that split, the full streaming
benchmark completed at 0.099 RTF on A100, with
`generation_vllm_live_decode_wall_ms_mean=34.0` and
`generation_vllm_live_delta_output_calls=22`.

## Validation Gates

```bash
pytest -q tests/test_qwen3_vllm_asr.py
python -m py_compile src/qwen3_asr_causal/vllm.py \
  benchmarks/qwen3_vllm_h100/run_vllm_h100.py
python benchmarks/qwen3_streaming_rtf/generate_combined_rtf_svg.py
```
