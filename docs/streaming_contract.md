# Streaming contract

The live display is append-only.

At each non-final update the runtime hides the hypothesis portion aligned to
the last 250 ms of audio, then publishes text only when the new hypothesis
extends the already published prefix. The end-of-stream update flushes the
held-back tail, but it still cannot revise words that were previously
published.

The HF backend estimates word timing from the newly committed span. The vLLM
backend uses Qwen3 ForcedAligner word timestamps when available, so the 250 ms
tail cut follows word-level alignment instead of a uniform text approximation.

This contract is the one used for the real-streaming WER numbers in the README.
