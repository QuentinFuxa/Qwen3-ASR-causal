"""Command line interface for qwen3-asr-causal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_audio(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is {sr} Hz; install librosa or provide {sample_rate} Hz audio"
            ) from exc
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def _build_asr(args: argparse.Namespace):
    common = {
        "lan": args.language,
        "model_size": args.model,
    }
    if args.backend == "hf":
        from .asr import Qwen3StreamingASR

        return Qwen3StreamingASR(
            **common,
            qwen3_streaming_audio_backend="causal",
            qwen3_streaming_tower_checkpoint=args.tower,
            qwen3_streaming_chunk_sec=args.chunk_sec,
            qwen3_streaming_left_context_sec=args.left_context_sec,
            qwen3_streaming_block_frames=args.block_frames,
        )

    from .vllm import Qwen3VLLMASR

    return Qwen3VLLMASR(
        **common,
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend=args.decoder_backend,
        qwen3_vllm_tower_checkpoint=args.tower,
        qwen3_vllm_left_context_sec=args.left_context_sec,
        qwen3_vllm_block_frames=args.block_frames,
        qwen3_vllm_live_idle_timeout_ms=args.live_idle_timeout_ms,
    )


def _processor_for(args: argparse.Namespace, asr):
    if args.backend == "hf":
        from .online import Qwen3StreamingOnlineProcessor

        return Qwen3StreamingOnlineProcessor(asr)
    from .vllm import Qwen3VLLMCausalOnlineProcessor

    return Qwen3VLLMCausalOnlineProcessor(asr)


def transcribe(args: argparse.Namespace) -> int:
    audio = _load_audio(Path(args.audio))
    asr = _build_asr(args)
    processor = _processor_for(args, asr)
    sample_rate = getattr(processor, "SAMPLING_RATE", 16_000)
    chunk_samples = max(1, int(args.feed_chunk_sec * sample_rate))
    tokens = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        end_time = min(len(audio), start + len(chunk)) / sample_rate
        processor.insert_audio_chunk(chunk, end_time)
        new_tokens, _ = processor.process_iter(is_last=False)
        tokens.extend(new_tokens)
    final_tokens, _ = processor.finish()
    tokens.extend(final_tokens)
    text = " ".join((token.text or "").strip() for token in tokens if (token.text or "").strip())
    if args.json:
        print(
            json.dumps(
                {
                    "text": text,
                    "tokens": [token.__dict__ for token in tokens],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen3-asr-causal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    transcribe_parser = subparsers.add_parser("transcribe", help="transcribe one audio file")
    transcribe_parser.add_argument("audio")
    transcribe_parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    transcribe_parser.add_argument("--language", default="en")
    transcribe_parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    transcribe_parser.add_argument("--tower", default="qfuxa/qwen3-asr-0.6b-streaming")
    transcribe_parser.add_argument(
        "--decoder-backend",
        choices=("vllm-live", "vllm-text", "append-kv", "rolling", "vllm"),
        default="vllm-live",
    )
    transcribe_parser.add_argument("--chunk-sec", type=float, default=2.0)
    transcribe_parser.add_argument("--feed-chunk-sec", type=float, default=0.5)
    transcribe_parser.add_argument("--left-context-sec", type=float, default=15.0)
    transcribe_parser.add_argument("--block-frames", type=int, default=192)
    transcribe_parser.add_argument("--live-idle-timeout-ms", type=float, default=20.0)
    transcribe_parser.add_argument("--json", action="store_true")
    transcribe_parser.set_defaults(func=transcribe)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
