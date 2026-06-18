#!/usr/bin/env python3
"""Two WERs for the causal streaming backend, to separate transcript quality
from live streaming behaviour.

  WER-A "final transcript": finalize('latest'). The causal encoder never sees
  future audio, but the scored transcript lets a word be revised by later
  audio WITHIN its ~12-16 s segment (segments are frozen across rollovers).
  Comparable to an offline transcript; this is the headline 18.1.

  WER-B "real streaming": replay the per-update hypotheses as an append-only
  live display. Each non-final snapshot hides the last 250 ms of estimated
  prediction text; the end-of-stream flush uses the final hypothesis, but still
  cannot rewrite words already displayed.

The gap between A and B is exactly how much "rewriting the past" the final
score benefits from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--audio-backend", choices=("windowed", "causal"), default="causal")
    ap.add_argument("--tower-checkpoint", default="")
    ap.add_argument("--model-size", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--language", default="en")
    ap.add_argument(
        "--chunk-frames",
        type=int,
        default=0,
        help="Mel frames per streaming update. Default: 192 causal, 200 windowed.",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--real-streaming-tail-ms", type=float, default=250.0)
    ap.add_argument("--experiments-dir", type=Path,
                    default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    if args.audio_backend == "causal" and not args.tower_checkpoint:
        raise SystemExit("--tower-checkpoint is required with --audio-backend causal")
    chunk_frames = args.chunk_frames or (192 if args.audio_backend == "causal" else 200)

    sys.path.insert(0, str(args.experiments_dir))
    import soundfile as sf
    from qwen3_streaming.metrics import simulate_real_streaming_text, word_error_rate

    from qwen3_asr_causal.asr import Qwen3StreamingASR

    asr = Qwen3StreamingASR(
        lan=args.language, model_size=args.model_size,
        qwen3_streaming_audio_backend=args.audio_backend,
        qwen3_streaming_tower_checkpoint=args.tower_checkpoint,
        qwen3_streaming_dtype=args.dtype,
        qwen3_streaming_device=args.device,
    )
    rows = [json.loads(l) for l in args.manifest_jsonl.read_text().splitlines() if l.strip()]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a_list, b_list, latencies, durations = [], [], [], []
    timestamp_sources: dict[str, int] = {}

    def sync_device() -> None:
        if getattr(asr.device, "type", None) == "cuda":
            import torch

            torch.cuda.synchronize(asr.device)

    with args.output_jsonl.open("w") as out:
        for i, row in enumerate(rows):
            path = row.get("audio") or row.get("wav")
            sync_device()
            start = time.perf_counter()
            audio, sr = sf.read(path, dtype="float32")
            audio_sec = float(audio.shape[0]) / float(sr)
            feats = asr.feature_extractor(
                audio, sampling_rate=sr, padding=True, truncation=False,
                return_attention_mask=True, return_tensors="pt",
            )["input_features"][0].T.to(asr.device)

            streamer = asr.build_streamer(args.language)
            for s in range(0, feats.shape[0], chunk_frames):
                streamer.append_mel_chunk(feats[s:s + chunk_frames, :].unsqueeze(0))
            streamer.flush_pending_audio()

            text_latest = streamer.finalize(finalize_mode="latest").final_text
            sync_device()
            latency_sec = time.perf_counter() - start
            rtf = latency_sec / audio_sec if audio_sec > 0.0 else None
            real_streaming = simulate_real_streaming_text(
                streamer.events,
                final_text=text_latest,
                tail_sec=float(args.real_streaming_tail_ms) / 1000.0,
                word_alignments=row.get("word_alignments"),
            )
            streaming_text = real_streaming["text"]

            ref = row.get("teacher_text") or row.get("text") or row.get("reference") or ""
            wer_a = word_error_rate(ref, text_latest) if ref else None
            wer_b = word_error_rate(ref, streaming_text) if ref else None
            if wer_a is not None: a_list.append(wer_a)
            if wer_b is not None: b_list.append(wer_b)
            if rtf is not None:
                latencies.append(latency_sec)
                durations.append(audio_sec)
            timestamp_source = str(real_streaming.get("timestamp_source") or "unknown")
            timestamp_sources[timestamp_source] = timestamp_sources.get(timestamp_source, 0) + 1
            out.write(json.dumps({
                "audio": str(path), "reference": ref,
                "audio_backend": args.audio_backend,
                "chunk_frames": chunk_frames,
                "audio_sec": audio_sec,
                "latency_sec": latency_sec,
                "realtime_factor": rtf,
                "final_text": text_latest,              # WER-A
                "streaming_text": streaming_text,        # WER-B
                "real_streaming": {
                    key: value
                    for key, value in real_streaming.items()
                    if key != "events"
                },
                "wer_final": wer_a, "wer_streaming": wer_b,
            }) + "\n")
            out.flush()
            print(f"[{i+1}/{len(rows)}] A={None if wer_a is None else round(wer_a,4)} "
                  f"B={None if wer_b is None else round(wer_b,4)} "
                  f"rtf={None if rtf is None else round(rtf,4)} {Path(path).name}", flush=True)

    summary = {
        "count": len(rows),
        "audio_backend": args.audio_backend,
        "model_size": args.model_size,
        "chunk_frames": chunk_frames,
        "dtype": args.dtype,
        "device": args.device,
        "wer_final_latest_mean": sum(a_list)/len(a_list) if a_list else None,
        "wer_streaming_norevise_mean": sum(b_list)/len(b_list) if b_list else None,
        "real_streaming_tail_ms": args.real_streaming_tail_ms,
        "real_streaming_timestamp_sources": timestamp_sources,
        "latency_mean_sec": sum(latencies)/len(latencies) if latencies else None,
        "audio_duration_total_sec": sum(durations),
        "realtime_factor_mean": (
            sum(lat / dur for lat, dur in zip(latencies, durations, strict=True))
            / len(latencies)
            if latencies
            else None
        ),
        "realtime_factor_total": (
            sum(latencies) / sum(durations) if durations and sum(durations) > 0.0 else None
        ),
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
