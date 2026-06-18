#!/usr/bin/env python3
"""D2a: decoder LoRA co-adaptation over the frozen distilled causal tower.

The tower comes from D1 (``--resume-tower``, frozen). LoRA wraps the Qwen
text decoder; teacher-forced CE on teacher transcripts is the only loss.
Because ``lora_b`` is zero-initialized, step 0 reproduces the D1 model
exactly — the step-0 gate doubles as a resume sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from qwen3_streaming.decoder_ce import (
    block_aligned_prefix_pair_targets,
    block_aligned_streaming_prefix_target,
    build_ce_inputs,
    ce_forward,
    consistency_kl_forward,
    sample_streaming_prefix_target,
)
from qwen3_streaming.gate import gate_eval
from qwen3_streaming.lora import (
    DECODER_LORA_TARGETS,
    add_lora_to_linear_modules,
    lora_parameters,
    lora_state_dict,
)
from qwen3_streaming.native_realtime_model import (
    Qwen3ASRRealtimeQwenAudioCausalModel,
    _register_qwen3_asr_transformers,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig
from qwen3_streaming.tower_distill import block_bidirectional_forward


def resolve_tower_checkpoint(reference: str | Path) -> Path:
    path = Path(reference).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for pattern in ("*.safetensors", "*.pt"):
            matches = sorted(path.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"no .safetensors or .pt checkpoint in {path}")
    ref = str(reference)
    if "/" not in ref:
        raise FileNotFoundError(f"tower checkpoint not found: {reference}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(ref, "model.safetensors", repo_type="model"))


def load_tower_checkpoint(audio_tower: torch.nn.Module, reference: str | Path) -> dict:
    checkpoint_path = resolve_tower_checkpoint(reference)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
        metadata: dict = {"checkpoint_path": str(checkpoint_path)}
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "tower_state_dict" in payload:
            state_dict = payload["tower_state_dict"]
            metadata = {
                key: payload.get(key)
                for key in ("step", "gate_wer", "gate_wers", "model_id")
                if key in payload
            }
        else:
            state_dict = payload
            metadata = {}
        metadata["checkpoint_path"] = str(checkpoint_path)
    audio_tower.load_state_dict(state_dict, strict=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-tower", required=True)
    parser.add_argument(
        "--train-manifests",
        required=True,
        help="Comma-separated JSONL manifests with audio + teacher_text (+language).",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-end-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--block-frames", type=int, default=96)
    parser.add_argument("--left-context-sec", type=float, default=15.0)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--max-target-tokens", type=int, default=384)
    parser.add_argument(
        "--prefix-ce-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of replacing full-utterance CE with a streaming "
            "prefix CE example. Requires word_alignments in the manifest."
        ),
    )
    parser.add_argument("--prefix-min-sec", type=float, default=1.0)
    parser.add_argument(
        "--prefix-right-context-sec",
        type=float,
        default=0.25,
        help="Live-safe label holdback: only words ending before prefix_end - this value are trained.",
    )
    parser.add_argument("--prefix-min-target-words", type=int, default=1)
    parser.add_argument(
        "--prefix-eos-weight",
        type=float,
        default=1.0,
        help=(
            "Extra CE weight on EOS for prefix examples. Values >1 encourage "
            "the decoder to stop at the stable live prefix instead of emitting "
            "future words that may be revised."
        ),
    )
    parser.add_argument(
        "--prefix-pair-consistency-prob",
        type=float,
        default=0.0,
        help="Probability of replacing a prefix CE sample with an adjacent-prefix pair.",
    )
    parser.add_argument(
        "--prefix-pair-gap-sec",
        type=float,
        default=0.96,
        help="Audio gap between the old and new prefix in a paired-prefix sample.",
    )
    parser.add_argument(
        "--prefix-pair-consistency-weight",
        type=float,
        default=0.0,
        help="Weight for KL(old-prefix logits || new-prefix logits) on common stable text.",
    )
    parser.add_argument(
        "--prefix-pair-prev-ce-weight",
        type=float,
        default=0.0,
        help="Optional extra CE weight on the earlier prefix in paired-prefix samples.",
    )
    parser.add_argument("--prefix-pair-min-common-words", type=int, default=1)
    parser.add_argument("--prefix-consistency-temperature", type=float, default=1.0)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-limit", type=int, default=10)
    parser.add_argument("--gate-every", type=int, default=500)
    parser.add_argument("--gate-chunk-ms", type=float, default=960.0)
    parser.add_argument(
        "--gate-score",
        choices=("latest", "real_streaming"),
        default="latest",
        help=(
            "Checkpoint-selection WER contract. 'real_streaming' scores the "
            "append-only prefix with the live tail cut and final flush."
        ),
    )
    parser.add_argument("--gate-real-streaming-tail-ms", type=float, default=250.0)
    parser.add_argument(
        "--save-gate-checkpoints",
        action="store_true",
        help="Also save a LoRA checkpoint at every gate step for offline rescoring.",
    )
    parser.add_argument("--language", required=True, help="Gate prompt language, e.g. English")
    parser.add_argument("--default-train-language", default="English")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(manifests: str, *, max_audio_sec: float) -> list[dict]:
    rows: list[dict] = []
    for path in manifests.split(","):
        path = path.strip()
        if not path:
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("teacher_text") or row.get("text")
            if not text or not row.get("audio"):
                continue
            duration = float(row.get("duration_sec") or 0.0)
            if duration and duration > max_audio_sec:
                continue
            rows.append(
                {
                    "audio": row["audio"],
                    "text": str(text),
                    "language": str(row.get("language") or ""),
                    "duration_sec": duration,
                    "word_alignments": row.get("word_alignments") or [],
                }
            )
    if not rows:
        raise SystemExit("no usable rows in the train manifests")
    return rows


def main() -> None:
    args = parse_args()
    if args.prefix_eos_weight <= 0.0:
        raise ValueError("--prefix-eos-weight must be > 0")
    if args.prefix_pair_consistency_prob < 0.0 or args.prefix_pair_consistency_prob > 1.0:
        raise ValueError("--prefix-pair-consistency-prob must be in [0, 1]")
    if args.prefix_pair_gap_sec <= 0.0:
        raise ValueError("--prefix-pair-gap-sec must be > 0")
    if args.prefix_pair_consistency_weight < 0.0:
        raise ValueError("--prefix-pair-consistency-weight must be >= 0")
    if args.prefix_pair_prev_ce_weight < 0.0:
        raise ValueError("--prefix-pair-prev-ce-weight must be >= 0")
    if args.prefix_consistency_temperature <= 0.0:
        raise ValueError("--prefix-consistency-temperature must be > 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)

    _register_qwen3_asr_transformers()
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    import soundfile as sf

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    hf_config = AutoConfig.from_pretrained(args.model_id)
    config = RealtimeAudioConfig(
        d_model=int(hf_config.thinker_config.text_config.hidden_size),
        qwen_audio_left_context_sec=args.left_context_sec,
        qwen_audio_block_bidirectional=True,
    )
    model = Qwen3ASRRealtimeQwenAudioCausalModel.from_qwen_pretrained(
        args.model_id,
        config=config,
        bos_token_id=(
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0
        ),
        wait_token_id=None,
        dtype=torch.float32,
        device_map="cpu",
    ).to(device)

    resume_metadata = load_tower_checkpoint(model.audio_encoder.audio_tower, args.resume_tower)
    print(json.dumps({
        "resumed_tower": str(args.resume_tower),
        "resume_metadata": resume_metadata,
    }))

    # Freeze everything, then add LoRA to the decoder.
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    wrapped = add_lora_to_linear_modules(
        model.text_model,
        target_names=DECODER_LORA_TARGETS,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.to(device)
    trainable = lora_parameters(model.text_model)
    for param in trainable:
        param.requires_grad_(True)
    n_trainable = sum(p.numel() for p in trainable)
    print(json.dumps({"lora_modules": len(wrapped), "trainable_params": n_trainable}))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        floor = args.lr * args.lr_end_ratio
        return floor + (args.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    audio_placeholder = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
    rows = load_rows(args.train_manifests, max_audio_sec=args.max_audio_sec)
    print(json.dumps({"train_rows": len(rows)}))

    def run_gate() -> float:
        return gate_eval(
            model,
            tokenizer,
            processor,
            manifest=args.gate_manifest,
            limit=args.gate_limit,
            chunk_ms=args.gate_chunk_ms,
            language=args.language,
            device=device,
            score_mode=args.gate_score,
            real_streaming_tail_ms=args.gate_real_streaming_tail_ms,
        )

    gate0 = run_gate()
    best_wer = gate0
    history = [{"step": 0, "gate_wer": gate0}]
    print(f"step 0 gate_wer={gate0:.4f} (LoRA no-op sanity)", flush=True)

    def checkpoint_payload(*, step: int, wer: float) -> dict:
        return {
            "lora_state_dict": lora_state_dict(model.text_model),
            "step": step,
            "gate_wer": wer,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": list(DECODER_LORA_TARGETS),
            "tower_checkpoint": str(args.resume_tower),
            "model_id": args.model_id,
            "gate_score": args.gate_score,
            "gate_real_streaming_tail_ms": args.gate_real_streaming_tail_ms,
            "prefix_ce_prob": args.prefix_ce_prob,
            "prefix_right_context_sec": args.prefix_right_context_sec,
            "prefix_eos_weight": args.prefix_eos_weight,
            "prefix_pair_consistency_prob": args.prefix_pair_consistency_prob,
            "prefix_pair_gap_sec": args.prefix_pair_gap_sec,
            "prefix_pair_consistency_weight": args.prefix_pair_consistency_weight,
            "prefix_pair_prev_ce_weight": args.prefix_pair_prev_ce_weight,
            "prefix_consistency_temperature": args.prefix_consistency_temperature,
        }

    def sample_forward(row: dict) -> tuple[torch.Tensor, dict]:
        use_prefix = (
            args.prefix_ce_prob > 0.0
            and random.random() < args.prefix_ce_prob
        )
        prefix_target = None
        if use_prefix:
            prefix_target = sample_streaming_prefix_target(
                row,
                rng=random,
                min_prefix_sec=args.prefix_min_sec,
                right_context_sec=args.prefix_right_context_sec,
                min_target_words=args.prefix_min_target_words,
            )
        audio, sr = sf.read(row["audio"], dtype="float32")
        features = processor.feature_extractor(
            audio,
            sampling_rate=sr,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )["input_features"][0].T
        target_text = row["text"]
        pair_target = None
        if prefix_target is not None:
            prefix_end_sec, _ = prefix_target
            if (
                args.prefix_pair_consistency_prob > 0.0
                and random.random() < args.prefix_pair_consistency_prob
            ):
                pair_target = block_aligned_prefix_pair_targets(
                    row,
                    prefix_end_sec=prefix_end_sec,
                    available_frames=int(features.shape[0]),
                    block_frames=args.block_frames,
                    pair_gap_sec=args.prefix_pair_gap_sec,
                    right_context_sec=args.prefix_right_context_sec,
                    min_target_words=args.prefix_min_target_words,
                    min_common_words=args.prefix_pair_min_common_words,
                )
            if pair_target is not None:
                previous_frames, previous_text, frames, target_text = pair_target
                aligned_target = (frames, target_text)
            else:
                aligned_target = block_aligned_streaming_prefix_target(
                    row,
                    prefix_end_sec=prefix_end_sec,
                    available_frames=int(features.shape[0]),
                    block_frames=args.block_frames,
                    right_context_sec=args.prefix_right_context_sec,
                    min_target_words=args.prefix_min_target_words,
                )
            if aligned_target is None:
                raise ValueError("block-aligned prefix has too few stable words")
            frames, target_text = aligned_target
        else:
            frames = min(features.shape[0], int(args.max_audio_sec * 100))
            frames -= frames % args.block_frames
            if frames < args.block_frames:
                frames = (features.shape[0] // 8) * 8  # short clip: single partial block

        def project_frame_hidden(frame_count: int) -> torch.Tensor:
            mels = features[:frame_count].unsqueeze(0).to(device)
            with torch.no_grad():
                tower_out = block_bidirectional_forward(
                    model.audio_encoder.audio_tower,
                    mels,
                    block_frames=min(args.block_frames, int(mels.shape[1])),
                    left_context_steps=model.audio_encoder.left_context_steps,
                )
                return model.adapter._project(tower_out)

        frame_hidden = project_frame_hidden(frames)
        prompt_ids, target_ids, _ = build_ce_inputs(
            tokenizer,
            audio_steps=int(frame_hidden.shape[1]),
            language=row["language"] or args.default_train_language,
            target_text=target_text,
            audio_placeholder_token_id=audio_placeholder,
            max_target_tokens=args.max_target_tokens,
        )
        loss, stats = ce_forward(
            model,
            frame_hidden,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
            audio_placeholder_token_id=audio_placeholder,
            eos_token_id=tokenizer.eos_token_id,
            eos_weight=(
                args.prefix_eos_weight if prefix_target is not None else 1.0
            ),
        )
        stats["prefix_ce"] = 1.0 if prefix_target is not None else 0.0
        stats["prefix_pair"] = 0.0
        stats["consistency_loss"] = 0.0
        stats["prev_ce_loss"] = 0.0
        if pair_target is not None and (
            args.prefix_pair_consistency_weight > 0.0
            or args.prefix_pair_prev_ce_weight > 0.0
        ):
            previous_frame_hidden = project_frame_hidden(previous_frames)
            previous_prompt_ids, previous_target_ids, _ = build_ce_inputs(
                tokenizer,
                audio_steps=int(previous_frame_hidden.shape[1]),
                language=row["language"] or args.default_train_language,
                target_text=previous_text,
                audio_placeholder_token_id=audio_placeholder,
                max_target_tokens=args.max_target_tokens,
            )
            if args.prefix_pair_prev_ce_weight > 0.0:
                previous_loss, previous_stats = ce_forward(
                    model,
                    previous_frame_hidden,
                    prompt_ids=previous_prompt_ids,
                    target_ids=previous_target_ids,
                    audio_placeholder_token_id=audio_placeholder,
                    eos_token_id=tokenizer.eos_token_id,
                    eos_weight=args.prefix_eos_weight,
                )
                loss = loss + args.prefix_pair_prev_ce_weight * previous_loss
                stats["prev_ce_loss"] = float(previous_loss.detach())
                stats["prev_ce_tokens"] = previous_stats["target_tokens"]
            if args.prefix_pair_consistency_weight > 0.0:
                old_common_prompt, common_target_ids, _ = build_ce_inputs(
                    tokenizer,
                    audio_steps=int(previous_frame_hidden.shape[1]),
                    language=row["language"] or args.default_train_language,
                    target_text=previous_text,
                    audio_placeholder_token_id=audio_placeholder,
                    max_target_tokens=args.max_target_tokens,
                    add_eos=False,
                )
                new_common_prompt, _, _ = build_ce_inputs(
                    tokenizer,
                    audio_steps=int(frame_hidden.shape[1]),
                    language=row["language"] or args.default_train_language,
                    target_text=previous_text,
                    audio_placeholder_token_id=audio_placeholder,
                    max_target_tokens=args.max_target_tokens,
                    add_eos=False,
                )
                consistency_loss, consistency_stats = consistency_kl_forward(
                    model,
                    previous_frame_hidden,
                    frame_hidden,
                    old_prompt_ids=old_common_prompt,
                    new_prompt_ids=new_common_prompt,
                    common_target_ids=common_target_ids,
                    audio_placeholder_token_id=audio_placeholder,
                    temperature=args.prefix_consistency_temperature,
                )
                loss = loss + args.prefix_pair_consistency_weight * consistency_loss
                stats["consistency_loss"] = float(consistency_loss.detach())
                stats.update(consistency_stats)
            stats["prefix_pair"] = 1.0
        return loss, stats

    order = list(range(len(rows)))
    random.shuffle(order)
    cursor = 0
    started = time.time()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        losses, accs, prefix_rates, pair_rates = [], [], [], []
        consistency_losses = []
        for _ in range(args.grad_accum):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            try:
                loss, stats = sample_forward(row)
            except Exception as exc:  # noqa: BLE001 - skip unreadable rows
                print(f"skip row ({exc})", flush=True)
                continue
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
            accs.append(stats["token_accuracy"])
            prefix_rates.append(float(stats.get("prefix_ce", 0.0)))
            pair_rates.append(float(stats.get("prefix_pair", 0.0)))
            consistency_losses.append(float(stats.get("consistency_loss", 0.0)))
        if not losses:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            speed = step / (time.time() - started)
            print(
                f"step {step} loss={sum(losses)/len(losses):.4f} "
                f"acc={sum(accs)/len(accs):.4f} "
                f"prefix={sum(prefix_rates)/len(prefix_rates):.2f} "
                f"pair={sum(pair_rates)/len(pair_rates):.2f} "
                f"cons={sum(consistency_losses)/len(consistency_losses):.4f} "
                f"steps/s={speed:.2f}",
                flush=True,
            )

        if step % args.gate_every == 0 or step == args.steps:
            wer = run_gate()
            history.append({"step": step, "gate_wer": wer})
            print(f"step {step} gate_wer={wer:.4f} (best {best_wer:.4f})", flush=True)
            payload = None
            if args.save_gate_checkpoints:
                payload = checkpoint_payload(step=step, wer=wer)
                torch.save(payload, args.output_dir / f"lora_step_{step}.pt")
            if wer < best_wer:
                best_wer = wer
                if payload is None:
                    payload = checkpoint_payload(step=step, wer=wer)
                torch.save(payload, args.output_dir / "lora_best.pt")
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "gate_wer_start": gate0,
                "gate_wer_best": best_wer,
                "gate_score": args.gate_score,
                "gate_real_streaming_tail_ms": args.gate_real_streaming_tail_ms,
                "prefix_eos_weight": args.prefix_eos_weight,
                "prefix_pair_consistency_prob": args.prefix_pair_consistency_prob,
                "prefix_pair_gap_sec": args.prefix_pair_gap_sec,
                "prefix_pair_consistency_weight": args.prefix_pair_consistency_weight,
                "prefix_pair_prev_ce_weight": args.prefix_pair_prev_ce_weight,
                "prefix_consistency_temperature": args.prefix_consistency_temperature,
                "trainable_params": n_trainable,
                "history": history,
            },
            indent=2,
        )
    )
    print(json.dumps({"gate_wer_start": gate0, "gate_wer_best": best_wer}))


if __name__ == "__main__":
    main()
