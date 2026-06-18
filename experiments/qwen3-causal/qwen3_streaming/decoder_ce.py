"""Teacher-forced CE objective for decoder co-adaptation (D2).

The student audio path produces ``frame_hidden`` (cached audio embeddings,
adapter-projected). The decoder is teacher-forced on the Qwen ASR prompt with
audio placeholders scattered to those embeddings, followed by the reference
transcript; cross-entropy applies only to transcript (+EOS) positions.
"""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F

from .cached_full_hypothesis import (
    expand_audio_prompt_placeholders,
    qwen_asr_prompt_text,
)
from .native_realtime_model import _cached_audio_prefix_embeds


def build_ce_inputs(
    tokenizer,
    *,
    audio_steps: int,
    language: str,
    target_text: str,
    audio_placeholder_token_id: int,
    context: str = "",
    max_target_tokens: int = 384,
    add_eos: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Returns (prompt_ids_with_expanded_audio, target_ids, labels).

    ``labels`` covers the full sequence (prompt + targets): ``-100`` on every
    prompt/audio position, token ids on transcript positions.
    """
    prompt_template = tokenizer.encode(
        qwen_asr_prompt_text(context=context, language=language),
        add_special_tokens=False,
    )
    prompt_ids = expand_audio_prompt_placeholders(
        prompt_template,
        audio_placeholder_token_id=audio_placeholder_token_id,
        audio_steps=int(audio_steps),
    )
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_ids = target_ids[:max_target_tokens]
    if add_eos and tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    labels = [-100] * len(prompt_ids) + [int(t) for t in target_ids]
    return prompt_ids, [int(t) for t in target_ids], labels


def stable_text_from_word_alignments(
    words: list[dict[str, object]],
    *,
    stable_until_sec: float,
) -> str:
    """Text whose aligned words have ended before the streaming frontier."""
    committed: list[str] = []
    for item in words:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            end_sec = float(item.get("end_sec", item.get("end", 0.0)))
        except (TypeError, ValueError):
            continue
        if end_sec <= stable_until_sec:
            committed.append(text)
    return " ".join(committed)


def sample_streaming_prefix_target(
    row: dict,
    *,
    rng: random.Random,
    min_prefix_sec: float,
    right_context_sec: float,
    min_target_words: int = 1,
    max_tries: int = 8,
) -> tuple[float, str] | None:
    """Sample a partial-audio CE target from word-aligned transcripts.

    Returns ``(prefix_end_sec, stable_text)`` or ``None`` when the row does not
    have enough aligned words for a prefix example. The target text is exactly
    the live-safe prefix: words ending before ``prefix_end_sec -
    right_context_sec``.
    """
    words = row.get("word_alignments") or []
    duration = float(row.get("duration_sec") or 0.0)
    if not words or duration <= min_prefix_sec:
        return None
    min_prefix = max(0.0, float(min_prefix_sec))
    right_context = max(0.0, float(right_context_sec))
    for _ in range(max(1, int(max_tries))):
        prefix_end = rng.uniform(min_prefix, duration)
        stable_until = max(0.0, prefix_end - right_context)
        text = stable_text_from_word_alignments(words, stable_until_sec=stable_until)
        if len(text.split()) >= int(min_target_words):
            return prefix_end, text
    return None


def block_aligned_streaming_prefix_target(
    row: dict,
    *,
    prefix_end_sec: float,
    available_frames: int,
    block_frames: int,
    feature_hz: float = 100.0,
    right_context_sec: float,
    min_target_words: int = 1,
) -> tuple[int, str] | None:
    """Return a target whose text matches the block-aligned audio prefix.

    Training runs the causal audio path on whole blocks. If the sampled
    ``prefix_end_sec`` is rounded down to a block boundary, the CE target must
    be rounded down too; otherwise the decoder is trained to emit words whose
    audio is not available yet.
    """
    if block_frames <= 0:
        raise ValueError("block_frames must be > 0")
    if feature_hz <= 0.0:
        raise ValueError("feature_hz must be > 0")
    frames = min(int(available_frames), int(prefix_end_sec * feature_hz))
    frames -= frames % int(block_frames)
    if frames < int(block_frames):
        return None
    actual_prefix_end_sec = frames / float(feature_hz)
    stable_until = max(0.0, actual_prefix_end_sec - max(0.0, right_context_sec))
    text = stable_text_from_word_alignments(
        row.get("word_alignments") or [],
        stable_until_sec=stable_until,
    )
    if len(text.split()) < int(min_target_words):
        return None
    return frames, text


def block_aligned_prefix_pair_targets(
    row: dict,
    *,
    prefix_end_sec: float,
    available_frames: int,
    block_frames: int,
    pair_gap_sec: float,
    feature_hz: float = 100.0,
    right_context_sec: float,
    min_target_words: int = 1,
    min_common_words: int = 1,
) -> tuple[int, str, int, str] | None:
    """Return two nearby block-aligned prefix targets from the same row.

    The earlier target must be a word prefix of the later target. That common
    text is the part whose decoder logits should remain stable when new audio
    is appended.
    """
    current = block_aligned_streaming_prefix_target(
        row,
        prefix_end_sec=prefix_end_sec,
        available_frames=available_frames,
        block_frames=block_frames,
        feature_hz=feature_hz,
        right_context_sec=right_context_sec,
        min_target_words=max(min_target_words, min_common_words),
    )
    if current is None:
        return None
    current_frames, current_text = current
    current_sec = current_frames / float(feature_hz)
    previous_sec = max(float(block_frames) / float(feature_hz), current_sec - pair_gap_sec)
    previous = block_aligned_streaming_prefix_target(
        row,
        prefix_end_sec=previous_sec,
        available_frames=available_frames,
        block_frames=block_frames,
        feature_hz=feature_hz,
        right_context_sec=right_context_sec,
        min_target_words=min_common_words,
    )
    if previous is None:
        return None
    previous_frames, previous_text = previous
    if previous_frames >= current_frames:
        return None
    previous_words = previous_text.split()
    current_words = current_text.split()
    if len(previous_words) < int(min_common_words):
        return None
    if current_words[: len(previous_words)] != previous_words:
        return None
    return previous_frames, previous_text, current_frames, current_text


def _decoder_target_logits(
    model,
    frame_hidden: torch.Tensor,
    *,
    prompt_ids: list[int],
    target_ids: list[int],
    audio_placeholder_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = frame_hidden.device
    prefix_embeds = _cached_audio_prefix_embeds(
        model,
        frame_hidden,
        prefix_token_ids=prompt_ids,
        audio_placeholder_token_id=int(audio_placeholder_token_id),
    )
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
    target_embeds = model.embed_tokens(target_tensor)
    inputs_embeds = torch.cat(
        [prefix_embeds, target_embeds.to(dtype=prefix_embeds.dtype)], dim=1
    )
    outputs = model.text_model(inputs_embeds=inputs_embeds, use_cache=False)
    logits = model.lm_head(outputs.last_hidden_state)
    prompt_len = len(prompt_ids)
    pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + len(target_ids), :]
    return pred_logits, target_tensor


def ce_forward(
    model,
    frame_hidden: torch.Tensor,
    *,
    prompt_ids: list[int],
    target_ids: list[int],
    audio_placeholder_token_id: int,
    eos_token_id: int | None = None,
    eos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """One teacher-forced decoder pass; CE on transcript positions only.

    frame_hidden: [1, steps, d_model] adapter-projected audio embeddings.
    """
    if eos_weight <= 0.0:
        raise ValueError("eos_weight must be > 0")
    pred_logits, target_tensor = _decoder_target_logits(
        model,
        frame_hidden,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        audio_placeholder_token_id=audio_placeholder_token_id,
    )
    per_token_loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]).float(),
        target_tensor.reshape(-1),
        reduction="none",
    )
    if eos_token_id is not None and eos_weight != 1.0:
        flat_target = target_tensor.reshape(-1)
        weights = torch.ones_like(per_token_loss)
        weights = torch.where(
            flat_target == int(eos_token_id),
            weights.new_full((), float(eos_weight)),
            weights,
        )
        loss = (per_token_loss * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        loss = per_token_loss.mean()
    with torch.no_grad():
        accuracy = float(
            (pred_logits.argmax(dim=-1) == target_tensor).float().mean()
        )
    return loss, {
        "token_accuracy": accuracy,
        "target_tokens": len(target_ids),
        "eos_weight": float(eos_weight),
    }


def consistency_kl_forward(
    model,
    old_frame_hidden: torch.Tensor,
    new_frame_hidden: torch.Tensor,
    *,
    old_prompt_ids: list[int],
    new_prompt_ids: list[int],
    common_target_ids: list[int],
    audio_placeholder_token_id: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """KL(old || new) over teacher-forced logits for already-stable text.

    The old prefix acts as a detached teacher for the common stable target.
    Minimizing this discourages later audio from changing the distribution over
    words that were already safe to publish.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if not common_target_ids:
        raise ValueError("common_target_ids cannot be empty")
    old_logits, _ = _decoder_target_logits(
        model,
        old_frame_hidden,
        prompt_ids=old_prompt_ids,
        target_ids=common_target_ids,
        audio_placeholder_token_id=audio_placeholder_token_id,
    )
    new_logits, _ = _decoder_target_logits(
        model,
        new_frame_hidden,
        prompt_ids=new_prompt_ids,
        target_ids=common_target_ids,
        audio_placeholder_token_id=audio_placeholder_token_id,
    )
    old_log_probs = F.log_softmax(old_logits.float() / float(temperature), dim=-1)
    new_log_probs = F.log_softmax(new_logits.float() / float(temperature), dim=-1)
    loss = F.kl_div(
        new_log_probs,
        old_log_probs.detach().exp(),
        reduction="batchmean",
    ) * (float(temperature) ** 2)
    return loss, {"consistency_tokens": len(common_target_ids)}
