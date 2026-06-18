from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import floor
from typing import Any, Iterable

from jiwer import wer

from .stable_commit import (
    join_text_units,
    normalize_text_unit_for_match,
    split_text_units,
)

_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().lower())


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        return None
    return float(wer(reference, hypothesis))


def _as_int_tokens(token_ids) -> list[int]:
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()
    return [int(token_id) for token_id in token_ids]


def token_repetition_stats(
    token_ids,
    *,
    ignored_token_ids: set[int] | tuple[int, ...] | list[int] = (),
    max_ngram: int = 3,
) -> dict[str, int | float]:
    ignored = {int(token_id) for token_id in ignored_token_ids}
    tokens = [
        token_id for token_id in _as_int_tokens(token_ids) if token_id not in ignored
    ]
    stats: dict[str, int | float] = {
        "text_token_count": len(tokens),
        "unique_text_token_count": len(set(tokens)),
    }
    for n in range(1, max_ngram + 1):
        prefix = {1: "unigram", 2: "bigram", 3: "trigram"}.get(n, f"{n}gram")
        total = max(0, len(tokens) - n + 1)
        if total == 0:
            unique = 0
        elif n == 1:
            unique = len(set(tokens))
        else:
            unique = len(
                {
                    tuple(tokens[start : start + n])
                    for start in range(0, len(tokens) - n + 1)
                }
            )
        repeated = total - unique
        stats[f"{prefix}_total"] = total
        stats[f"{prefix}_repeated"] = repeated
        stats[f"{prefix}_repetition_ratio"] = (
            float(repeated / total) if total else 0.0
        )
    return stats


def merge_token_repetition_stats(
    items: list[dict[str, int | float]],
    *,
    max_ngram: int = 3,
) -> dict[str, int | float]:
    merged: dict[str, int | float] = {
        "text_token_count": sum(int(item.get("text_token_count", 0)) for item in items),
        "unique_text_token_count_sum": sum(
            int(item.get("unique_text_token_count", 0)) for item in items
        ),
    }
    for n in range(1, max_ngram + 1):
        prefix = {1: "unigram", 2: "bigram", 3: "trigram"}.get(n, f"{n}gram")
        total = sum(int(item.get(f"{prefix}_total", 0)) for item in items)
        repeated = sum(int(item.get(f"{prefix}_repeated", 0)) for item in items)
        merged[f"{prefix}_total"] = total
        merged[f"{prefix}_repeated"] = repeated
        merged[f"{prefix}_repetition_ratio"] = (
            float(repeated / total) if total else 0.0
        )
    return merged


@dataclass(frozen=True)
class StablePrefixStats:
    reference_words: int
    hypothesis_words: int
    common_prefix_words: int
    revision_words: int

    @property
    def common_prefix_ratio(self) -> float:
        if self.reference_words == 0:
            return 1.0 if self.hypothesis_words == 0 else 0.0
        return self.common_prefix_words / self.reference_words


def stable_prefix_stats(reference: str, hypothesis: str) -> StablePrefixStats:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    common = 0
    for ref_word, hyp_word in zip(ref_words, hyp_words):
        if ref_word != hyp_word:
            break
        common += 1
    revision = max(0, len(hyp_words) - common)
    return StablePrefixStats(
        reference_words=len(ref_words),
        hypothesis_words=len(hyp_words),
        common_prefix_words=common,
        revision_words=revision,
    )


def normalized_words(text: str) -> list[str]:
    return normalize_text(text).split()


def text_revision_stats(texts: list[str]) -> dict[str, int | float]:
    revision_events = 0
    revision_words = 0
    max_revision_words = 0
    previous: list[str] = []
    nonempty_updates = 0

    for text in texts:
        words = normalized_words(text)
        if words:
            nonempty_updates += 1
        common = 0
        for previous_word, current_word in zip(previous, words):
            if previous_word != current_word:
                break
            common += 1
        revised = max(0, len(previous) - common)
        if revised:
            revision_events += 1
            revision_words += revised
            max_revision_words = max(max_revision_words, revised)
        previous = words

    update_count = max(0, len(texts) - 1)
    return {
        "updates": len(texts),
        "nonempty_updates": nonempty_updates,
        "revision_events": revision_events,
        "revision_words": revision_words,
        "max_revision_words": max_revision_words,
        "revision_event_ratio": (
            float(revision_events / update_count) if update_count else 0.0
        ),
    }


def time_truncated_text(
    text: str,
    *,
    audio_sec: float,
    tail_sec: float,
    final: bool = False,
    word_alignments: Iterable[Any] | None = None,
) -> str:
    """Drop the transcript portion aligned to the last tail_sec.

    If forced-aligner word timestamps are available, hypothesis words are
    matched to those timestamped reference slots. Otherwise this falls back to
    distributing hypothesis words uniformly over the audio prefix.
    """
    if tail_sec < 0.0:
        raise ValueError("tail_sec must be >= 0")
    units = split_text_units(text)
    if final or tail_sec == 0.0 or not units:
        return join_text_units(units)
    if audio_sec <= 0.0 or audio_sec <= tail_sec:
        return ""
    if word_alignments is not None:
        aligned_text = _time_truncated_text_from_alignments(
            units,
            audio_sec=audio_sec,
            tail_sec=tail_sec,
            word_alignments=word_alignments,
        )
        if aligned_text is not None:
            return aligned_text
    keep_ratio = max(0.0, min(1.0, (audio_sec - tail_sec) / audio_sec))
    keep_units = int(floor(len(units) * keep_ratio))
    return join_text_units(units[:keep_units])


def _time_truncated_text_from_alignments(
    units: list[str],
    *,
    audio_sec: float,
    tail_sec: float,
    word_alignments: Iterable[Any],
) -> str | None:
    entries = [
        entry
        for entry in _alignment_entries(word_alignments)
        if entry["start_sec"] <= audio_sec
    ]
    if not entries:
        return None

    cutoff_sec = max(0.0, audio_sec - tail_sec)
    hyp_norm = [_normalize_match_word(unit) for unit in units]
    ref_norm = [_normalize_match_word(str(entry["text"])) for entry in entries]
    hyp_end_sec: list[float | None] = [None] * len(units)

    matcher = SequenceMatcher(None, ref_norm, hyp_norm, autojunk=False)
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag == "delete":
            continue
        if tag == "equal":
            for ref_idx, hyp_idx in zip(
                range(ref_start, ref_end),
                range(hyp_start, hyp_end),
                strict=True,
            ):
                hyp_end_sec[hyp_idx] = float(entries[ref_idx]["end_sec"])
            continue
        if tag == "replace":
            _spread_hypothesis_times_over_reference_span(
                hyp_end_sec,
                entries,
                ref_start=ref_start,
                ref_end=ref_end,
                hyp_start=hyp_start,
                hyp_end=hyp_end,
            )
            continue
        if tag == "insert":
            _spread_inserted_hypothesis_times(
                hyp_end_sec,
                entries,
                ref_index=ref_start,
                hyp_start=hyp_start,
                hyp_end=hyp_end,
            )

    kept_units = [
        unit
        for unit, end_sec in zip(units, hyp_end_sec, strict=True)
        if end_sec is not None and end_sec <= cutoff_sec
    ]
    return join_text_units(kept_units)


def _alignment_entries(word_alignments: Iterable[Any]) -> list[dict[str, float | str]]:
    entries: list[dict[str, float | str]] = []
    for item in word_alignments:
        if isinstance(item, dict):
            text = item.get("text") or item.get("word") or item.get("unit")
            start = item.get("start_sec", item.get("start_time", item.get("start")))
            end = item.get("end_sec", item.get("end_time", item.get("end")))
        else:
            text = getattr(item, "text", None)
            start = getattr(
                item,
                "start_sec",
                getattr(item, "start_time", getattr(item, "start", None)),
            )
            end = getattr(
                item,
                "end_sec",
                getattr(item, "end_time", getattr(item, "end", None)),
            )
        if text is None or start is None or end is None:
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        entries.append(
            {
                "text": str(text),
                "start_sec": start_f,
                "end_sec": end_f,
            }
        )
    return entries


def _normalize_match_word(word: str) -> str:
    return normalize_text_unit_for_match(word, case_sensitive=False)


def _spread_hypothesis_times_over_reference_span(
    hyp_end_sec: list[float | None],
    entries: list[dict[str, float | str]],
    *,
    ref_start: int,
    ref_end: int,
    hyp_start: int,
    hyp_end: int,
) -> None:
    hyp_count = hyp_end - hyp_start
    if hyp_count <= 0:
        return
    ref_count = ref_end - ref_start
    if ref_count <= 0:
        return
    if hyp_count == ref_count:
        for ref_idx, hyp_idx in zip(
            range(ref_start, ref_end),
            range(hyp_start, hyp_end),
            strict=True,
        ):
            hyp_end_sec[hyp_idx] = float(entries[ref_idx]["end_sec"])
        return

    span_start = float(entries[ref_start]["start_sec"])
    span_end = float(entries[ref_end - 1]["end_sec"])
    span = max(0.0, span_end - span_start)
    for offset, hyp_idx in enumerate(range(hyp_start, hyp_end), start=1):
        hyp_end_sec[hyp_idx] = span_start + span * offset / hyp_count


def _spread_inserted_hypothesis_times(
    hyp_end_sec: list[float | None],
    entries: list[dict[str, float | str]],
    *,
    ref_index: int,
    hyp_start: int,
    hyp_end: int,
) -> None:
    hyp_count = hyp_end - hyp_start
    if hyp_count <= 0:
        return
    prev_end = float(entries[ref_index - 1]["end_sec"]) if ref_index > 0 else 0.0
    if ref_index < len(entries):
        next_start = float(entries[ref_index]["start_sec"])
    else:
        next_start = prev_end + 0.2 * hyp_count
    span = max(0.0, next_start - prev_end)
    for offset, hyp_idx in enumerate(range(hyp_start, hyp_end), start=1):
        hyp_end_sec[hyp_idx] = prev_end + span * offset / hyp_count


def _append_only_update(
    published_units: list[str],
    source_text: str,
    *,
    normalize_for_match: bool,
) -> tuple[list[str], bool, int]:
    source_units = split_text_units(source_text)
    common = _common_text_prefix_units(
        published_units,
        source_units,
        normalize_for_match=normalize_for_match,
    )
    if common == len(published_units):
        return source_units, True, 0

    blocked_revision_units = max(0, len(published_units) - common)
    return published_units, False, blocked_revision_units


def _common_text_prefix_units(
    left: list[str],
    right: list[str],
    *,
    normalize_for_match: bool,
) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if normalize_for_match:
            left_unit = normalize_text_unit_for_match(
                left[idx],
                case_sensitive=False,
            )
            right_unit = normalize_text_unit_for_match(
                right[idx],
                case_sensitive=False,
            )
        else:
            left_unit = left[idx].strip()
            right_unit = right[idx].strip()
        if left_unit != right_unit:
            return idx
    return limit


def simulate_real_streaming_text(
    events: Iterable[dict[str, Any]],
    *,
    final_text: str,
    tail_sec: float = 0.25,
    normalize_for_match: bool = False,
    word_alignments: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the text a true append-only stream would have displayed.

    Per update we score only the hypothesis prefix older than ``tail_sec``.
    Already-published words are immutable: if a later snapshot would revise
    them, the published prefix is kept and the blocked rewrite is counted. The
    final flush uses ``final_text`` without the tail cut, but still cannot
    rewrite already-published words.
    """
    if tail_sec < 0.0:
        raise ValueError("tail_sec must be >= 0")

    alignments = list(word_alignments) if word_alignments is not None else None
    has_alignment_timestamps = bool(_alignment_entries(alignments)) if alignments else False
    published_units: list[str] = []
    annotated_events: list[dict[str, Any]] = []
    prefix_mismatch_events = 0
    blocked_revision_units = 0

    for event in events:
        event_audio_sec = float(event.get("audio_sec", 0.0) or 0.0)
        is_flush = bool(event.get("is_flush", False))
        source_text = time_truncated_text(
            str(event.get("hypothesis", "")),
            audio_sec=event_audio_sec,
            tail_sec=tail_sec,
            final=is_flush,
            word_alignments=alignments if has_alignment_timestamps else None,
        )
        published_units, prefix_match, blocked = _append_only_update(
            published_units,
            source_text,
            normalize_for_match=normalize_for_match,
        )
        if not prefix_match:
            prefix_mismatch_events += 1
            blocked_revision_units += blocked

        annotated = dict(event)
        annotated["real_streaming_source_text"] = source_text
        annotated["real_streaming_committed_text"] = join_text_units(published_units)
        annotated["real_streaming_prefix_match"] = prefix_match
        annotated["real_streaming_tail_sec"] = tail_sec
        annotated_events.append(annotated)

    published_units, final_prefix_match, final_blocked = _append_only_update(
        published_units,
        final_text,
        normalize_for_match=normalize_for_match,
    )
    if not final_prefix_match:
        prefix_mismatch_events += 1
        blocked_revision_units += final_blocked

    real_text = join_text_units(published_units)
    final_words = normalized_words(final_text)
    real_words = normalized_words(real_text)
    return {
        "text": real_text,
        "events": annotated_events,
        "tail_sec": tail_sec,
        "timestamp_source": (
            "forced_aligner" if has_alignment_timestamps else "uniform_text"
        ),
        "prefix_mismatch_events": prefix_mismatch_events,
        "blocked_revision_words": blocked_revision_units,
        "final_flush_prefix_match": final_prefix_match,
        "final_word_count": len(final_words),
        "real_streaming_word_count": len(real_words),
        "real_streaming_coverage_ratio": (
            float(len(real_words) / len(final_words)) if final_words else 0.0
        ),
    }


def streaming_text_event_stats(
    events: list[dict],
    *,
    final_text: str,
    stable_text: str,
) -> dict[str, int | float | None]:
    first_display_sec = None
    first_commit_sec = None
    for event in events:
        event_sec = float(event.get("audio_sec", 0.0))
        if first_display_sec is None and str(event.get("display", "")).strip():
            first_display_sec = event_sec
        if first_commit_sec is None and str(event.get("committed", "")).strip():
            first_commit_sec = event_sec

    final_words = normalized_words(final_text)
    stable_words = normalized_words(stable_text)
    display_revision = text_revision_stats(
        [str(event.get("display", "")) for event in events]
    )
    hypothesis_revision = text_revision_stats(
        [str(event.get("hypothesis", "")) for event in events]
    )
    committed_revision = text_revision_stats(
        [str(event.get("committed", "")) for event in events]
    )

    return {
        "first_display_sec": first_display_sec,
        "first_commit_sec": first_commit_sec,
        "final_word_count": len(final_words),
        "stable_word_count": len(stable_words),
        "stable_coverage_ratio": (
            float(len(stable_words) / len(final_words)) if final_words else 0.0
        ),
        "display_revision_events": int(display_revision["revision_events"]),
        "display_revision_words": int(display_revision["revision_words"]),
        "display_max_revision_words": int(display_revision["max_revision_words"]),
        "display_revision_event_ratio": float(
            display_revision["revision_event_ratio"]
        ),
        "hypothesis_revision_events": int(hypothesis_revision["revision_events"]),
        "hypothesis_revision_words": int(hypothesis_revision["revision_words"]),
        "committed_revision_events": int(committed_revision["revision_events"]),
        "committed_revision_words": int(committed_revision["revision_words"]),
    }
