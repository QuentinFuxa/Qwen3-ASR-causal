#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_streaming.metrics import (  # noqa: E402
    stable_prefix_stats,
    streaming_text_event_stats,
    time_truncated_text,
    word_error_rate,
)
from qwen3_streaming.stable_commit import (  # noqa: E402
    join_text_units,
    longest_common_text_prefix_length,
    split_text_units,
    text_prefix_matches,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.mean(kept) if kept else None


def _manifest_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    return {str(row.get("id")): row for row in rows if row.get("id") is not None}


def _target_text(
    row: dict[str, Any],
    *,
    target_mode: str,
    reference: str,
) -> str:
    if target_mode == "reference":
        return reference
    if target_mode == "final":
        return str(
            row.get("last_hypothesis_text")
            or row.get("causal_offline_text")
            or row.get("final_text")
            or ""
        )
    raise ValueError(f"unknown target_mode: {target_mode}")


def _oracle_update_units(
    *,
    committed_units: list[str],
    source_text: str,
    target_units: list[str],
    normalize_for_match: bool,
) -> list[str]:
    source_units = split_text_units(source_text)
    if not text_prefix_matches(
        committed_units,
        source_units,
        case_sensitive=False,
        normalize_for_match=normalize_for_match,
    ):
        return committed_units
    if not text_prefix_matches(
        committed_units,
        target_units,
        case_sensitive=False,
        normalize_for_match=normalize_for_match,
    ):
        return committed_units

    lcp_len = longest_common_text_prefix_length(
        source_units,
        target_units,
        case_sensitive=False,
        normalize_for_match=normalize_for_match,
    )
    commit_len = max(len(committed_units), lcp_len)
    if commit_len <= len(committed_units):
        return committed_units
    delta_units = source_units[len(committed_units) : commit_len]
    if (
        committed_units
        and delta_units
        and not committed_units[-1].endswith((" ", "\t", "\n"))
        and not delta_units[0].startswith((" ", "\t", "\n"))
    ):
        delta_units = [" " + delta_units[0]] + delta_units[1:]
    return committed_units + delta_units


def replay_oracle_commit(
    events: list[dict[str, Any]],
    *,
    target_text: str,
    tail_sec: float,
    normalize_for_match: bool,
    word_alignments: Iterable[Any] | None = None,
    final_flush: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    target_units = split_text_units(target_text)
    committed_units: list[str] = []
    replayed: list[dict[str, Any]] = []

    for event in events:
        source_text = time_truncated_text(
            str(event.get("hypothesis", "")),
            audio_sec=float(event.get("audio_sec", 0.0) or 0.0),
            tail_sec=tail_sec,
            final=False,
            word_alignments=word_alignments,
        )
        committed_units = _oracle_update_units(
            committed_units=committed_units,
            source_text=source_text,
            target_units=target_units,
            normalize_for_match=normalize_for_match,
        )
        replay_event = dict(event)
        replay_event["oracle_source_text"] = source_text
        replay_event["committed"] = join_text_units(committed_units)
        replay_event["display"] = source_text
        replay_event["unstable"] = ""
        replay_event["committed_units"] = len(committed_units)
        replayed.append(replay_event)

    if final_flush and replayed and text_prefix_matches(
        committed_units,
        target_units,
        case_sensitive=False,
        normalize_for_match=normalize_for_match,
    ):
        delta_units = target_units[len(committed_units) :]
        if (
            committed_units
            and delta_units
            and not committed_units[-1].endswith((" ", "\t", "\n"))
            and not delta_units[0].startswith((" ", "\t", "\n"))
        ):
            delta_units = [" " + delta_units[0]] + delta_units[1:]
        committed_units = committed_units + delta_units
        replayed[-1]["committed"] = join_text_units(committed_units)
        replayed[-1]["display"] = join_text_units(committed_units)
        replayed[-1]["committed_units"] = len(committed_units)

    return replayed, join_text_units(committed_units)


def evaluate(
    *,
    prediction_rows: list[dict[str, Any]],
    events_dir: Path,
    manifest_rows: dict[str, dict[str, Any]],
    target_mode: str,
    tail_sec: float,
    normalize_for_match: bool,
    final_flush: bool,
) -> dict[str, Any]:
    item_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        if row.get("error") is not None:
            continue
        item_id = str(row["id"])
        event_path = events_dir / f"{item_id}.jsonl"
        if not event_path.exists():
            raise FileNotFoundError(event_path)
        manifest_row = manifest_rows.get(item_id, {})
        reference = str(
            row.get("reference")
            or manifest_row.get("teacher_text")
            or manifest_row.get("text")
            or ""
        )
        target = _target_text(row, target_mode=target_mode, reference=reference)
        events = load_jsonl(event_path)
        replayed, committed_text = replay_oracle_commit(
            events,
            target_text=target,
            tail_sec=tail_sec,
            normalize_for_match=normalize_for_match,
            word_alignments=manifest_row.get("word_alignments"),
            final_flush=final_flush,
        )
        streaming = streaming_text_event_stats(
            replayed,
            final_text=target,
            stable_text=committed_text,
        )
        target_prefix = stable_prefix_stats(target, committed_text)
        item_rows.append(
            {
                "id": item_id,
                "reference": reference,
                "target_text": target,
                "oracle_committed_text": committed_text,
                "wer_oracle": word_error_rate(reference, committed_text)
                if reference
                else None,
                "streaming": streaming,
                "target_prefix_revision_words": target_prefix.revision_words,
                "target_prefix_mismatch": target_prefix.revision_words > 0,
            }
        )

    return {
        "target_mode": target_mode,
        "tail_sec": tail_sec,
        "normalize_for_match": normalize_for_match,
        "final_flush": final_flush,
        "count": len(item_rows),
        "wer_oracle_mean": _mean(item.get("wer_oracle") for item in item_rows),
        "first_display_sec_mean": _mean(
            item["streaming"].get("first_display_sec") for item in item_rows
        ),
        "first_commit_sec_mean": _mean(
            item["streaming"].get("first_commit_sec") for item in item_rows
        ),
        "stable_coverage_ratio_mean": _mean(
            item["streaming"].get("stable_coverage_ratio") for item in item_rows
        ),
        "stable_word_count_mean": _mean(
            item["streaming"].get("stable_word_count") for item in item_rows
        ),
        "committed_revision_events_total": sum(
            int(item["streaming"].get("committed_revision_events", 0))
            for item in item_rows
        ),
        "committed_revision_words_total": sum(
            int(item["streaming"].get("committed_revision_words", 0))
            for item in item_rows
        ),
        "target_prefix_mismatch_count": sum(
            1 for item in item_rows if item["target_prefix_mismatch"]
        ),
        "items": item_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved full-hypothesis streaming events with an oracle "
            "append-only commit policy. This is an upper bound for emission "
            "policy training; it never changes ASR hypotheses."
        )
    )
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, default=None)
    parser.add_argument(
        "--target-mode",
        choices=("final", "reference"),
        default="final",
        help=(
            "'final' uses the model final transcript as a deployable oracle "
            "upper bound; 'reference' is a non-deployable content upper bound."
        ),
    )
    parser.add_argument("--real-streaming-tail-ms", type=float, default=250.0)
    parser.add_argument("--normalize-commit-match", action="store_true")
    parser.add_argument("--no-final-flush", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_streaming_tail_ms < 0.0:
        raise ValueError("--real-streaming-tail-ms must be >= 0")
    prediction_rows = load_jsonl(args.predictions_jsonl)
    result = evaluate(
        prediction_rows=prediction_rows,
        events_dir=args.events_dir,
        manifest_rows=_manifest_by_id(args.manifest_jsonl),
        target_mode=args.target_mode,
        tail_sec=float(args.real_streaming_tail_ms) / 1000.0,
        normalize_for_match=args.normalize_commit_match,
        final_flush=not args.no_final_flush,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    printable = {key: value for key, value in result.items() if key != "items"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
