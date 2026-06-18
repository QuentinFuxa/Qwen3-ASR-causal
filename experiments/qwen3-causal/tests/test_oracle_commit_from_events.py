import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "oracle_commit_from_events.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "oracle_commit_from_events",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)


def test_oracle_commit_skips_wrong_prefix_and_later_appends():
    events = [
        {"audio_sec": 1.0, "hypothesis": "yellow world"},
        {"audio_sec": 2.0, "hypothesis": "hello worm"},
        {"audio_sec": 3.0, "hypothesis": "hello world"},
    ]

    replayed, committed = oracle.replay_oracle_commit(
        events,
        target_text="hello world",
        tail_sec=0.0,
        normalize_for_match=True,
        final_flush=False,
    )

    assert replayed[0]["committed"] == ""
    assert replayed[1]["committed"] == "hello"
    assert replayed[2]["committed"] == "hello world"
    assert committed == "hello world"


def test_oracle_final_flush_is_append_only_to_target():
    events = [
        {"audio_sec": 1.0, "hypothesis": "hello"},
        {"audio_sec": 2.0, "hypothesis": "hello wor"},
    ]

    replayed, committed = oracle.replay_oracle_commit(
        events,
        target_text="hello world",
        tail_sec=0.0,
        normalize_for_match=True,
        final_flush=True,
    )

    assert replayed[-1]["committed"] == "hello world"
    assert committed == "hello world"


def test_oracle_commit_preserves_existing_surface_form():
    events = [
        {"audio_sec": 1.0, "hypothesis": "Hello"},
        {"audio_sec": 2.0, "hypothesis": "hello world"},
    ]

    replayed, committed = oracle.replay_oracle_commit(
        events,
        target_text="hello world",
        tail_sec=0.0,
        normalize_for_match=True,
        final_flush=False,
    )

    assert replayed[0]["committed"] == "Hello"
    assert replayed[1]["committed"] == "Hello world"
    assert committed == "Hello world"


def test_oracle_final_flush_preserves_existing_surface_form():
    events = [
        {"audio_sec": 1.0, "hypothesis": "Hello"},
    ]

    replayed, committed = oracle.replay_oracle_commit(
        events,
        target_text="hello world",
        tail_sec=0.0,
        normalize_for_match=True,
        final_flush=True,
    )

    assert replayed[-1]["committed"] == "Hello world"
    assert committed == "Hello world"


def test_evaluate_reads_manifest_alignments_and_reports_metrics(tmp_path):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "sample.jsonl").write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in [
                {"audio_sec": 1.0, "hypothesis": "hello early"},
                {"audio_sec": 2.0, "hypothesis": "hello world"},
            ]
        ),
        encoding="utf-8",
    )

    result = oracle.evaluate(
        prediction_rows=[
            {
                "id": "sample",
                "reference": "hello world",
                "final_text": "hello world",
                "last_hypothesis_text": "hello world",
                "error": None,
            }
        ],
        events_dir=events_dir,
        manifest_rows={
            "sample": {
                "word_alignments": [
                    {"text": "hello", "start_sec": 0.0, "end_sec": 0.5},
                    {"text": "world", "start_sec": 1.0, "end_sec": 1.8},
                ]
            }
        },
        target_mode="final",
        tail_sec=0.25,
        normalize_for_match=True,
        final_flush=True,
    )

    assert result["count"] == 1
    assert result["wer_oracle_mean"] == 0.0
    assert result["committed_revision_events_total"] == 0
    assert result["target_prefix_mismatch_count"] == 0
