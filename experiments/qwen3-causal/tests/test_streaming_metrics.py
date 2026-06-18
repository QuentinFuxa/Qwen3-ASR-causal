from qwen3_streaming.metrics import (
    simulate_real_streaming_text,
    streaming_text_event_stats,
    text_revision_stats,
    time_truncated_text,
)


def test_text_revision_stats_counts_replaced_tail_words():
    stats = text_revision_stats(
        [
            "hello brave world",
            "hello bright world",
            "hello bright world today",
        ]
    )

    assert stats["revision_events"] == 1
    assert stats["revision_words"] == 2
    assert stats["max_revision_words"] == 2


def test_text_revision_stats_does_not_count_extensions():
    stats = text_revision_stats(["hello", "hello world", "hello world today"])

    assert stats["revision_events"] == 0
    assert stats["revision_words"] == 0


def test_streaming_text_event_stats_reports_latency_coverage_and_revisions():
    events = [
        {
            "audio_sec": 1.0,
            "display": "",
            "committed": "",
            "hypothesis": "",
        },
        {
            "audio_sec": 2.0,
            "display": "hello brave",
            "committed": "",
            "hypothesis": "hello brave",
        },
        {
            "audio_sec": 3.0,
            "display": "hello bright world",
            "committed": "hello",
            "hypothesis": "hello bright world",
        },
    ]

    stats = streaming_text_event_stats(
        events,
        final_text="hello bright world today",
        stable_text="hello",
    )

    assert stats["first_display_sec"] == 2.0
    assert stats["first_commit_sec"] == 3.0
    assert stats["stable_word_count"] == 1
    assert stats["final_word_count"] == 4
    assert stats["stable_coverage_ratio"] == 0.25
    assert stats["display_revision_events"] == 1
    assert stats["display_revision_words"] == 1
    assert stats["committed_revision_events"] == 0


def test_time_truncated_text_drops_estimated_audio_tail():
    assert (
        time_truncated_text(
            "one two three four",
            audio_sec=2.0,
            tail_sec=0.5,
        )
        == "one two three"
    )
    assert (
        time_truncated_text(
            "one two three four",
            audio_sec=2.0,
            tail_sec=0.5,
            final=True,
        )
        == "one two three four"
    )


def test_time_truncated_text_uses_forced_alignment_timestamps_when_available():
    alignments = [
        {"text": "one", "start_sec": 0.0, "end_sec": 0.2},
        {"text": "two", "start_sec": 0.2, "end_sec": 0.4},
        {"text": "three", "start_sec": 0.4, "end_sec": 0.6},
        {"text": "four", "start_sec": 1.2, "end_sec": 1.7},
    ]

    assert (
        time_truncated_text(
            "one two three four",
            audio_sec=2.0,
            tail_sec=0.25,
            word_alignments=alignments,
        )
        == "one two three four"
    )


def test_time_truncated_text_maps_substitutions_to_reference_time_slots():
    alignments = [
        {"text": "one", "start_sec": 0.0, "end_sec": 0.2},
        {"text": "two", "start_sec": 0.2, "end_sec": 0.4},
        {"text": "three", "start_sec": 0.4, "end_sec": 0.6},
    ]

    assert (
        time_truncated_text(
            "one too three",
            audio_sec=0.5,
            tail_sec=0.2,
            word_alignments=alignments,
        )
        == "one"
    )


def test_real_streaming_simulation_never_rewrites_published_prefix():
    result = simulate_real_streaming_text(
        [
            {"audio_sec": 1.0, "hypothesis": "one two three four"},
            {"audio_sec": 2.0, "hypothesis": "one two revised four five"},
        ],
        final_text="one two three four five",
        tail_sec=0.25,
    )

    assert result["events"][0]["real_streaming_committed_text"] == "one two three"
    assert result["events"][1]["real_streaming_committed_text"] == "one two three"
    assert result["prefix_mismatch_events"] == 1
    assert result["blocked_revision_words"] == 1
    assert result["final_flush_prefix_match"] is True
    assert result["text"] == "one two three four five"


def test_real_streaming_normalized_match_allows_case_and_punctuation_changes():
    result = simulate_real_streaming_text(
        [{"audio_sec": 1.0, "hypothesis": "the english forwarded"}],
        final_text="The English forwarded to the French baskets.",
        tail_sec=0.0,
        normalize_for_match=True,
    )

    assert result["final_flush_prefix_match"] is True
    assert result["prefix_mismatch_events"] == 0
    assert result["text"] == "The English forwarded to the French baskets."


def test_real_streaming_simulation_reports_forced_aligner_timestamp_source():
    result = simulate_real_streaming_text(
        [{"audio_sec": 1.0, "hypothesis": "one two"}],
        final_text="one two",
        tail_sec=0.25,
        word_alignments=[
            {"text": "one", "start_sec": 0.0, "end_sec": 0.2},
            {"text": "two", "start_sec": 0.2, "end_sec": 0.7},
        ],
    )

    assert result["timestamp_source"] == "forced_aligner"
    assert result["events"][0]["real_streaming_committed_text"] == "one two"


def test_real_streaming_final_flush_cannot_rewrite_past():
    result = simulate_real_streaming_text(
        [{"audio_sec": 1.0, "hypothesis": "one wrong"}],
        final_text="one right done",
        tail_sec=0.0,
    )

    assert result["text"] == "one wrong"
    assert result["final_flush_prefix_match"] is False
    assert result["prefix_mismatch_events"] == 1
