from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qwen3_asr_causal.vllm import (
    Qwen3VLLMCausalOnlineProcessor,
    Qwen3VLLMOnlineProcessor,
    _AlignedWord,
    _Qwen3VLLMCausalSession,
    _VLLMLiveCompactPromptEmbeds,
    _VLLMLiveTextDecoderSession,
    _accepted_vllm_prompt_draft_tokens,
    _apply_vllm_live_request_metadata_from_prompt,
    _compact_vllm_live_prompt_embeds,
    _decode_vllm_live_cache_salt,
    _decode_vllm_live_prompt_mask_metadata,
    _encode_vllm_live_cache_salt,
    _ensure_qwen3_asr_text_decoder_model,
    _qwen3_audio_feature_length_for_output_steps,
    _resolve_audio_backend,
    _resolve_causal_decoder_backend,
    _trim_token_ids_at_stop,
    _fix_timestamps,
    _load_vllm_runtime,
    _normalize_timestamp_segment_time,
    _patch_vllm_live_prompt_embeds_streaming,
    _prompt_input_token_count,
    _request_output_num_cached_tokens,
    _split_align_words,
    _vllm_live_full_prompt_is_token_ids,
    _vllm_live_prompt_input,
    _vllm_live_prompt_is_token_ids_from_prompt_embeds,
    _vllm_live_prompt_is_token_ids_from_cache_salt,
    _vllm_live_prompt_is_token_ids_from_repeated_placeholder,
)
import qwen3_asr_causal.vllm as qwen_vllm


class MockQwen3VLLM:
    def __init__(self, aligned):
        self.aligned = aligned
        self.calls = 0

    def transcribe_aligned(self, audio):
        self.calls += 1
        return self.aligned, "English"


class SequencedMockQwen3VLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def transcribe_aligned(self, audio):
        self.calls += 1
        if not self.responses:
            return [], "English"
        return self.responses.pop(0), "English"


class MockCausalSession:
    def __init__(self, text, *, cached_steps=0):
        self.text = text
        self.appended_samples = 0
        self.audio_samples_seen = 0
        self.decode_flush_flags = []
        self.cached_steps = cached_steps

    def append_audio(self, audio):
        self.appended_samples += len(audio)
        self.audio_samples_seen += len(audio)

    def decode_text(self, *, flush=False):
        self.decode_flush_flags.append(flush)
        return self.text, "English"

    def cached_audio_steps(self):
        return self.cached_steps


class SequencedMockCausalSession:
    def __init__(self, texts, *, cached_steps=0):
        self.texts = list(texts)
        self.appended_samples = 0
        self.audio_samples_seen = 0
        self.decode_flush_flags = []
        self.cached_steps = cached_steps

    def append_audio(self, audio):
        self.appended_samples += len(audio)
        self.audio_samples_seen += len(audio)

    def decode_text(self, *, flush=False):
        self.decode_flush_flags.append(flush)
        if self.texts:
            return self.texts.pop(0), "English"
        return "", "English"

    def cached_audio_steps(self):
        return self.cached_steps


class MockCausalQwen3VLLM:
    audio_backend = "causal"
    original_language = "en"

    def __init__(self, aligned, text="one two three", *, segment_max_steps=0):
        self.aligned = aligned
        self.text = text
        self.sessions = []
        self.align_calls = 0
        self.session_contexts = []
        self.causal_asr = SimpleNamespace(
            segment_max_steps=segment_max_steps,
            prompt_context_words=4,
        )

    def new_causal_session(self, language=None, *, context=""):
        self.session_contexts.append(context)
        session = MockCausalSession(self.text)
        self.sessions.append(session)
        return session

    def align_words(self, audio, text, language):
        self.align_calls += 1
        return self.aligned


class SequencedMockCausalQwen3VLLM(MockCausalQwen3VLLM):
    def __init__(self, aligned_responses, texts, *, segment_max_steps=0, cached_steps=0):
        super().__init__([], segment_max_steps=segment_max_steps)
        self.aligned_responses = list(aligned_responses)
        self.texts = list(texts)
        self.align_texts = []
        self.align_audio_lengths = []
        self.cached_steps = cached_steps

    def new_causal_session(self, language=None, *, context=""):
        self.session_contexts.append(context)
        session = SequencedMockCausalSession(
            self.texts,
            cached_steps=self.cached_steps,
        )
        self.sessions.append(session)
        return session

    def align_words(self, audio, text, language):
        self.align_calls += 1
        self.align_texts.append(text)
        self.align_audio_lengths.append(len(audio))
        if not self.aligned_responses:
            return []
        return self.aligned_responses.pop(0)


def _audio(seconds):
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def test_qwen3_vllm_commits_only_before_last_250ms():
    asr = MockQwen3VLLM(
        [
            _AlignedWord("one", 0.0, 1.0),
            _AlignedWord("two", 1.0, 9.70),
            _AlignedWord("three", 9.70, 9.80),
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(10), 10.0)

    committed, _ = processor.process_iter()

    assert [token.text.strip() for token in committed] == ["one", "two"]
    assert processor.get_buffer().text.strip() == "three"


def test_qwen3_vllm_finish_flushes_last_250ms():
    asr = MockQwen3VLLM(
        [
            _AlignedWord("one", 0.0, 1.0),
            _AlignedWord("two", 1.0, 9.80),
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(10), 10.0)

    first, _ = processor.process_iter()
    final, _ = processor.finish()

    assert [token.text.strip() for token in first] == ["one"]
    assert [token.text.strip() for token in final] == ["two"]
    assert processor.get_buffer().text == ""


def test_qwen3_vllm_finish_flushes_cached_buffer_if_final_retry_is_empty():
    asr = SequencedMockQwen3VLLM(
        [
            [_AlignedWord("late", 0.80, 0.95)],
            [],
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(1.0), 1.0)

    first, _ = processor.process_iter()
    final, _ = processor.finish()

    assert first == []
    assert processor.get_buffer().text == ""
    assert [token.text.strip() for token in final] == ["late"]


def test_qwen3_vllm_no_duplicate_on_same_buffer():
    asr = MockQwen3VLLM(
        [
            _AlignedWord("one", 0.0, 1.0),
            _AlignedWord("two", 1.0, 2.0),
            _AlignedWord("three", 2.0, 3.0),
        ]
    )
    processor = Qwen3VLLMOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(4), 4.0)

    first, _ = processor.process_iter()
    second, _ = processor.process_iter(is_last=True)

    assert [token.text.strip() for token in first] == ["one", "two", "three"]
    assert second == []


def test_qwen3_vllm_causal_processor_keeps_append_only_prefix_and_holdback():
    asr = SequencedMockCausalQwen3VLLM(
        aligned_responses=[
            [
                _AlignedWord("one", 0.0, 1.0),
                _AlignedWord("two", 1.0, 9.70),
                _AlignedWord("three", 9.70, 9.80),
            ],
            [_AlignedWord("three", 2.0, 2.1)],
        ],
        texts=["one two three", "one two three"],
    )
    processor = Qwen3VLLMCausalOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(10), 10.0)

    committed, _ = processor.process_iter()
    final, _ = processor.finish()

    assert asr.sessions[0].appended_samples == 160_000
    assert asr.sessions[0].decode_flush_flags == [False, True]
    assert [token.text.strip() for token in committed] == ["one", "two"]
    assert [token.text.strip() for token in final] == ["three"]


def test_qwen3_vllm_causal_processor_aligns_only_uncommitted_suffix():
    asr = SequencedMockCausalQwen3VLLM(
        aligned_responses=[
            [
                _AlignedWord("one", 0.0, 1.0),
                _AlignedWord("two", 4.0, 5.0),
            ],
            [_AlignedWord("three", 3.5, 4.0)],
        ],
        texts=["one two", "one two three"],
    )
    processor = Qwen3VLLMCausalOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(6), 6.0)
    first, _ = processor.process_iter()
    processor.insert_audio_chunk(_audio(2), 8.0)
    second, _ = processor.process_iter()

    assert [token.text.strip() for token in first] == ["one", "two"]
    assert [token.text.strip() for token in second] == ["three"]
    assert second[0].text == " three"
    assert asr.align_texts == ["one two", "three"]
    assert asr.align_audio_lengths[0] == 6 * 16_000
    assert asr.align_audio_lengths[1] == 5 * 16_000
    assert processor._buffer_time_offset == 3.0
    assert asr.sessions[0].appended_samples == 8 * 16_000


def test_qwen3_vllm_causal_processor_keeps_overlap_for_decode_prefix():
    processor = Qwen3VLLMCausalOnlineProcessor.__new__(Qwen3VLLMCausalOnlineProcessor)
    processor.asr = SimpleNamespace(
        causal_decode_committed_prefix=True,
        causal_decode_prefix_overlap_words=2,
    )
    processor._session_committed_text = "one two three four"
    processor.session = SimpleNamespace(decode_prefix_text="one two")

    assert processor._decode_forced_prefix_text() == "one two"
    assert processor._text_suffix_after_committed("one two three four five") == (
        "five",
        True,
    )
    assert processor._text_suffix_after_committed("one two three five six") == (
        "five six",
        True,
    )
    assert processor._text_suffix_after_committed("one two five six") == (
        "five six",
        True,
    )


def test_qwen3_vllm_causal_processor_rolls_bounded_audio_session_after_commit():
    asr = SequencedMockCausalQwen3VLLM(
        aligned_responses=[
            [
                _AlignedWord("one", 0.0, 1.0),
                _AlignedWord("two", 4.0, 5.0),
                _AlignedWord("three", 5.8, 5.9),
            ],
        ],
        texts=["one two three"],
        segment_max_steps=2,
        cached_steps=3,
    )
    processor = Qwen3VLLMCausalOnlineProcessor(asr)
    processor.insert_audio_chunk(_audio(6), 6.0)

    committed, _ = processor.process_iter()

    assert [token.text.strip() for token in committed] == ["one", "two"]
    assert len(asr.sessions) == 2
    assert asr.session_contexts == ["", "one two"]
    assert processor._session_committed_text == ""
    assert processor._committed_text == "one two"
    assert processor._buffer_time_offset == 3.0
    assert asr.sessions[1].appended_samples == 3 * 16_000


def test_qwen3_vllm_causal_session_prompt_carries_cache_salt():
    import torch

    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [11]
    session.asr = SimpleNamespace(causal_audio_embeds_mm_enabled=False)
    session.frame_hidden = torch.ones((1, 2, 3), dtype=torch.float32)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
    )

    prompt = session._prompt_input()

    assert prompt["cache_salt"] == "session-salt"
    assert prompt["prompt_token_ids"] == [10, 7, 7, 11]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True]
    assert prompt["is_token_ids"] == [True, False, False, True]
    assert prompt["prompt_embeds"].shape == (4, 3)


def test_qwen3_vllm_causal_session_prompt_appends_committed_text_prefix():
    import torch

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "hello world"
            assert add_special_tokens is False
            return [21, 22]

    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [11]
    session.decode_prefix_text = "hello world"
    session.asr = SimpleNamespace(
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-text",
        causal_decode_committed_prefix=True,
    )
    session.frame_hidden = torch.ones((1, 2, 3), dtype=torch.float32)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        qwen_tokenizer=FakeTokenizer(),
    )

    prompt = session._prompt_input()

    assert prompt["prompt_token_ids"] == [10, 7, 7, 11, 21, 22]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True, True, True]
    assert prompt["prompt_embeds"].shape == (6, 3)
    assert torch.equal(prompt["prompt_embeds"][1:3], torch.ones((2, 3)))
    assert torch.equal(prompt["prompt_embeds"][4:], torch.zeros((2, 3)))


def test_qwen3_vllm_causal_session_live_prompt_marks_reused_audio_prefix():
    import torch

    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [11]
    session.asr = SimpleNamespace(causal_audio_embeds_mm_enabled=False)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
    )
    session._vllm_live_prompt_audio_steps = 0

    session.frame_hidden = torch.ones((1, 2, 3), dtype=torch.float32)
    first_prompt = session._vllm_live_prompt_input()
    assert first_prompt["type"] == "embeds"
    assert first_prompt["is_token_ids"] == first_prompt["prompt_is_token_ids"]
    assert len(first_prompt["is_token_ids"]) == first_prompt["prompt_embeds"].shape[0]
    assert _decode_vllm_live_cache_salt(first_prompt["cache_salt"]) == (
        "session-salt",
        0,
    )

    session.frame_hidden = torch.ones((1, 5, 3), dtype=torch.float32)
    second_prompt = session._vllm_live_prompt_input()
    assert second_prompt["type"] == "embeds"
    assert second_prompt["is_token_ids"] == second_prompt["prompt_is_token_ids"]
    assert len(second_prompt["is_token_ids"]) == second_prompt["prompt_embeds"].shape[0]
    assert second_prompt["prompt_token_ids"] == [10, 7, 7, 7, 7, 7, 11]
    assert _decode_vllm_live_cache_salt(second_prompt["cache_salt"]) == (
        "session-salt",
        3,
    )


def test_qwen3_vllm_causal_session_can_use_multimodal_audio_embeds_blocks():
    import torch

    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [11]
    session.asr = SimpleNamespace(
        causal_audio_embeds_mm_enabled=True,
        causal_vllm_audio_embed_block_steps=2,
    )
    session.frame_hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
    )

    prompt = session._prompt_input()

    assert "prompt_embeds" not in prompt
    assert prompt["cache_salt"] == "session-salt"
    assert prompt["prompt_token_ids"] == [10, 7, 7, 7, 11]
    audio_data = prompt["multi_modal_data"]["audio"]
    assert audio_data["audio_embeds"].shape == (5, 3)
    assert audio_data["audio_embed_lengths"].tolist() == [2, 2, 1]
    assert audio_data["audio_feature_lengths"].tolist() == [
        _qwen3_audio_feature_length_for_output_steps(2),
        _qwen3_audio_feature_length_for_output_steps(2),
        _qwen3_audio_feature_length_for_output_steps(1),
    ]


def test_qwen3_vllm_causal_session_rolling_decoder_uses_previous_draft():
    import torch

    calls = []

    class FakeModel:
        def generate_full_hypothesis_rolling(self, *args, **kwargs):
            calls.append((args, kwargs))
            return torch.tensor([[1, 2, 99]]), {"decoder_path": "rolling+draft"}

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [1, 2]
            assert skip_special_tokens is True
            return "language English<asr_text>hello world"

    state = SimpleNamespace()
    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.asr = SimpleNamespace(
        causal_decoder_backend="rolling",
        max_decode_tokens=5,
    )
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        eos_token_id=99,
        max_consecutive_text_tokens=0,
        model=FakeModel(),
        no_repeat_ngram_size=3,
        qwen_tokenizer=FakeTokenizer(),
        repetition_penalty=1.15,
        suppress_token_ids=(10, 11),
        wait_token_id=10,
        word_start_token_id=11,
    )
    session.frame_hidden = torch.zeros((1, 3, 4), dtype=torch.float32)
    session.last_hypothesis_tokens = [42]
    session.prompt_template_ids = [12, 7, 13]
    session.qwen_language = "English"
    session.state = state

    text, language = session.decode_text()

    assert text == "hello world"
    assert language == "English"
    assert session.last_hypothesis_tokens == [1, 2]
    assert session.last_generation_stats == {
        "decoder_path": "rolling+draft",
        "decoder_backend": "rolling",
        "audio_steps": 3,
    }
    assert calls[0][0] == (session.frame_hidden,)
    assert calls[0][1]["state"] is state
    assert calls[0][1]["template_token_ids"] == [12, 7, 13]
    assert calls[0][1]["audio_placeholder_token_id"] == 7
    assert calls[0][1]["draft_token_ids"] == [42]
    assert calls[0][1]["max_new_tokens"] == 5


def test_qwen3_vllm_causal_session_append_kv_reports_append_stats():
    import torch

    class FakeModel:
        def generate_full_hypothesis_rolling(self, *args, **kwargs):
            return torch.tensor([[1, 2, 99]]), {
                "decoder_path": "rolling+draft",
                "decoder_rebuilt": False,
                "audio_steps": 12,
                "audio_delta_steps": 3,
                "reused_audio_steps": 9,
                "prefill_positions": 7,
            }

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return "language English<asr_text>hello world"

    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.asr = SimpleNamespace(
        causal_decoder_backend="append-kv",
        max_decode_tokens=5,
    )
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        eos_token_id=99,
        max_consecutive_text_tokens=0,
        model=FakeModel(),
        no_repeat_ngram_size=3,
        qwen_tokenizer=FakeTokenizer(),
        repetition_penalty=1.15,
        suppress_token_ids=(10, 11),
        wait_token_id=10,
        word_start_token_id=11,
    )
    session.frame_hidden = torch.zeros((1, 12, 4), dtype=torch.float32)
    session.last_hypothesis_tokens = [42]
    session.prompt_template_ids = [12, 7, 13]
    session.qwen_language = "English"
    session.state = SimpleNamespace()

    text, language = session.decode_text()

    assert text == "hello world"
    assert language == "English"
    assert session.last_generation_stats["decoder_path"] == "append-kv+draft"
    assert session.last_generation_stats["decoder_backend"] == "append-kv"
    assert session.last_generation_stats["audio_steps"] == 12
    assert session.last_generation_stats["audio_delta_steps"] == 3
    assert session.last_generation_stats["reused_audio_steps"] == 9
    assert session.last_generation_stats["prefill_positions"] == 7


def test_qwen3_vllm_causal_session_records_vllm_prefix_cache_stats():
    import torch

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOutput:
        num_cached_tokens = 3

        def __init__(self):
            self.outputs = [SimpleNamespace(text="language English<asr_text>hello")]

    class FakeLLM:
        def generate(self, prompts, params, use_tqdm=False):
            assert len(prompts) == 1
            assert prompts[0]["prompt_embeds"].shape == (6, 4)
            assert params.kwargs["temperature"] == 0.0
            assert use_tqdm is False
            return [FakeOutput()]

    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10, 11]
    session.prompt_tail_ids = [12]
    session.asr = SimpleNamespace(
        _SamplingParams=FakeSamplingParams,
        asr_llm=FakeLLM(),
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-text",
        max_decode_tokens=5,
    )
    session.frame_hidden = torch.ones((1, 3, 4), dtype=torch.float32)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        eos_token_id=99,
    )
    session.qwen_language = "English"

    text, language = session.decode_text()

    assert text == "hello"
    assert language == "English"
    assert session.last_generation_stats == {
        "decoder_path": "vllm-text",
        "decoder_backend": "vllm-text",
        "prompt_tokens": 6,
        "vllm_cached_tokens": 3,
        "vllm_effective_prefill_tokens": 3,
        "audio_steps": 3,
        "decode_prefix_tokens": 0,
        "decode_prefix_chars": 0,
    }


def test_qwen3_vllm_text_draft_accepts_verified_previous_hypothesis():
    import torch

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOutput:
        num_cached_tokens = 3

        def __init__(self):
            self.prompt_logprobs = [
                None,
                None,
                None,
                None,
                {21: SimpleNamespace(rank=1)},
                {22: SimpleNamespace(rank=1)},
            ]
            self.outputs = [
                SimpleNamespace(
                    text="unused",
                    token_ids=[23, 99],
                )
            ]

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompts, params, use_tqdm=False):
            self.calls.append((prompts[0], params, use_tqdm))
            assert params.kwargs["prompt_logprobs"] == 1
            return [FakeOutput()]

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [21, 22, 23]
            assert skip_special_tokens is True
            return "hello world now"

    llm = FakeLLM()
    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [12]
    session.asr = SimpleNamespace(
        _SamplingParams=FakeSamplingParams,
        asr_llm=llm,
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-text",
        causal_vllm_text_draft_enabled=True,
        max_decode_tokens=8,
    )
    session.frame_hidden = torch.ones((1, 2, 4), dtype=torch.float32)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        eos_token_id=99,
        qwen_tokenizer=FakeTokenizer(),
        wait_token_id=None,
        word_start_token_id=None,
    )
    session.qwen_language = "English"
    session.last_hypothesis_tokens = [21, 22]
    session.decode_prefix_text = ""

    text, language = session.decode_text()

    assert text == "hello world now"
    assert language == "English"
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0]
    assert prompt["prompt_token_ids"] == [10, 7, 7, 12, 21, 22]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True, True, True]
    assert session.last_hypothesis_tokens == [21, 22, 23]
    assert session.last_generation_stats["decoder_path"] == "vllm-text+draft"
    assert session.last_generation_stats["draft_tokens"] == 2
    assert session.last_generation_stats["draft_accepted"] == 2
    assert session.last_generation_stats["draft_unverifiable"] == 0
    assert session.last_generation_stats["draft_all_accepted"] is True
    assert session.last_generation_stats["draft_fallback"] == 0
    assert session.last_generation_stats["prompt_tokens"] == 6


def test_qwen3_vllm_text_draft_falls_back_when_prompt_logprobs_reject():
    import torch

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompts, params, use_tqdm=False):
            self.calls.append((prompts[0], params, use_tqdm))
            if len(self.calls) == 1:
                return [
                    SimpleNamespace(
                        num_cached_tokens=3,
                        prompt_logprobs=[
                            None,
                            None,
                            None,
                            None,
                            {21: SimpleNamespace(rank=1)},
                            {22: SimpleNamespace(rank=2)},
                        ],
                        outputs=[SimpleNamespace(text="bad path", token_ids=[50])],
                    )
                ]
            return [
                SimpleNamespace(
                    num_cached_tokens=3,
                    prompt_logprobs=None,
                    outputs=[
                        SimpleNamespace(
                            text="language English<asr_text>fresh",
                            token_ids=[24, 99],
                        )
                    ],
                )
            ]

    llm = FakeLLM()
    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [12]
    session.asr = SimpleNamespace(
        _SamplingParams=FakeSamplingParams,
        asr_llm=llm,
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-text",
        causal_vllm_text_draft_enabled=True,
        max_decode_tokens=8,
    )
    session.frame_hidden = torch.ones((1, 2, 4), dtype=torch.float32)
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        eos_token_id=99,
        qwen_tokenizer=SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=True: "unused"
        ),
    )
    session.qwen_language = "English"
    session.last_hypothesis_tokens = [21, 22]
    session.decode_prefix_text = ""

    text, language = session.decode_text()

    assert text == "fresh"
    assert language == "English"
    assert len(llm.calls) == 2
    assert llm.calls[0][0]["prompt_token_ids"] == [10, 7, 7, 12, 21, 22]
    assert llm.calls[0][1].kwargs["prompt_logprobs"] == 1
    assert llm.calls[1][0]["prompt_token_ids"] == [10, 7, 7, 12]
    assert "prompt_logprobs" not in llm.calls[1][1].kwargs
    assert session.last_hypothesis_tokens == [24]
    assert session.last_generation_stats["decoder_path"] == "vllm-text"
    assert session.last_generation_stats["draft_tokens"] == 2
    assert session.last_generation_stats["draft_accepted"] == 1
    assert session.last_generation_stats["draft_unverifiable"] == 0
    assert session.last_generation_stats["draft_all_accepted"] is False
    assert session.last_generation_stats["draft_fallback"] == 1


def test_vllm_prompt_draft_verification_helpers():
    logprobs = [
        None,
        {10: SimpleNamespace(rank=1)},
        {11: SimpleNamespace(rank=1)},
        {12: SimpleNamespace(rank=3)},
    ]

    assert _accepted_vllm_prompt_draft_tokens(
        logprobs,
        draft_start=1,
        draft_token_ids=[10, 11, 12],
    ) == 2
    shifted_logprobs = [
        {10: SimpleNamespace(rank=1)},
        {11: SimpleNamespace(rank=1)},
        {12: SimpleNamespace(rank=3)},
    ]
    assert _accepted_vllm_prompt_draft_tokens(
        shifted_logprobs,
        draft_start=1,
        draft_token_ids=[10, 11, 12],
        prompt_len=4,
    ) == 2
    assert _accepted_vllm_prompt_draft_tokens(
        logprobs,
        draft_start=1,
        draft_token_ids=[10, 99],
    ) is None
    assert _trim_token_ids_at_stop([1, 2, 99, 3], [99]) == [1, 2]


def test_qwen3_vllm_causal_session_vllm_live_reuses_audio_prefix():
    import torch

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLiveSession:
        def __init__(self):
            self.calls = []

        def decode(self, prompt, params, *, idle_timeout=None):
            self.calls.append((prompt, params, idle_timeout))
            return SimpleNamespace(outputs=[SimpleNamespace(text="language English<asr_text>hello")])

    live_session = FakeLiveSession()
    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10, 11]
    session.prompt_tail_ids = [12]
    session.asr = SimpleNamespace(
        _SamplingParams=FakeSamplingParams,
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-live",
        max_decode_tokens=5,
        causal_vllm_live_idle_timeout_sec=0.012,
    )
    session.vllm_live_decoder_session = live_session
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        eos_token_id=99,
    )
    session.qwen_language = "English"
    session._vllm_live_prompt_audio_steps = 0

    session.frame_hidden = torch.ones((1, 2, 4), dtype=torch.float32)
    text, language = session.decode_text()

    assert text == "hello"
    assert language == "English"
    assert _decode_vllm_live_cache_salt(live_session.calls[0][0]["cache_salt"]) == (
        "session-salt",
        0,
    )
    assert live_session.calls[0][0]["type"] == "embeds"
    assert live_session.calls[0][0]["is_token_ids"] == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert session.last_generation_stats["decoder_path"] == "vllm-live"
    assert session.last_generation_stats["vllm_live_prefix_keep_tokens"] == 0
    assert session.last_generation_stats["vllm_live_suffix_prefill_tokens"] == 5
    assert session.last_generation_stats["audio_delta_steps"] == 2
    assert session.last_generation_stats["reused_audio_steps"] == 0

    session.frame_hidden = torch.ones((1, 5, 4), dtype=torch.float32)
    text, language = session.decode_text()

    assert text == "hello"
    assert language == "English"
    assert _decode_vllm_live_cache_salt(live_session.calls[1][0]["cache_salt"]) == (
        "session-salt",
        4,
    )
    assert session.last_generation_stats["prompt_tokens"] == 8
    assert session.last_generation_stats["vllm_cached_tokens"] == 4
    assert session.last_generation_stats["vllm_effective_prefill_tokens"] == 4
    assert session.last_generation_stats["vllm_live_prefix_keep_tokens"] == 4
    assert session.last_generation_stats["vllm_live_suffix_prefill_tokens"] == 4
    assert session.last_generation_stats["audio_delta_steps"] == 3
    assert session.last_generation_stats["reused_audio_steps"] == 2
    assert live_session.calls[1][1].kwargs["temperature"] == 0.0
    assert live_session.calls[1][1].kwargs["detokenize"] is False
    output_kind = live_session.calls[1][1].kwargs.get("output_kind")
    if output_kind is not None:
        assert getattr(output_kind, "name", None) == "DELTA"
    assert live_session.calls[1][1].kwargs["stop_token_ids"] == [99]
    assert live_session.calls[1][2] == 0.012


def test_qwen3_vllm_live_draft_accepts_verified_previous_hypothesis():
    import torch

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [21, 22, 23]
            assert skip_special_tokens is True
            return "hello world now"

    class FakeLiveSession:
        def __init__(self):
            self.calls = []

        def decode(self, prompt, params, *, idle_timeout=None):
            self.calls.append((prompt, params, idle_timeout))
            base = len(prompt["prompt_token_ids"]) - 2
            return SimpleNamespace(
                prompt_logprobs=(
                    [None] * (base - 1)
                    + [{21: SimpleNamespace(rank=1)}, {22: SimpleNamespace(rank=1)}]
                ),
                outputs=[
                    SimpleNamespace(
                        text="unused",
                        token_ids=[23, 99],
                    )
                ],
            )

    live_session = FakeLiveSession()
    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [12]
    session.asr = SimpleNamespace(
        _SamplingParams=FakeSamplingParams,
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-live",
        causal_vllm_live_draft_enabled=True,
        causal_vllm_live_idle_timeout_sec=0.012,
        max_decode_tokens=8,
    )
    session.vllm_live_decoder_session = live_session
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        eos_token_id=99,
        qwen_tokenizer=FakeTokenizer(),
        wait_token_id=None,
        word_start_token_id=None,
    )
    session.qwen_language = "English"
    session._vllm_live_prompt_audio_steps = 2
    session.frame_hidden = torch.ones((1, 5, 4), dtype=torch.float32)
    session.last_hypothesis_tokens = [21, 22]
    session.decode_prefix_text = ""

    text, language = session.decode_text()

    assert text == "hello world now"
    assert language == "English"
    assert len(live_session.calls) == 1
    prompt, params, idle_timeout = live_session.calls[0]
    assert prompt["prompt_token_ids"] == [10, 7, 7, 7, 7, 7, 12, 21, 22]
    assert prompt["prompt_is_token_ids"] == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert _decode_vllm_live_cache_salt(prompt["cache_salt"]) == (
        "session-salt",
        3,
    )
    assert params.kwargs["max_tokens"] == 6
    assert params.kwargs["prompt_logprobs"] == 1
    assert idle_timeout == 0.012
    assert session.last_hypothesis_tokens == [21, 22, 23]
    assert session.last_generation_stats["decoder_path"] == "vllm-live+draft"
    assert session.last_generation_stats["draft_tokens"] == 2
    assert session.last_generation_stats["draft_accepted"] == 2
    assert session.last_generation_stats["draft_unverifiable"] == 0
    assert session.last_generation_stats["draft_all_accepted"] is True
    assert session.last_generation_stats["draft_fallback"] == 0
    assert session.last_generation_stats["audio_delta_steps"] == 3
    assert session.last_generation_stats["reused_audio_steps"] == 2


def test_qwen3_vllm_live_draft_falls_back_to_audio_prefix_on_reject():
    import torch

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLiveSession:
        def __init__(self):
            self.calls = []

        def decode(self, prompt, params, *, idle_timeout=None):
            self.calls.append((prompt, params, idle_timeout))
            if len(self.calls) == 1:
                base = len(prompt["prompt_token_ids"]) - 2
                return SimpleNamespace(
                    prompt_logprobs=(
                        [None] * (base - 1)
                        + [
                            {21: SimpleNamespace(rank=1)},
                            {22: SimpleNamespace(rank=2)},
                        ]
                    ),
                    outputs=[SimpleNamespace(text="bad path", token_ids=[50])],
                )
            return SimpleNamespace(
                prompt_logprobs=None,
                outputs=[
                    SimpleNamespace(
                        text="language English<asr_text>fresh",
                        token_ids=[24, 99],
                    )
                ],
            )

    live_session = FakeLiveSession()
    session = _Qwen3VLLMCausalSession.__new__(_Qwen3VLLMCausalSession)
    session.cache_salt = "session-salt"
    session.prompt_head_ids = [10]
    session.prompt_tail_ids = [12]
    session.asr = SimpleNamespace(
        _SamplingParams=FakeSamplingParams,
        causal_audio_embeds_mm_enabled=False,
        causal_decoder_backend="vllm-live",
        causal_vllm_live_draft_enabled=True,
        causal_vllm_live_idle_timeout_sec=0.012,
        max_decode_tokens=8,
    )
    session.vllm_live_decoder_session = live_session
    session.causal_asr = SimpleNamespace(
        audio_placeholder_token_id=7,
        decode_lock=nullcontext(),
        device="cpu",
        eos_token_id=99,
    )
    session.qwen_language = "English"
    session._vllm_live_prompt_audio_steps = 2
    session.frame_hidden = torch.ones((1, 5, 4), dtype=torch.float32)
    session.last_hypothesis_tokens = [21, 22]
    session.decode_prefix_text = ""

    text, language = session.decode_text()

    assert text == "fresh"
    assert language == "English"
    assert len(live_session.calls) == 2
    first_prompt, first_params, _ = live_session.calls[0]
    second_prompt, second_params, _ = live_session.calls[1]
    assert first_prompt["prompt_token_ids"] == [10, 7, 7, 7, 7, 7, 12, 21, 22]
    assert _decode_vllm_live_cache_salt(first_prompt["cache_salt"]) == (
        "session-salt",
        3,
    )
    assert first_params.kwargs["prompt_logprobs"] == 1
    assert second_prompt["prompt_token_ids"] == [10, 7, 7, 7, 7, 7, 12]
    assert _decode_vllm_live_cache_salt(second_prompt["cache_salt"]) == (
        "session-salt",
        6,
    )
    assert "prompt_logprobs" not in second_params.kwargs
    assert session.last_hypothesis_tokens == [24]
    assert session.last_generation_stats["decoder_path"] == "vllm-live"
    assert session.last_generation_stats["draft_tokens"] == 2
    assert session.last_generation_stats["draft_accepted"] == 1
    assert session.last_generation_stats["draft_unverifiable"] == 0
    assert session.last_generation_stats["draft_all_accepted"] is False
    assert session.last_generation_stats["draft_fallback"] == 1
    assert session.last_generation_stats["audio_delta_steps"] == 0
    assert session.last_generation_stats["reused_audio_steps"] == 5


def test_vllm_live_decoder_session_returns_last_output_on_idle():
    import queue

    output = SimpleNamespace(
        finished=False,
        outputs=[SimpleNamespace(text="partial", finish_reason=None)],
    )

    class FakeDecoder:
        def __init__(self):
            self.inputs = []

        def _put_session_input(self, session, prompt_input, sampling_params):
            self.inputs.append((prompt_input, sampling_params))

    session = _VLLMLiveTextDecoderSession.__new__(_VLLMLiveTextDecoderSession)
    session._decoder = FakeDecoder()
    session._result_queue = queue.Queue()
    session._result_queue.put(output)
    session._async_input_queue = None
    session._started = True
    session._closed = False
    session._last_returned_on_idle = False

    params = SimpleNamespace()
    result = session.decode(
        {"prompt_token_ids": [1]},
        params,
        timeout=1.0,
        idle_timeout=0.01,
    )

    assert result is output
    assert session._last_returned_on_idle is True
    assert session._decoder.inputs == [({"prompt_token_ids": [1]}, params)]


def test_vllm_live_decoder_session_accumulates_delta_outputs_on_idle():
    import queue

    first = SimpleNamespace(
        finished=False,
        prompt_logprobs=["prompt-logprobs"],
        outputs=[
            SimpleNamespace(
                text="hel",
                token_ids=[10],
                finish_reason=None,
            )
        ],
    )
    second = SimpleNamespace(
        finished=False,
        prompt_logprobs=None,
        outputs=[
            SimpleNamespace(
                text="lo",
                token_ids=[11, 12],
                finish_reason=None,
            )
        ],
    )

    class FakeDecoder:
        def __init__(self):
            self.inputs = []

        def _put_session_input(self, session, prompt_input, sampling_params):
            self.inputs.append((prompt_input, sampling_params))

    session = _VLLMLiveTextDecoderSession.__new__(_VLLMLiveTextDecoderSession)
    session._decoder = FakeDecoder()
    session._result_queue = queue.Queue()
    session._result_queue.put(first)
    session._result_queue.put(second)
    session._async_input_queue = None
    session._started = True
    session._closed = False
    session._last_returned_on_idle = False

    params = SimpleNamespace(output_kind=SimpleNamespace(name="DELTA"))
    result = session.decode(
        {"prompt_token_ids": [1]},
        params,
        timeout=1.0,
        idle_timeout=0.01,
    )

    assert result is not second
    assert result.outputs[0].text == "hello"
    assert result.outputs[0].token_ids == [10, 11, 12]
    assert result.prompt_logprobs == ["prompt-logprobs"]
    assert session._last_delta_output is True
    assert session._last_delta_token_count == 3


def test_qwen3_vllm_causal_processor_summarizes_generation_stats():
    processor = Qwen3VLLMCausalOnlineProcessor.__new__(Qwen3VLLMCausalOnlineProcessor)
    processor._generation_stats = [
        {
            "decoder_path": "vllm-text",
            "decoder_backend": "vllm-text",
            "prompt_tokens": 10,
            "vllm_cached_tokens": 0,
            "vllm_effective_prefill_tokens": 10,
            "audio_steps": 4,
            "prefill_positions": 10,
            "decoder_rebuilt": True,
            "stream_decode_wall_ms": 12.5,
            "align_wall_ms": 4.0,
            "decode_prefix_tokens": 0,
            "decode_prefix_chars": 0,
            "decode_prefix_overlap_words": 3,
        },
        {
            "decoder_path": "vllm-text",
            "decoder_backend": "vllm-text",
            "prompt_tokens": 14,
            "vllm_cached_tokens": 8,
            "vllm_effective_prefill_tokens": 6,
            "audio_steps": 8,
            "prefill_positions": 6,
            "vllm_live_prefix_keep_tokens": 8,
            "vllm_live_suffix_prefill_tokens": 6,
            "vllm_live_returned_on_idle": True,
            "decoder_rebuilt": False,
            "stream_decode_wall_ms": 9.25,
            "prompt_build_wall_ms": 1.5,
            "vllm_live_decode_wall_ms": 7.75,
            "vllm_live_session_wall_ms": 7.5,
            "vllm_live_time_to_first_output_ms": 3.0,
            "vllm_live_idle_tail_ms": 2.0,
            "align_wall_ms": 5.5,
            "audio_buffer_sec": 2.0,
            "aligned_words": 3,
            "vllm_live_output_events": 2,
            "vllm_live_delta_output": True,
            "vllm_live_delta_tokens": 3,
            "decode_prefix_tokens": 4,
            "decode_prefix_chars": 17,
            "decode_prefix_overlap_words": 3,
            "draft_unverifiable": 1,
        },
    ]

    summary = processor.generation_stats_summary()

    assert summary["decode_calls"] == 2
    assert summary["decoder_path"] == "vllm-text"
    assert summary["prompt_tokens_last"] == 14
    assert summary["prompt_tokens_max"] == 14
    assert summary["vllm_cached_tokens_last"] == 8
    assert summary["vllm_effective_prefill_tokens_last"] == 6
    assert summary["vllm_effective_prefill_tokens_max"] == 10
    assert summary["audio_steps_max"] == 8
    assert summary["prefill_positions_last"] == 6
    assert summary["prefill_positions_max"] == 10
    assert summary["vllm_live_prefix_keep_tokens_last"] == 8
    assert summary["vllm_live_suffix_prefill_tokens_last"] == 6
    assert summary["vllm_live_returned_on_idle_calls"] == 1
    assert summary["decoder_rebuilds"] == 1
    assert summary["decoder_backend"] == "vllm-text"
    assert summary["vllm_cached_token_ratio_last"] == 8 / 14
    assert summary["stream_decode_wall_ms_last"] == 9.25
    assert summary["stream_decode_wall_ms_max"] == 12.5
    assert summary["stream_decode_wall_ms_mean"] == pytest.approx(10.875)
    assert summary["prompt_build_wall_ms_last"] == 1.5
    assert summary["vllm_live_decode_wall_ms_last"] == 7.75
    assert summary["vllm_live_idle_tail_ms_last"] == 2.0
    assert summary["align_wall_ms_last"] == 5.5
    assert summary["audio_buffer_sec_last"] == 2.0
    assert summary["aligned_words_last"] == 3
    assert summary["vllm_live_output_events_last"] == 2
    assert summary["vllm_live_delta_output_calls"] == 1
    assert summary["vllm_live_delta_tokens_last"] == 3
    assert summary["decode_prefix_tokens_last"] == 4
    assert summary["decode_prefix_chars_last"] == 17
    assert summary["decode_prefix_overlap_words_last"] == 3
    assert summary["draft_unverifiable_last"] == 1


def test_qwen3_vllm_prefix_cache_stat_helpers_handle_missing_values():
    import torch

    assert _prompt_input_token_count({"prompt_embeds": torch.zeros(3, 2)}) == 3
    assert _prompt_input_token_count({"prompt_token_ids": [1, 2, 3, 4]}) == 4
    assert _prompt_input_token_count({}) == 0
    assert _request_output_num_cached_tokens(SimpleNamespace(num_cached_tokens=5)) == 5
    assert _request_output_num_cached_tokens(SimpleNamespace(num_cached_tokens=None)) is None
    assert _request_output_num_cached_tokens(None) is None


def test_qwen3_audio_feature_length_inverse_covers_streaming_blocks():
    for output_steps in range(1, 80):
        fake_length = _qwen3_audio_feature_length_for_output_steps(output_steps)
        input_lengths_leave = fake_length % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        recovered = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2
            + 1
            + (fake_length // 100) * 13
        )
        assert recovered == output_steps


def test_qwen3_vllm_causal_uses_multimodal_audio_embeds_when_patchable(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return SimpleNamespace(
                unk_token_id=-1,
                convert_tokens_to_ids=lambda token: 123,
                encode=lambda *args, **kwargs: [123],
            )

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(timestamp_token_id=456, timestamp_segment_time=0.02)

    monkeypatch.setattr(
        qwen_vllm,
        "_load_vllm_runtime",
        lambda: (FakeLLM, object, dict, FakeAutoConfig),
    )
    monkeypatch.setattr(qwen_vllm, "_patch_vllm_qwen3_asr_audio_embeds", lambda: True)
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_mrope_prompt_embeds",
        lambda: (_ for _ in ()).throw(AssertionError("prompt_embeds fallback used")),
    )
    init_calls = []
    monkeypatch.setattr(
        qwen_vllm.Qwen3VLLMASR,
        "_init_causal_audio_backend",
        lambda self, kwargs, *, dtype, keep_decoder=False: init_calls.append(
            {"dtype": dtype, "keep_decoder": keep_decoder}
        ),
    )

    asr = qwen_vllm.Qwen3VLLMASR(
        lan="en",
        model_size="0.6b",
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend="vllm",
        qwen3_vllm_causal_attn_implementation="sdpa",
    )

    assert asr.causal_decoder_backend == "vllm"
    assert asr.causal_attn_implementation == "sdpa"
    assert init_calls == [{"dtype": "auto", "keep_decoder": False}]
    assert asr.causal_prefix_cache_enabled is True
    assert asr.causal_audio_embeds_mm_enabled is True
    assert "enable_prompt_embeds" not in calls[0]
    assert calls[0]["enable_prefix_caching"] is True
    assert calls[0]["enable_mm_embeds"] is True
    assert calls[0]["compilation_config"] == {"cudagraph_mode": "NONE"}
    assert "enforce_eager" not in calls[0]
    assert "enforce_eager" not in calls[1]


def test_qwen3_vllm_causal_falls_back_to_prompt_embeds(monkeypatch):
    calls = []
    patched_prompt_embeds = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return SimpleNamespace(
                unk_token_id=-1,
                convert_tokens_to_ids=lambda token: 123,
                encode=lambda *args, **kwargs: [123],
            )

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(timestamp_token_id=456, timestamp_segment_time=0.02)

    monkeypatch.setattr(
        qwen_vllm,
        "_load_vllm_runtime",
        lambda: (FakeLLM, object, dict, FakeAutoConfig),
    )
    monkeypatch.setattr(qwen_vllm, "_patch_vllm_qwen3_asr_audio_embeds", lambda: False)
    monkeypatch.setattr(qwen_vllm, "_patch_vllm_mrope_prompt_embeds", lambda: patched_prompt_embeds.append(True))
    monkeypatch.setattr(qwen_vllm, "_vllm_prompt_embeds_prefix_cache_supported", lambda: True)
    monkeypatch.setattr(
        qwen_vllm.Qwen3VLLMASR,
        "_init_causal_audio_backend",
        lambda self, kwargs, *, dtype, keep_decoder=False: None,
    )

    asr = qwen_vllm.Qwen3VLLMASR(
        lan="en",
        model_size="0.6b",
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend="vllm",
    )

    assert asr.causal_audio_embeds_mm_enabled is False
    assert patched_prompt_embeds == [True]
    assert calls[0]["enable_prompt_embeds"] is True
    assert calls[0]["enable_prefix_caching"] is True
    assert calls[0]["enforce_eager"] is True
    assert "enforce_eager" not in calls[1]


def test_qwen3_vllm_causal_rolling_skips_asr_vllm_decoder(monkeypatch):
    calls = []
    init_calls = []
    fake_tokenizer = SimpleNamespace(
        unk_token_id=-1,
        convert_tokens_to_ids=lambda token: 123,
        encode=lambda *args, **kwargs: [123],
    )

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return fake_tokenizer

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(timestamp_token_id=456, timestamp_segment_time=0.02)

    def fake_init_causal_audio_backend(self, kwargs, *, dtype, keep_decoder=False):
        init_calls.append({"dtype": dtype, "keep_decoder": keep_decoder})
        self.causal_asr = SimpleNamespace(qwen_tokenizer=fake_tokenizer)

    monkeypatch.setattr(
        qwen_vllm,
        "_load_vllm_runtime",
        lambda: (FakeLLM, object, dict, FakeAutoConfig),
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_qwen3_asr_audio_embeds",
        lambda: (_ for _ in ()).throw(AssertionError("vLLM ASR patch used")),
    )
    monkeypatch.setattr(
        qwen_vllm.Qwen3VLLMASR,
        "_init_causal_audio_backend",
        fake_init_causal_audio_backend,
    )

    asr = qwen_vllm.Qwen3VLLMASR(
        lan="en",
        model_size="0.6b",
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend="rolling",
        qwen3_vllm_causal_attn_implementation="sdpa",
    )

    assert asr.causal_decoder_backend == "rolling"
    assert asr.causal_attn_implementation == "sdpa"
    assert asr.asr_llm is None
    assert asr.tokenizer is fake_tokenizer
    assert init_calls == [{"dtype": "auto", "keep_decoder": True}]
    assert len(calls) == 1
    assert calls[0]["runner"] == "pooling"


def test_qwen3_vllm_causal_append_kv_skips_asr_vllm_decoder(monkeypatch):
    calls = []
    init_calls = []
    fake_tokenizer = SimpleNamespace(
        unk_token_id=-1,
        convert_tokens_to_ids=lambda token: 123,
        encode=lambda *args, **kwargs: [123],
    )

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return fake_tokenizer

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(timestamp_token_id=456, timestamp_segment_time=0.02)

    def fake_init_causal_audio_backend(self, kwargs, *, dtype, keep_decoder=False):
        init_calls.append({"dtype": dtype, "keep_decoder": keep_decoder})
        self.causal_asr = SimpleNamespace(qwen_tokenizer=fake_tokenizer)

    monkeypatch.setattr(
        qwen_vllm,
        "_load_vllm_runtime",
        lambda: (FakeLLM, object, dict, FakeAutoConfig),
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_qwen3_asr_audio_embeds",
        lambda: (_ for _ in ()).throw(AssertionError("vLLM ASR patch used")),
    )
    monkeypatch.setattr(
        qwen_vllm.Qwen3VLLMASR,
        "_init_causal_audio_backend",
        fake_init_causal_audio_backend,
    )

    asr = qwen_vllm.Qwen3VLLMASR(
        lan="en",
        model_size="0.6b",
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend="append-kv",
    )

    assert asr.causal_decoder_backend == "append-kv"
    assert asr.asr_llm is None
    assert asr.tokenizer is fake_tokenizer
    assert init_calls == [{"dtype": "auto", "keep_decoder": True}]
    assert len(calls) == 1
    assert calls[0]["runner"] == "pooling"


def test_qwen3_vllm_causal_vllm_text_loads_exported_text_decoder(monkeypatch):
    calls = []
    init_calls = []
    fake_tokenizer = SimpleNamespace(
        unk_token_id=-1,
        convert_tokens_to_ids=lambda token: 123,
        encode=lambda *args, **kwargs: [123],
    )

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return fake_tokenizer

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(timestamp_token_id=456, timestamp_segment_time=0.02)

    def fake_init_causal_audio_backend(self, kwargs, *, dtype, keep_decoder=False):
        init_calls.append({"dtype": dtype, "keep_decoder": keep_decoder})
        self.causal_asr = SimpleNamespace(qwen_tokenizer=fake_tokenizer)

    monkeypatch.setattr(
        qwen_vllm,
        "_load_vllm_runtime",
        lambda: (FakeLLM, object, dict, FakeAutoConfig),
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_ensure_qwen3_asr_text_decoder_model",
        lambda model_path: "/tmp/qwen3-text-decoder",
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_qwen3_asr_audio_embeds",
        lambda: (_ for _ in ()).throw(AssertionError("multimodal ASR patch used")),
    )
    monkeypatch.setattr(
        qwen_vllm.Qwen3VLLMASR,
        "_init_causal_audio_backend",
        fake_init_causal_audio_backend,
    )

    asr = qwen_vllm.Qwen3VLLMASR(
        lan="en",
        model_size="0.6b",
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend="vllm-text",
        qwen3_vllm_cache_block_size=8,
    )

    assert asr.causal_decoder_backend == "vllm-text"
    assert asr.causal_text_decoder_model == "/tmp/qwen3-text-decoder"
    assert asr.asr_llm is not None
    assert asr.tokenizer is fake_tokenizer
    assert init_calls == [{"dtype": "auto", "keep_decoder": False}]
    assert len(calls) == 2
    assert calls[0]["model"] == "/tmp/qwen3-text-decoder"
    assert calls[0]["runner"] == "generate"
    assert calls[0]["enable_prompt_embeds"] is True
    assert calls[0]["enable_prefix_caching"] is True
    assert calls[0]["block_size"] == 8
    assert calls[1]["runner"] == "pooling"
    assert calls[1]["block_size"] == 8


def test_qwen3_vllm_causal_vllm_live_loads_live_text_decoder(monkeypatch):
    calls = []
    live_calls = []
    patch_calls = []
    init_calls = []
    fake_tokenizer = SimpleNamespace(
        unk_token_id=-1,
        convert_tokens_to_ids=lambda token: 123,
        encode=lambda *args, **kwargs: [123],
    )

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return fake_tokenizer

    class FakeLiveTextDecoder:
        def __init__(self, model_path, *, engine_kwargs):
            live_calls.append(
                {"model_path": model_path, "engine_kwargs": dict(engine_kwargs)}
            )

        def new_session(self):
            return SimpleNamespace(decode=lambda prompt, params: None)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(timestamp_token_id=456, timestamp_segment_time=0.02)

    def fake_init_causal_audio_backend(self, kwargs, *, dtype, keep_decoder=False):
        init_calls.append({"dtype": dtype, "keep_decoder": keep_decoder})
        self.causal_asr = SimpleNamespace(qwen_tokenizer=fake_tokenizer)

    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setattr(
        qwen_vllm,
        "_load_vllm_runtime",
        lambda: (FakeLLM, object, dict, FakeAutoConfig),
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_ensure_qwen3_asr_text_decoder_model",
        lambda model_path: "/tmp/qwen3-text-decoder",
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_mrope_prompt_embeds",
        lambda: patch_calls.append("mrope"),
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_live_prompt_embeds_streaming",
        lambda: patch_calls.append("live") or True,
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_VLLMLiveTextDecoder",
        FakeLiveTextDecoder,
    )
    monkeypatch.setattr(
        qwen_vllm,
        "_patch_vllm_qwen3_asr_audio_embeds",
        lambda: (_ for _ in ()).throw(AssertionError("multimodal ASR patch used")),
    )
    monkeypatch.setattr(
        qwen_vllm.Qwen3VLLMASR,
        "_init_causal_audio_backend",
        fake_init_causal_audio_backend,
    )

    asr = qwen_vllm.Qwen3VLLMASR(
        lan="en",
        model_size="0.6b",
        qwen3_vllm_audio_backend="causal",
        qwen3_vllm_causal_decoder_backend="vllm-live",
        qwen3_vllm_cache_block_size=8,
        qwen3_vllm_live_idle_timeout_ms=25,
    )

    assert asr.causal_decoder_backend == "vllm-live"
    assert asr.causal_text_decoder_model == "/tmp/qwen3-text-decoder"
    assert asr.causal_vllm_live_idle_timeout_sec == 0.025
    assert asr.asr_llm is not None
    assert asr.tokenizer is fake_tokenizer
    assert patch_calls == ["mrope", "live"]
    assert init_calls == [{"dtype": "auto", "keep_decoder": False}]
    assert live_calls == [
        {
            "model_path": "/tmp/qwen3-text-decoder",
            "engine_kwargs": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.45,
                "dtype": "auto",
                "block_size": 8,
                "compilation_config": {"cudagraph_mode": "NONE"},
            },
        }
    ]
    assert len(calls) == 1
    assert calls[0]["runner"] == "pooling"
    assert calls[0]["block_size"] == 8


def test_qwen3_vllm_aligner_helpers():
    assert _split_align_words("Hello, 世界!") == ["Hello", "世", "界"]
    assert _fix_timestamps([0, 3, 2, 4]) == [0.0, 3.0, 3.0, 4.0]
    assert _normalize_timestamp_segment_time(20.0) == 0.02
    assert _normalize_timestamp_segment_time(0.02) == 0.02


def test_qwen3_vllm_lazy_import_error_is_clear():
    try:
        _load_vllm_runtime()
    except ImportError as exc:
        assert "qwen3-vllm requires vLLM" in str(exc)


def test_qwen3_vllm_audio_backend_validation():
    assert _resolve_audio_backend({}) == "standard"
    assert _resolve_audio_backend({"qwen3_vllm_audio_backend": "causal"}) == "causal"
    try:
        _resolve_audio_backend({"qwen3_vllm_audio_backend": "nope"})
    except ValueError as exc:
        assert "must be 'standard' or 'causal'" in str(exc)
    else:
        raise AssertionError("invalid qwen3-vllm audio backend should fail")


def test_qwen3_vllm_causal_decoder_backend_validation():
    assert _resolve_causal_decoder_backend({}, "causal") == "vllm-text"
    assert (
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "append-kv"},
            "causal",
        )
        == "append-kv"
    )
    assert (
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "vllm"},
            "causal",
        )
        == "vllm"
    )
    assert (
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "auto"},
            "causal",
        )
        == "vllm-text"
    )
    assert (
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "vllm-text"},
            "causal",
        )
        == "vllm-text"
    )
    assert (
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "vllm-live"},
            "causal",
        )
        == "vllm-live"
    )
    assert (
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "rolling"},
            "standard",
        )
        == "vllm"
    )
    try:
        _resolve_causal_decoder_backend(
            {"qwen3_vllm_causal_decoder_backend": "nope"},
            "causal",
        )
    except ValueError as exc:
        assert "must be 'append-kv', 'rolling', 'vllm', 'vllm-live', or 'vllm-text'" in str(exc)
    else:
        raise AssertionError("invalid qwen3-vllm causal decoder backend should fail")


def test_vllm_live_cache_salt_metadata_roundtrip():
    tagged = _encode_vllm_live_cache_salt("stable-salt", prefix_keep_len=37)
    clean_salt, prefix_keep_len = _decode_vllm_live_cache_salt(tagged)

    assert clean_salt == "stable-salt"
    assert prefix_keep_len == 37
    assert _decode_vllm_live_cache_salt("stable-salt") == ("stable-salt", None)
    assert _decode_vllm_live_cache_salt(None) == (None, None)

    prompt = _vllm_live_prompt_input(
        {"cache_salt": "base", "x": 1},
        prefix_keep_len=4,
        prompt_head_len=2,
        audio_steps=3,
    )
    assert prompt["x"] == 1
    assert _decode_vllm_live_cache_salt(prompt["cache_salt"]) == ("base", 4)
    assert _decode_vllm_live_prompt_mask_metadata(prompt["cache_salt"]) == (2, 3)
    assert _vllm_live_prompt_is_token_ids_from_cache_salt(
        prompt["cache_salt"],
        prompt_len=8,
    ) == [True, True, False, False, False, True, True, True]
    assert _vllm_live_full_prompt_is_token_ids(
        [True] * 5,
        cache_salt=prompt["cache_salt"],
        prompt_len=8,
    ) == [True, True, False, False, False, True, True, True]
    assert _vllm_live_prompt_is_token_ids_from_repeated_placeholder(
        [10, 11, 7, 7, 7, 12, 13, 14],
        prompt_len=8,
        compact_mask_len=5,
    ) == [True, True, False, False, False, True, True, True]
    assert _vllm_live_full_prompt_is_token_ids(
        [True] * 5,
        cache_salt="stable-salt",
        prompt_len=8,
        prompt_token_ids=[10, 11, 7, 7, 7, 12, 13, 14],
    ) == [True, True, False, False, False, True, True, True]
    prompt_embeds = np.zeros((8, 4), dtype=np.float32)
    prompt_embeds[2:5] = 1.0
    assert _vllm_live_prompt_is_token_ids_from_prompt_embeds(prompt_embeds) == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert _vllm_live_full_prompt_is_token_ids(
        [True] * 5,
        cache_salt="stable-salt",
        prompt_len=8,
        prompt_embeds=prompt_embeds,
    ) == [True, True, False, False, False, True, True, True]


def test_vllm_live_compact_prompt_embeds_keeps_absolute_suffix(monkeypatch):
    torch = pytest.importorskip("torch")

    values = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    compact = _VLLMLiveCompactPromptEmbeds(
        values[6:].contiguous(),
        offset=6,
        full_shape=(10, 4),
    )

    assert compact.shape == (10, 4)
    assert torch.equal(compact[6:9], values[6:9])
    assert torch.equal(compact[5:8][1:], values[6:8])
    assert torch.equal(compact[2:4], torch.zeros((2, 4)))
    assert np.asarray(compact).shape == (10, 4)

    salt = _encode_vllm_live_cache_salt(
        "stable-salt",
        prefix_keep_len=8,
        prompt_head_len=2,
        audio_steps=6,
    )
    monkeypatch.setenv("WLK_VLLM_LIVE_COMPACT_LOOKBACK_TOKENS", "2")
    monkeypatch.setenv("WLK_VLLM_LIVE_COMPACT_PROMPT_EMBEDS", "1")
    compressed = _compact_vllm_live_prompt_embeds(values, salt)

    assert isinstance(compressed, _VLLMLiveCompactPromptEmbeds)
    assert compressed.offset == 6
    assert compressed.shape == (10, 4)
    assert torch.equal(compressed[8:10], values[8:10])


def test_vllm_live_request_metadata_restores_prompt_token_ids():
    cache_salt = _encode_vllm_live_cache_salt("stable-salt", prefix_keep_len=0)
    request = SimpleNamespace(
        prompt_embeds=object(),
        prompt_token_ids=None,
        prompt_is_token_ids=None,
    )
    _apply_vllm_live_request_metadata_from_prompt(
        request,
        {
            "prompt_embeds": object(),
            "prompt_token_ids": [1, 2, 3],
            "prompt_is_token_ids": [True, False, True],
            "cache_salt": cache_salt,
        },
    )

    assert request.prompt_token_ids == [1, 2, 3]
    assert request.prompt_is_token_ids == [True, False, True]
    assert request._wlk_live_prompt_embeds_streaming is True


def test_vllm_live_request_metadata_compacts_prompt_embeds(monkeypatch):
    torch = pytest.importorskip("torch")

    prompt_embeds = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    salt = _encode_vllm_live_cache_salt(
        "stable-salt",
        prefix_keep_len=8,
        prompt_head_len=2,
        audio_steps=6,
    )
    request = SimpleNamespace(
        prompt_embeds=prompt_embeds,
        prompt_token_ids=None,
        prompt_is_token_ids=None,
    )
    monkeypatch.setenv("WLK_VLLM_LIVE_COMPACT_LOOKBACK_TOKENS", "2")
    monkeypatch.setenv("WLK_VLLM_LIVE_COMPACT_PROMPT_EMBEDS", "1")

    _apply_vllm_live_request_metadata_from_prompt(
        request,
        {
            "prompt_embeds": prompt_embeds,
            "prompt_token_ids": list(range(10)),
            "prompt_is_token_ids": [True] * 2 + [False] * 6 + [True] * 2,
            "cache_salt": salt,
        },
    )

    assert request.prompt_token_ids == list(range(10))
    assert request.prompt_is_token_ids == [True] * 2 + [False] * 6 + [True] * 2
    assert isinstance(request.prompt_embeds, _VLLMLiveCompactPromptEmbeds)
    assert request.prompt_embeds.offset == 6
    assert torch.equal(request.prompt_embeds[8:10], prompt_embeds[8:10])


def test_vllm_live_output_processor_patch_keeps_prompt_embeds_on_updates():
    from collections import deque

    torch = pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    assert vllm is not None
    if not _patch_vllm_live_prompt_embeds_streaming():
        pytest.skip("installed vLLM does not expose compatible V1 streaming hooks")

    from vllm.v1.engine import output_processor as output_processor_module

    prompt_embeds = torch.zeros((3, 4), dtype=torch.float32)
    request = SimpleNamespace(
        resumable=True,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=prompt_embeds,
        prompt_is_token_ids=[True, False, True],
        arrival_time=7.0,
        cache_salt=_encode_vllm_live_cache_salt("stable-salt", prefix_keep_len=2),
        sampling_params=SimpleNamespace(
            output_kind="cumulative",
            max_tokens=16,
            top_p=1.0,
            n=1,
            temperature=0.0,
            detokenize=False,
            logprobs=None,
            num_logprobs=None,
            prompt_logprobs=None,
            flat_logprobs=False,
        ),
    )

    class FakeState(SimpleNamespace):
        def apply_streaming_update(self, update):
            self.prompt_token_ids = list(update.prompt_token_ids or [])
            self.prompt_len = len(self.prompt_token_ids)
            self.is_prefilling = True

    immediate_state = FakeState(
        input_chunk_queue=None,
        prompt_token_ids=[0],
        prompt_embeds=None,
        prompt_len=1,
        stats=SimpleNamespace(arrival_time=0.0),
        is_prefilling=False,
        sent_tokens_offset=4,
        routed_experts_chunks=[object()],
        num_cached_tokens=7,
        detokenizer=SimpleNamespace(token_ids=[9]),
        logprobs_processor=object(),
    )
    output_processor_module.OutputProcessor._update_streaming_request_state(
        SimpleNamespace(),
        immediate_state,
        request,
        None,
    )

    assert immediate_state.prompt_token_ids == [1, 2, 3]
    assert immediate_state.prompt_embeds is prompt_embeds
    assert immediate_state.prompt_len == 3
    assert immediate_state.is_prefilling is True
    assert immediate_state.stats.arrival_time == 7.0
    assert immediate_state.sent_tokens_offset == 0
    assert immediate_state.routed_experts_chunks == []
    assert immediate_state.num_cached_tokens == 0
    assert immediate_state.detokenizer.output_token_ids == []

    queued_state = FakeState(
        input_chunk_queue=deque(),
        prompt_token_ids=[0],
        prompt_embeds=None,
        prompt_len=1,
        stats=None,
        is_prefilling=False,
        sent_tokens_offset=4,
        routed_experts_chunks=[object()],
        num_cached_tokens=7,
        detokenizer=SimpleNamespace(token_ids=[9]),
        logprobs_processor=object(),
    )
    output_processor_module.OutputProcessor._update_streaming_request_state(
        SimpleNamespace(),
        queued_state,
        request,
        None,
    )
    update = queued_state.input_chunk_queue[-1]
    assert update.prompt_embeds is prompt_embeds
    assert update.prompt_is_token_ids == [True, False, True]
    assert _decode_vllm_live_cache_salt(update.cache_salt) == ("stable-salt", 2)
    assert update._wlk_detokenizer.output_token_ids == []
    assert update._wlk_logprobs_processor.logprobs is None


def test_vllm_live_scheduler_patch_replaces_prompt_and_keeps_kv_prefix():
    torch = pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    assert vllm is not None
    if not _patch_vllm_live_prompt_embeds_streaming():
        pytest.skip("installed vLLM does not expose compatible V1 streaming hooks")

    from vllm.v1.core.sched import scheduler as sched_module
    from vllm.v1.request import StreamingUpdate

    class FakeSession(SimpleNamespace):
        def update_block_hashes(self):
            self.update_hash_calls += 1
            self.block_hashes.append(f"hash-{len(self._all_token_ids)}")

        def record_event(self, event):
            self.recorded_events.append(event)

    prompt_embeds = torch.zeros((8, 4), dtype=torch.float32)
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=list(range(100, 108)),
        max_tokens=16,
        arrival_time=123.0,
        sampling_params="sampling",
    )
    update.prompt_embeds = prompt_embeds
    update.prompt_is_token_ids = [True, True, False, False, False, True, True, True]
    update.cache_salt = _encode_vllm_live_cache_salt(
        "stable-salt",
        prefix_keep_len=5,
    )

    session = FakeSession(
        prompt_embeds=torch.ones((7, 4), dtype=torch.float32),
        prompt_token_ids=list(range(7)),
        prompt_is_token_ids=[True] * 7,
        cache_salt="stable-salt",
        mm_features=["old-mm"],
        _all_token_ids=list(range(7)) + [201, 202],
        _output_token_ids=[201, 202],
        spec_token_ids=[301],
        num_output_placeholders=3,
        discard_latest_async_tokens=True,
        num_computed_tokens=7,
        num_prompt_tokens=7,
        block_hashes=["old-block-0", "old-block-1", "old-block-2"],
        _prompt_embeds_per_block_hashes={(1, 2): b"old"},
        arrival_time=0.0,
        sampling_params=None,
        status=sched_module.RequestStatus.WAITING_FOR_STREAMING_REQ,
        update_hash_calls=0,
        recorded_events=[],
    )
    scheduler = SimpleNamespace(
        block_size=4,
        log_stats=False,
        num_waiting_for_streaming_input=1,
    )

    sched_module.Scheduler._update_request_as_session(scheduler, session, update)

    assert session.prompt_embeds is prompt_embeds
    assert session.prompt_token_ids == list(range(100, 108))
    assert session.prompt_is_token_ids == update.prompt_is_token_ids
    assert session._all_token_ids == list(range(100, 108))
    assert session._output_token_ids == []
    assert session.spec_token_ids == []
    assert session.num_output_placeholders == 0
    assert session.discard_latest_async_tokens is True
    assert session.num_computed_tokens == 5
    assert session.num_prompt_tokens == 8
    assert session.cache_salt == "stable-salt"
    assert session.mm_features == []
    assert session.arrival_time == 123.0
    assert session.sampling_params == "sampling"
    assert session.status == sched_module.RequestStatus.WAITING
    assert session._prompt_embeds_per_block_hashes == {}
    assert session.block_hashes == ["old-block-0", "hash-8"]
    assert session.update_hash_calls == 1
    assert scheduler.num_waiting_for_streaming_input == 0


def test_vllm_live_scheduler_patch_discards_stale_async_output():
    vllm = pytest.importorskip("vllm")
    assert vllm is not None
    if not _patch_vllm_live_prompt_embeds_streaming():
        pytest.skip("installed vLLM does not expose compatible V1 streaming hooks")

    from vllm.v1.core.sched import async_scheduler as async_sched_module

    request = SimpleNamespace(
        discard_latest_async_tokens=True,
        num_output_placeholders=0,
    )

    new_token_ids, stopped = async_sched_module.AsyncScheduler._update_request_with_output(
        SimpleNamespace(),
        request,
        [123],
    )

    assert new_token_ids == []
    assert stopped is False
    assert request.discard_latest_async_tokens is False
    assert request.num_output_placeholders == 0


def test_vllm_live_async_scheduler_patch_tolerates_excess_tokens():
    vllm = pytest.importorskip("vllm")
    assert vllm is not None
    if not _patch_vllm_live_prompt_embeds_streaming():
        pytest.skip("installed vLLM does not expose compatible V1 streaming hooks")

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched import async_scheduler as async_sched_module
    from vllm.v1.request import Request, RequestStatus

    class FakeKVCacheManager:
        def __init__(self):
            self.cached = []

        def cache_blocks(self, request, num_computed_tokens):
            self.cached.append((request.request_id, num_computed_tokens))

    request = Request(
        request_id="wlk-live",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        cache_salt=_encode_vllm_live_cache_salt("stable-salt", prefix_keep_len=0),
    )
    request._wlk_live_prompt_embeds_streaming = True
    request.status = RequestStatus.RUNNING
    request.num_output_placeholders = 1
    request.num_computed_tokens = 4
    scheduler = SimpleNamespace(max_model_len=128, kv_cache_manager=FakeKVCacheManager())

    new_token_ids, stopped = async_sched_module.AsyncScheduler._update_request_with_output(
        scheduler,
        request,
        [123, 124],
    )

    assert new_token_ids == [123, 124]
    assert stopped is False
    assert request.num_output_placeholders == 0
    assert request.output_token_ids[-2:] == [123, 124]
    assert scheduler.kv_cache_manager.cached == [("wlk-live", 4)]
