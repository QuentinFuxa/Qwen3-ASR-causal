import numpy as np
import pytest


def _processor_for(text: str, holdback_words: int = 2):
    from qwen3_asr_causal.metal import Qwen3VLLMMetalOnlineProcessor

    class FakeASR:
        sep = ""
        trim_sentence_buffer = False
        min_chunk_size = 0.0

        def __init__(self):
            self.holdback_words = holdback_words

        def transcribe_text(self, audio):
            return text

    processor = Qwen3VLLMMetalOnlineProcessor(FakeASR())
    processor.insert_audio_chunk(np.zeros(16000, dtype=np.float32), 1.0)
    return processor


def _texts(tokens):
    return [token.text for token in tokens]


def test_qwen3_vllm_metal_empty_text_commits_nothing():
    processor = _processor_for("")

    tokens, _ = processor.process_iter()

    assert tokens == []
    assert processor.get_buffer().text == ""


def test_qwen3_vllm_metal_two_words_are_buffered():
    processor = _processor_for("one two")

    tokens, _ = processor.process_iter()

    assert tokens == []
    assert processor.get_buffer().text == "one two"


def test_qwen3_vllm_metal_three_words_commits_one_and_buffers_two():
    processor = _processor_for("one two three")

    tokens, _ = processor.process_iter()

    assert _texts(tokens) == ["one"]
    assert processor.get_buffer().text == "two three"


def test_qwen3_vllm_metal_start_silence_flushes_buffered_words():
    processor = _processor_for("one two three")
    tokens, _ = processor.process_iter()
    assert _texts(tokens) == ["one"]

    tokens, _ = processor.start_silence()

    assert _texts(tokens) == [" two", " three"]
    assert processor.get_buffer().text == ""
    assert len(processor.audio_buffer) == 0


def test_qwen3_vllm_metal_finish_flushes_buffered_words():
    processor = _processor_for("one two three")
    tokens, _ = processor.process_iter()
    assert _texts(tokens) == ["one"]

    tokens, _ = processor.finish()

    assert _texts(tokens) == [" two", " three"]
    assert processor.get_buffer().text == ""


def test_qwen3_vllm_metal_holdback_words_is_configurable():
    processor = _processor_for("one two three", holdback_words=1)

    tokens, _ = processor.process_iter()

    assert _texts(tokens) == ["one", " two"]
    assert processor.get_buffer().text == "three"


def test_qwen3_vllm_metal_model_path_aliases():
    from qwen3_asr_causal.metal import (
        DEFAULT_QWEN3_VLLM_METAL_MODEL,
        QWEN3_VLLM_METAL_1_7B_MODEL,
        _resolve_model_path,
    )

    assert _resolve_model_path({}) == DEFAULT_QWEN3_VLLM_METAL_MODEL
    assert _resolve_model_path({"model_size": "0.6b"}) == DEFAULT_QWEN3_VLLM_METAL_MODEL
    assert _resolve_model_path({"model_size": "qwen3-asr-0.6b"}) == DEFAULT_QWEN3_VLLM_METAL_MODEL
    assert _resolve_model_path({"model_size": "1.7b"}) == QWEN3_VLLM_METAL_1_7B_MODEL
    assert _resolve_model_path({"model_size": "qwen3-asr-1.7b"}) == QWEN3_VLLM_METAL_1_7B_MODEL
    assert _resolve_model_path({"model_size": "Qwen/Qwen3-ASR-0.6B"}) == "Qwen/Qwen3-ASR-0.6B"
    assert _resolve_model_path({"model_size": "./local-model"}) == "./local-model"
    assert _resolve_model_path({"model_path": "/models/qwen"}) == "/models/qwen"
    assert _resolve_model_path({"model_dir": "/models/qwen-dir"}) == "/models/qwen-dir"


def test_qwen3_vllm_metal_rejects_unsupported_model_alias():
    from qwen3_asr_causal.metal import _resolve_model_path

    with pytest.raises(ValueError, match="supports Qwen3-ASR 0.6B and 1.7B"):
        _resolve_model_path({"model_size": "large-v3"})


def test_qwen3_vllm_metal_resolves_shared_vllm_dtype_names():
    from qwen3_asr_causal.metal import _resolve_mlx_dtype

    class FakeMX:
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "auto"}) == "float16"
    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "float16"}) == "float16"
    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "bf16"}) == "bfloat16"
    assert _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "float32"}) == "float32"
    assert _resolve_mlx_dtype(FakeMX, {"dtype": "custom", "vllm_dtype": "bf16"}) == "custom"

    with pytest.raises(ValueError, match="vllm_dtype must be one of"):
        _resolve_mlx_dtype(FakeMX, {"vllm_dtype": "int8"})


def test_qwen3_vllm_metal_dependency_errors_are_specific():
    from qwen3_asr_causal.metal import _missing_dependency_error

    assert "vLLM CPU package is missing" in str(
        _missing_dependency_error(ImportError("missing", name="vllm"))
    )
    assert "requires vllm-metal with STT support" in str(
        _missing_dependency_error(ImportError("missing", name="vllm_metal.stt"))
    )
    assert "requires MLX" in str(
        _missing_dependency_error(ImportError("missing", name="mlx.core"))
    )


def test_qwen3_vllm_metal_rejects_unsupported_platform(monkeypatch):
    import qwen3_asr_causal.metal as qwen_metal

    monkeypatch.setattr(qwen_metal.platform, "system", lambda: "Linux")
    monkeypatch.setattr(qwen_metal.platform, "machine", lambda: "x86_64")

    with pytest.raises(ImportError, match="Apple Silicon"):
        qwen_metal._ensure_supported_platform()


def test_qwen3_vllm_metal_decodes_without_special_tokens():
    from qwen3_asr_causal.metal import Qwen3VLLMMetalASR

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            assert tokens == [1, 2, 3]
            assert skip_special_tokens is True
            return "hello"

    asr = object.__new__(Qwen3VLLMMetalASR)
    asr.tokenizer = FakeTokenizer()
    asr._post_process_output = lambda text: text

    assert asr._decode_output_tokens([1, 2, 3]) == "hello"


def test_qwen3_vllm_metal_rejects_unknown_audio_backend():
    from qwen3_asr_causal.metal import _resolve_audio_backend

    with pytest.raises(ValueError, match="standard.*causal"):
        _resolve_audio_backend({"qwen3_vllm_metal_audio_backend": "windowed"})


def test_qwen3_vllm_metal_causal_trims_sentence_and_resets_audio_cache():
    from qwen3_asr_causal.metal import Qwen3VLLMMetalCausalOnlineProcessor

    class FakeASR:
        sep = ""
        holdback_words = 2
        trim_sentence_buffer = True
        min_chunk_size = 0.0
        audio_backend = "causal"

        def __init__(self):
            self.state_resets = 0
            self.feature_resets = 0
            self.decoder_resets = 0
            self.seen_decoder_states = []

        def new_causal_state(self):
            self.state_resets += 1
            return {"state": self.state_resets}

        def empty_causal_features(self):
            self.feature_resets += 1
            return {"features": self.feature_resets}

        def new_causal_decoder_state(self):
            self.decoder_resets += 1
            return {"decoder": self.decoder_resets}

        def causal_decode_from_audio(
            self,
            audio,
            *,
            mel_frames_encoded,
            audio_features,
            state,
            decoder_state,
            flush=False,
        ):
            self.seen_decoder_states.append(decoder_state)
            return "one. two three", 192, audio_features, state, decoder_state

    asr = FakeASR()
    processor = Qwen3VLLMMetalCausalOnlineProcessor(asr)
    processor.insert_audio_chunk(np.zeros(16000, dtype=np.float32), 1.0)

    tokens, _ = processor.process_iter()

    assert _texts(tokens) == ["one."]
    assert asr.state_resets == 2
    assert asr.feature_resets == 2
    assert asr.decoder_resets == 2
    assert asr.seen_decoder_states == [{"decoder": 1}]
    assert processor._mel_frames_encoded == 0
    assert processor.get_buffer().text == "two three"


def test_qwen3_vllm_metal_prompt_parts_match_expanded_prompt():
    from qwen3_asr_causal.metal import Qwen3VLLMMetalASR

    class FakeTokenizer:
        mapping = {
            "<|im_start|>": [10],
            "system\n": [20],
            "<|im_end|>\n": [30],
            "user\n": [40],
            "<|audio_start|>": [50],
            "<|audio_end|>": [60],
            "assistant\n": [70],
        }

        def encode(self, text, add_special_tokens=False):
            return list(self.mapping[text])

    class FakeConfig:
        audio_token_id = 99

    class FakeModel:
        config = FakeConfig()

    asr = object.__new__(Qwen3VLLMMetalASR)
    asr.tokenizer = FakeTokenizer()
    asr.model = FakeModel()

    head, tail = asr._build_prompt_token_parts()

    assert asr._build_prompt_token_ids(3) == head + [99, 99, 99] + tail


def test_qwen3_vllm_metal_decoder_cache_helpers_crop_prefix():
    import numpy as np

    from qwen3_asr_causal.metal import (
        _crop_mlx_decoder_cache,
        _mlx_decoder_cache_seq_length,
    )

    key = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    value = key + 1000

    cropped = _crop_mlx_decoder_cache([(key, value)], 3)

    assert _mlx_decoder_cache_seq_length([(key, value)]) == 5
    assert _mlx_decoder_cache_seq_length(cropped) == 3
    assert cropped[0][0].shape == (2, 3, 3, 7)
    assert np.array_equal(cropped[0][0], key[:, :, :3, :])
    assert np.array_equal(cropped[0][1], value[:, :, :3, :])
