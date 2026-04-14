import pytest

from qwen3_asr_causal.vllm import (
    DEFAULT_QWEN3_VLLM_CAUSAL_TOWER,
    DEFAULT_QWEN3_VLLM_MODEL,
    _decode_vllm_live_cache_salt,
    _encode_vllm_live_cache_salt,
    _resolve_causal_decoder_backend,
)


def test_vllm_defaults_point_to_public_causal_model():
    assert DEFAULT_QWEN3_VLLM_MODEL == "Qwen/Qwen3-ASR-0.6B"
    assert DEFAULT_QWEN3_VLLM_CAUSAL_TOWER == "qfuxa/qwen3-asr-0.6b-streaming"


def test_vllm_text_is_causal_default():
    assert _resolve_causal_decoder_backend({}, "causal") == "vllm-text"
    assert (
        _resolve_causal_decoder_backend({"qwen3_vllm_causal_decoder_backend": "vllm-text"}, "causal")
        == "vllm-text"
    )


def test_invalid_causal_decoder_backend_fails():
    with pytest.raises(ValueError):
        _resolve_causal_decoder_backend({"qwen3_vllm_causal_decoder_backend": "bad"}, "causal")


def test_live_cache_salt_roundtrip():
    encoded = _encode_vllm_live_cache_salt("stream", 12)

    assert _decode_vllm_live_cache_salt(encoded) == ("stream", 12)
