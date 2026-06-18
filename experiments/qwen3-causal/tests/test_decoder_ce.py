"""D2 objective tests: LoRA no-op init, CE label layout, gradient isolation."""

import random

import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.decoder_ce import (  # noqa: E402
    block_aligned_prefix_pair_targets,
    block_aligned_streaming_prefix_target,
    build_ce_inputs,
    ce_forward,
    consistency_kl_forward,
    sample_streaming_prefix_target,
    stable_text_from_word_alignments,
)
from qwen3_streaming.lora import (  # noqa: E402
    DECODER_LORA_TARGETS,
    add_lora_to_linear_modules,
    lora_parameters,
    lora_state_dict,
)

D_MODEL = 32


class FakeTokenizer:
    eos_token_id = 5

    def encode(self, text, add_special_tokens=False):
        if "<|audio_pad|>" in text:
            # prompt template: [system, audio_start, AUDIO, audio_end, lang]
            return [10, 11, 7, 12, 13]
        return [20 + (ord(c) % 4) for c in text.replace(" ", "")][:8]

    def convert_tokens_to_ids(self, token):
        return 7 if token == "<|audio_pad|>" else -1


class FakeTextModel(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden=D_MODEL):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden)
        self.layers = torch.nn.ModuleList(
            [torch.nn.ModuleDict({
                "q_proj": torch.nn.Linear(hidden, hidden),
                "down_proj": torch.nn.Linear(hidden, hidden),
            }) for _ in range(2)]
        )
        self.norm = torch.nn.Identity()

    def forward(self, inputs_embeds, **_kwargs):
        h = inputs_embeds
        for layer in self.layers:
            h = h + layer["down_proj"](torch.tanh(layer["q_proj"](h)))

        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return Output(h)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = FakeTextModel()
        self.embed_tokens = self.text_model.embed_tokens
        self.lm_head = torch.nn.Linear(D_MODEL, 32, bias=False)


def test_build_ce_inputs_label_layout():
    tok = FakeTokenizer()
    prompt_ids, target_ids, labels = build_ce_inputs(
        tok,
        audio_steps=3,
        language="English",
        target_text="hello world",
        audio_placeholder_token_id=7,
    )
    assert prompt_ids.count(7) == 3  # placeholder expanded to audio steps
    assert target_ids[-1] == tok.eos_token_id
    assert labels[: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids) :] == target_ids


def test_lora_zero_init_is_noop_and_isolated():
    torch.manual_seed(0)
    model = FakeModel().eval()
    x = torch.randn(1, 4, D_MODEL)
    target = [1, 2, 3]
    prompt = [10, 7, 7, 12]

    with torch.no_grad():
        loss_before, _ = ce_forward(
            model, x[:, :2, :], prompt_ids=prompt, target_ids=target,
            audio_placeholder_token_id=7,
        )
    wrapped = add_lora_to_linear_modules(
        model.text_model, target_names=DECODER_LORA_TARGETS, rank=4, alpha=8.0
    )
    assert wrapped, "no decoder linears wrapped"
    with torch.no_grad():
        loss_after, _ = ce_forward(
            model, x[:, :2, :], prompt_ids=prompt, target_ids=target,
            audio_placeholder_token_id=7,
        )
    torch.testing.assert_close(loss_after, loss_before)  # zero-init no-op

    # Gradients flow into LoRA params only.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in lora_parameters(model.text_model):
        param.requires_grad_(True)
    loss, stats = ce_forward(
        model, x[:, :2, :], prompt_ids=prompt, target_ids=target,
        audio_placeholder_token_id=7,
    )
    loss.backward()
    lora_grads = [p.grad for p in lora_parameters(model.text_model)]
    assert any(g is not None and g.abs().sum() > 0 for g in lora_grads)
    assert model.lm_head.weight.grad is None
    assert 0.0 <= stats["token_accuracy"] <= 1.0

    sd = lora_state_dict(model.text_model)
    assert sd and all(".lora_" in k for k in sd)


def test_ce_forward_supports_prefix_eos_weight():
    torch.manual_seed(1)
    model = FakeModel().eval()
    x = torch.randn(1, 3, D_MODEL)
    prompt = [10, 7, 7, 7, 12]
    target = [1, 2, FakeTokenizer.eos_token_id]

    loss, stats = ce_forward(
        model,
        x,
        prompt_ids=prompt,
        target_ids=target,
        audio_placeholder_token_id=7,
        eos_token_id=FakeTokenizer.eos_token_id,
        eos_weight=4.0,
    )

    assert torch.isfinite(loss)
    assert stats["eos_weight"] == 4.0


def test_ce_forward_rejects_non_positive_eos_weight():
    model = FakeModel().eval()
    with pytest.raises(ValueError, match="eos_weight"):
        ce_forward(
            model,
            torch.randn(1, 1, D_MODEL),
            prompt_ids=[10, 7, 12],
            target_ids=[FakeTokenizer.eos_token_id],
            audio_placeholder_token_id=7,
            eos_token_id=FakeTokenizer.eos_token_id,
            eos_weight=0.0,
        )


def test_consistency_kl_forward_is_zero_for_identical_logits():
    torch.manual_seed(3)
    model = FakeModel().eval()
    x = torch.randn(1, 3, D_MODEL)
    prompt = [10, 7, 7, 7, 12]
    target = [1, 2, 3]

    loss, stats = consistency_kl_forward(
        model,
        x,
        x,
        old_prompt_ids=prompt,
        new_prompt_ids=prompt,
        common_target_ids=target,
        audio_placeholder_token_id=7,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    assert stats["consistency_tokens"] == len(target)


def test_consistency_kl_forward_rejects_bad_inputs():
    model = FakeModel().eval()
    with pytest.raises(ValueError, match="common_target_ids"):
        consistency_kl_forward(
            model,
            torch.randn(1, 1, D_MODEL),
            torch.randn(1, 1, D_MODEL),
            old_prompt_ids=[10, 7, 12],
            new_prompt_ids=[10, 7, 12],
            common_target_ids=[],
            audio_placeholder_token_id=7,
        )


def test_stable_text_from_word_alignments_uses_word_end_times():
    words = [
        {"text": "one", "start_sec": 0.0, "end_sec": 0.3},
        {"text": "two", "start_sec": 0.4, "end_sec": 0.8},
        {"text": "three", "start_sec": 0.9, "end_sec": 1.4},
    ]
    assert stable_text_from_word_alignments(words, stable_until_sec=0.8) == "one two"


def test_sample_streaming_prefix_target_holds_back_unstable_words():
    row = {
        "duration_sec": 4.0,
        "word_alignments": [
            {"text": "one", "start_sec": 0.0, "end_sec": 0.5},
            {"text": "two", "start_sec": 1.0, "end_sec": 1.5},
            {"text": "three", "start_sec": 2.0, "end_sec": 2.5},
            {"text": "four", "start_sec": 3.0, "end_sec": 3.5},
        ],
    }
    prefix_end, text = sample_streaming_prefix_target(
        row,
        rng=random.Random(2),
        min_prefix_sec=2.0,
        right_context_sec=0.5,
        min_target_words=2,
    )
    assert 2.0 <= prefix_end <= 4.0
    assert text.split()[-1] != "four"


def test_block_aligned_streaming_prefix_target_recomputes_text_after_rounding():
    row = {
        "word_alignments": [
            {"text": "one", "start_sec": 0.0, "end_sec": 0.8},
            {"text": "two", "start_sec": 1.0, "end_sec": 1.9},
            {"text": "future", "start_sec": 2.0, "end_sec": 2.2},
        ],
    }

    result = block_aligned_streaming_prefix_target(
        row,
        prefix_end_sec=2.3,
        available_frames=230,
        block_frames=48,
        right_context_sec=0.25,
        min_target_words=1,
    )

    assert result == (192, "one")


def test_block_aligned_streaming_prefix_target_rejects_too_short_prefix():
    row = {"word_alignments": [{"text": "one", "start_sec": 0.0, "end_sec": 0.4}]}

    result = block_aligned_streaming_prefix_target(
        row,
        prefix_end_sec=0.47,
        available_frames=47,
        block_frames=48,
        right_context_sec=0.25,
    )

    assert result is None


def test_block_aligned_prefix_pair_targets_returns_stable_common_prefix():
    row = {
        "word_alignments": [
            {"text": "one", "start_sec": 0.0, "end_sec": 0.4},
            {"text": "two", "start_sec": 0.5, "end_sec": 0.9},
            {"text": "three", "start_sec": 1.0, "end_sec": 1.4},
            {"text": "four", "start_sec": 1.5, "end_sec": 1.9},
        ],
    }

    result = block_aligned_prefix_pair_targets(
        row,
        prefix_end_sec=2.3,
        available_frames=230,
        block_frames=48,
        pair_gap_sec=0.96,
        right_context_sec=0.25,
        min_target_words=2,
        min_common_words=1,
    )

    assert result == (96, "one", 192, "one two three")


def test_block_aligned_prefix_pair_targets_requires_common_words():
    row = {
        "word_alignments": [
            {"text": "late", "start_sec": 1.0, "end_sec": 1.3},
        ],
    }

    result = block_aligned_prefix_pair_targets(
        row,
        prefix_end_sec=1.5,
        available_frames=150,
        block_frames=48,
        pair_gap_sec=0.96,
        right_context_sec=0.25,
        min_common_words=1,
    )

    assert result is None
