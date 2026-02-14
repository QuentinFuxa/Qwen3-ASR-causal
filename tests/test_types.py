from qwen3_asr_causal.types import ASRToken, Transcript


def test_transcript_from_tokens():
    tokens = [
        ASRToken(start=0.0, end=0.2, text="hello"),
        ASRToken(start=0.2, end=0.5, text="world"),
    ]

    transcript = Transcript.from_tokens(tokens, offset=1.0)

    assert transcript.start == 1.0
    assert transcript.end == 1.5
    assert transcript.text == "hello world"


def test_token_offset_preserves_none_timestamps():
    token = ASRToken(start=None, end=1.0, text="x")

    shifted = token.with_offset(2.0)

    assert shifted.start is None
    assert shifted.end == 3.0
    assert shifted.text == "x"
