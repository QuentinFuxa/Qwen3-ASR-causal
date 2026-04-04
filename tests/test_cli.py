from qwen3_asr_causal.cli import build_parser


def test_cli_accepts_hf_defaults():
    args = build_parser().parse_args(["transcribe", "audio.wav", "--backend", "hf", "--language", "en"])

    assert args.backend == "hf"
    assert args.model == "Qwen/Qwen3-ASR-0.6B"
    assert args.tower == "qfuxa/qwen3-asr-0.6b-streaming"


def test_cli_accepts_vllm_live_decoder():
    args = build_parser().parse_args(
        [
            "transcribe",
            "audio.wav",
            "--backend",
            "vllm",
            "--decoder-backend",
            "vllm-live",
            "--language",
            "en",
        ]
    )

    assert args.backend == "vllm"
    assert args.decoder_backend == "vllm-live"
