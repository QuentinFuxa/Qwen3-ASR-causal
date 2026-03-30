from qwen3_asr_causal.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
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
    )
