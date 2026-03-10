"""Model and checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(reference: str) -> Path:
    """Resolve a local path or Hugging Face repo id to a local directory."""
    path = Path(reference).expanduser()
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise ImportError("huggingface-hub is required to download model checkpoints") from exc
    return Path(snapshot_download(reference))
