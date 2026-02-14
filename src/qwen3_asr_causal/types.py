"""Small timestamped text types used by the streaming processors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRToken:
    start: Optional[float] = 0
    end: Optional[float] = 0
    text: Optional[str] = ""
    speaker: Optional[int] = -1
    detected_language: Optional[str] = None
    probability: Optional[float] = None

    def with_offset(self, offset: float) -> "ASRToken":
        return ASRToken(
            start=None if self.start is None else self.start + offset,
            end=None if self.end is None else self.end + offset,
            text=self.text,
            speaker=self.speaker,
            detected_language=self.detected_language,
            probability=self.probability,
        )

    def is_silence(self) -> bool:
        return False


@dataclass
class Transcript:
    start: Optional[float] = 0
    end: Optional[float] = 0
    text: Optional[str] = ""
    speaker: Optional[int] = -1
    detected_language: Optional[str] = None

    @classmethod
    def from_tokens(
        cls,
        tokens: list[ASRToken],
        sep: Optional[str] = None,
        offset: float = 0,
    ) -> "Transcript":
        sep = " " if sep is None else sep
        text = sep.join(token.text or "" for token in tokens)
        if tokens:
            start = None if tokens[0].start is None else offset + tokens[0].start
            end = None if tokens[-1].end is None else offset + tokens[-1].end
        else:
            start = None
            end = None
        return cls(start=start, end=end, text=text)

    def __bool__(self) -> bool:
        return bool(self.text)
