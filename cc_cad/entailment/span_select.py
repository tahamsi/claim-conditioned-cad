"""Citation span selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SpanSelection:
    start: int
    end: int
    score: float


def select_best_span(passage: str, claim: str, window_size: int = 30) -> SpanSelection:
    """Select the best matching span based on token overlap."""
    words = passage.split()
    if not words:
        return SpanSelection(start=0, end=0, score=0.0)
    claim_tokens = set(claim.lower().split())
    best_score = -1.0
    best_window = (0, min(window_size, len(words)))
    for start in range(0, len(words), max(1, window_size // 2)):
        end = min(len(words), start + window_size)
        window_tokens = set(" ".join(words[start:end]).lower().split())
        if not claim_tokens:
            score = 0.0
        else:
            score = len(window_tokens & claim_tokens) / max(len(claim_tokens), 1)
        if score > best_score:
            best_score = score
            best_window = (start, end)
        if end == len(words):
            break
    start_char = len(" ".join(words[: best_window[0]]))
    if start_char > 0:
        start_char += 1
    span_text = " ".join(words[best_window[0] : best_window[1]])
    end_char = start_char + len(span_text)
    return SpanSelection(start=start_char, end=end_char, score=best_score)
