"""Claim splitting utilities."""

from __future__ import annotations

import re
from typing import List


_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
}


def split_claims(text: str) -> List[str]:
    """Split text into claims on sentence boundaries."""
    normalized = text.strip()
    parts = re.split(r"(?:;|(?<=[.])\s+|\n+)", normalized)
    claims: List[str] = []
    buffer = ""
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        lower = candidate.lower()
        if any(lower.endswith(abbrev) for abbrev in _ABBREVIATIONS):
            buffer = f"{buffer} {candidate}".strip()
            continue
        if buffer:
            candidate = f"{buffer} {candidate}".strip()
            buffer = ""
        claims.append(candidate)
    if buffer:
        claims.append(buffer.strip())
    return claims
