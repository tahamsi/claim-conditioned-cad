"""Loader for FEVER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None


def _load_local(path: str | Path) -> List[Dict]:
    data = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            answers = row.get("answers") or row.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            data.append(
                {
                    "id": str(row.get("id") or len(data)),
                    "question": row.get("claim") or row.get("question"),
                    "answers": answers,
                    "context": row.get("context"),
                    "type": "fever",
                }
            )
    return data


def load_fever(split: str = "validation", local_path: Optional[str] = None) -> List[Dict]:
    if local_path:
        return _load_local(local_path)
    if load_dataset is None:
        raise ImportError("datasets is required to download FEVER")
    ds = load_dataset("fever", split=split)
    data = []
    for row in ds:
        label = row.get("label")
        answers = [label] if label else []
        data.append(
            {
                "id": str(row["id"]),
                "question": row["claim"],
                "answers": answers,
                "context": None,
                "type": "fever",
            }
        )
    return data
