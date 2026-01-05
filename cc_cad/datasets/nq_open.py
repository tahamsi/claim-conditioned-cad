"""Loader for Natural Questions (open)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
                    "id": str(row.get("id") or row.get("qid") or len(data)),
                    "question": row["question"],
                    "answers": answers,
                    "context": row.get("context"),
                    "type": "nq",
                }
            )
    return data


def load_nq_open(split: str = "validation", local_path: Optional[str] = None) -> List[Dict]:
    if local_path:
        return _load_local(local_path)
    if load_dataset is None:
        raise ImportError("datasets is required to download NQ")
    ds = load_dataset("nq_open", split=split)
    data = []
    for idx, row in enumerate(ds):
        data.append(
            {
                "id": str(
                    row.get("id")
                    or row.get("example_id")
                    or row.get("question_id")
                    or idx
                ),
                "question": row["question"],
                "answers": row["answer"],
                "context": None,
                "type": "nq",
            }
        )
    return data
