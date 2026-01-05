"""Download Wikipedia and write a paragraph JSONL corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("datasets is required to download Wikipedia") from exc


def iter_wikipedia(split: str, streaming: bool, language: str) -> Iterable[dict]:
    """Load Wikipedia from a non-script dataset if possible."""
    try:
        return load_dataset(
            "wikimedia/wikipedia",
            language,
            split=split,
            streaming=streaming,
        )
    except ValueError:
        if language.endswith(".en") and language != "20231101.en":
            language = "20231101.en"
            return load_dataset(
                "wikimedia/wikipedia",
                language,
                split=split,
                streaming=streaming,
            )
        raise
    except Exception:
        return _load_parquet_fallback(language, split)


def _load_parquet_fallback(language: str, split: str) -> Iterable[dict]:
    """Fallback to direct parquet shards (no dataset script)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("huggingface_hub is required for parquet fallback") from exc

    repo_id = "wikimedia/wikipedia"
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [
        f
        for f in files
        if f.endswith(".parquet")
        and language in f
        and (f"/{split}-" in f or f"/{split}/" in f or f"{split}-" in f)
    ]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for {language} split={split}")
    local_files = [
        hf_hub_download(repo_id, filename=path, repo_type="dataset") for path in parquet_files
    ]
    return load_dataset("parquet", data_files={"train": local_files}, split="train")


def split_paragraphs(text: str) -> List[str]:
    paragraphs = []
    for chunk in text.split("\n\n"):
        para = " ".join(chunk.split())
        if para:
            paragraphs.append(para)
    return paragraphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/wiki_paragraphs.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="20231101.en")
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--hard_exit", action="store_true", default=False)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = iter_wikipedia(args.split, args.streaming, args.language)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in ds:
            title = row.get("title", "")
            doc_id = row.get("id", str(count))
            text = row.get("text", "")
            paragraphs = split_paragraphs(text)
            for p_idx, paragraph in enumerate(paragraphs):
                record = {"id": f"{doc_id}:{p_idx}", "title": title, "text": paragraph}
                handle.write(json.dumps(record) + "\n")
                count += 1
                if args.max_docs and count >= args.max_docs:
                    break
            if args.max_docs and count >= args.max_docs:
                break

    print(f"Wrote {count} passages to {output_path}")
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
