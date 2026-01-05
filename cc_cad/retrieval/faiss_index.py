"""FAISS index wrapper for Wikipedia paragraph retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None

from sentence_transformers import SentenceTransformer


@dataclass
class Passage:
    pid: str
    title: str
    text: str


class FaissIndex:
    """Builds and queries a FAISS index over passage embeddings."""

    def __init__(self, model_name: str = "intfloat/e5-large-v2") -> None:
        if faiss is None:
            raise ImportError("faiss is required for FaissIndex")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages: List[Passage] = []

    def build_from_jsonl(self, jsonl_path: str | Path, batch_size: int = 64) -> None:
        """Build index from a jsonl file of {id,title,text}."""
        path = Path(jsonl_path)
        passages: List[Passage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                passages.append(Passage(str(data["id"]), data.get("title", ""), data["text"]))

        self.passages = passages
        texts = [f"{p.title}\n{p.text}".strip() for p in passages]
        embeddings = self._encode(texts, batch_size=batch_size)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def save(self, dir_path: str | Path) -> None:
        """Persist the FAISS index and metadata to a directory."""
        if self.index is None:
            raise ValueError("Index not built")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        meta_path = dir_path / "passages.jsonl"
        with meta_path.open("w", encoding="utf-8") as handle:
            for p in self.passages:
                handle.write(json.dumps({"id": p.pid, "title": p.title, "text": p.text}) + "\n")

    def load(self, dir_path: str | Path) -> None:
        """Load an index and metadata from a directory."""
        dir_path = Path(dir_path)
        self.index = faiss.read_index(str(dir_path / "index.faiss"))
        self.passages = []
        with (dir_path / "passages.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                self.passages.append(Passage(str(data["id"]), data.get("title", ""), data["text"]))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Passage, float]]:
        """Return top_k passages and scores for the query."""
        if self.index is None:
            raise ValueError("Index not built")
        query_emb = self._encode([query])
        faiss.normalize_L2(query_emb)
        scores, idxs = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append((self.passages[int(idx)], float(score)))
        return results

    def _encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)
