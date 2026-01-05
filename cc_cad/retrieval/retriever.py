"""Retriever abstraction over FAISS index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from cc_cad.retrieval.faiss_index import FaissIndex, Passage


@dataclass
class RetrievedPassage:
    pid: str
    title: str
    text: str
    score: float


class Retriever:
    """Retrieves top-k passages for a query."""

    def __init__(self, index: FaissIndex) -> None:
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        results = self.index.search(query, top_k=top_k)
        return [
            RetrievedPassage(pid=p.pid, title=p.title, text=p.text, score=score)
            for p, score in results
        ]

    @staticmethod
    def format_evidence(passages: List[RetrievedPassage]) -> str:
        """Create a formatted evidence block for prompting."""
        chunks = []
        for passage in passages:
            chunks.append(f"[{passage.pid}] {passage.title}\n{passage.text}")
        return "\n\n".join(chunks)
