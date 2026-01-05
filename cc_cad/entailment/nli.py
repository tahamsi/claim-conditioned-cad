"""Entailment scoring with NLI models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class EntailmentResult:
    passage_id: str
    score: float


class NLIScorer:
    """Scores entailment probabilities for (premise, hypothesis) pairs."""

    def __init__(self, model_name: Optional[str] = "roberta-large-mnli", device: str = "cpu") -> None:
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        if model_name and model_name != "mock":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()

    def score_entailment(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability."""
        if self.model_name == "mock" or self.model is None or self.tokenizer is None:
            return self._heuristic_entailment(premise, hypothesis)
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # MNLI labels: contradiction, neutral, entailment
        return float(probs[-1].item())

    def score_claim(self, claim: str, passages: Iterable[Tuple[str, str]]) -> EntailmentResult:
        """Score a claim against multiple passages and return best."""
        best_score = -1.0
        best_pid = ""
        for pid, text in passages:
            score = self.score_entailment(text, claim)
            if score > best_score:
                best_score = score
                best_pid = pid
        return EntailmentResult(passage_id=best_pid, score=best_score)

    @staticmethod
    def _heuristic_entailment(premise: str, hypothesis: str) -> float:
        premise_tokens = set(premise.lower().split())
        hypothesis_tokens = set(hypothesis.lower().split())
        if not hypothesis_tokens:
            return 0.0
        overlap = len(premise_tokens & hypothesis_tokens)
        return min(1.0, overlap / max(len(hypothesis_tokens), 1))
