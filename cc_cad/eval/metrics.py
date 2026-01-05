"""Evaluation metrics for QA and claim support."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from cc_cad.entailment.nli import NLIScorer


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ans) for ans in answers))


def f1_score(prediction: str, answers: Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    scores = []
    for ans in answers:
        gold_tokens = normalize_answer(ans).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            scores.append(0.0)
            continue
        precision = len(common) / max(len(pred_tokens), 1)
        recall = len(common) / max(len(gold_tokens), 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-8))
    return max(scores) if scores else 0.0


def claim_support_rate(claims: List[str], evidence: Dict[str, str], nli: NLIScorer, threshold: float) -> float:
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli.score_claim(claim, evidence.items())
        if result.score >= threshold:
            supported += 1
    return supported / max(len(claims), 1)


def compute_claim_metrics(claims: List[str], evidence: Dict[str, str], nli: NLIScorer, threshold: float) -> Dict[str, float]:
    support_rate = claim_support_rate(claims, evidence, nli, threshold)
    return {
        "claim_f": support_rate,
        "hall_r": 1.0 - support_rate,
    }
