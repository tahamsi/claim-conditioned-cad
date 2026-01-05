"""Citation metrics for claim verification."""

from __future__ import annotations

from typing import Dict, List, Tuple

from cc_cad.entailment.nli import NLIScorer


def compute_citation_metrics(
    claims: List[str],
    citations: List[str],
    evidence: Dict[str, str],
    nli: NLIScorer,
    threshold: float,
) -> Dict[str, float]:
    """Compute Cite-P and Cite-R based on cited span entailment."""
    if not claims:
        return {"cite_p": 0.0, "cite_r": 0.0}
    supported = 0
    cited = 0
    for claim, citation in zip(claims, citations):
        if not citation:
            continue
        cited += 1
        doc_id, span_text = _extract_span_text(citation, evidence)
        passage = span_text if span_text else evidence.get(doc_id, "")
        score = nli.score_entailment(passage, claim)
        if score >= threshold:
            supported += 1
    cite_p = supported / max(cited, 1)
    cite_r = supported / max(len(claims), 1)
    return {"cite_p": cite_p, "cite_r": cite_r}


def _extract_span_text(citation: str, evidence: Dict[str, str]) -> Tuple[str, str]:
    citation = citation.strip()
    if not citation.startswith("[") or not citation.endswith("]"):
        return citation, ""
    inner = citation.strip("[]")
    if ":" not in inner:
        return inner, ""
    doc_id, span = inner.split(":", 1)
    if "-" not in span:
        return doc_id, ""
    start_str, end_str = span.split("-", 1)
    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError:
        return doc_id, ""
    passage = evidence.get(doc_id, "")
    return doc_id, passage[start:end]
