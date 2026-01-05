"""Claim-conditioned CAD decoding pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from cc_cad.decoding.cad import APIClient, CADGeneration, CADOpenGenerator, cad_api_generate
from cc_cad.entailment.nli import NLIScorer
from cc_cad.entailment.span_select import select_best_span
from cc_cad.eval.claim_split import split_claims
from cc_cad.retrieval.retriever import RetrievedPassage, Retriever


@dataclass
class ClaimRecord:
    claim: str
    citation: str
    entailment_score: float
    regenerated: bool


@dataclass
class AnswerRecord:
    answer: str
    claims: List[ClaimRecord] = field(default_factory=list)
    cad_api_mode: Optional[str] = None


class CCCADDecoder:
    """Generate answers claim-by-claim with CAD and entailment filtering."""

    def __init__(
        self,
        retriever: Retriever,
        nli: NLIScorer,
        open_generator: Optional[CADOpenGenerator] = None,
        api_client: Optional[APIClient] = None,
    ) -> None:
        self.retriever = retriever
        self.nli = nli
        self.open_generator = open_generator
        self.api_client = api_client

    def generate(
        self,
        question: str,
        top_k: int,
        alpha: float,
        threshold: float,
        max_claims: int,
        mode: str = "open",
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        use_entailment: bool = True,
    ) -> AnswerRecord:
        passages = self.retriever.retrieve(question, top_k=top_k)
        evidence = Retriever.format_evidence(passages)
        evidence_map = {p.pid: p.text for p in passages}
        claims: List[ClaimRecord] = []
        answer_parts: List[str] = []
        cad_api_mode: Optional[str] = None

        for claim_index in range(max_claims):
            claim_prompt = (
                f"Question: {question}\n"
                "Write one concise factual claim that answers the question."
            )
            generation = self._generate_claim(
                claim_prompt, evidence, alpha, mode, temperature, top_p
            )
            if generation.approximation:
                cad_api_mode = _merge_api_mode(cad_api_mode, generation.approximation)
            candidate = generation.text.strip()
            if not candidate:
                break
            regenerated = False
            if use_entailment:
                entailment = self.nli.score_claim(candidate, evidence_map.items())
                if entailment.score < threshold:
                    regen_prompt = (
                        f"Question: {question}\n"
                        "Rewrite the claim to be directly supported by the evidence; "
                        "if not supported, say 'Not supported by evidence'."
                    )
                    generation = self._generate_claim(
                        regen_prompt, evidence, alpha * 1.5, mode, temperature, top_p
                    )
                    if generation.approximation:
                        cad_api_mode = _merge_api_mode(cad_api_mode, generation.approximation)
                    candidate = generation.text.strip()
                    entailment = self.nli.score_claim(candidate, evidence_map.items())
                    regenerated = True
                if entailment.score < threshold:
                    break
                best_passage = next(
                    (p for p in passages if p.pid == entailment.passage_id), None
                )
                entailment_score = entailment.score
            else:
                best_passage = passages[0] if passages else None
                entailment_score = 1.0
            citation = ""
            if best_passage:
                span = select_best_span(best_passage.text, candidate)
                citation = f"[{best_passage.pid}:{span.start}-{span.end}]"
            claims.append(
                ClaimRecord(
                    claim=candidate,
                    citation=citation,
                    entailment_score=entailment_score,
                    regenerated=regenerated,
                )
            )
            answer_parts.append(candidate)
        full_answer = " ".join(answer_parts).strip()
        return AnswerRecord(answer=full_answer, claims=claims, cad_api_mode=cad_api_mode)

    def _generate_claim(
        self,
        prompt: str,
        evidence: str,
        alpha: float,
        mode: str,
        temperature: float,
        top_p: Optional[float],
    ) -> CADGeneration:
        if mode == "open":
            if self.open_generator is None:
                raise ValueError("open_generator is required for open mode")
            return self.open_generator.generate(
                prompt=prompt,
                evidence=evidence,
                alpha=alpha,
                max_new_tokens=64,
                temperature=temperature,
                top_p=top_p,
                mode="greedy",
            )
        if mode == "api":
            if self.api_client is None:
                raise ValueError("api_client is required for api mode")
            return cad_api_generate(self.api_client, prompt, evidence, alpha)
        raise ValueError(f"Unknown mode: {mode}")


def extract_claims(answer_text: str) -> List[str]:
    """Split answer into claims for post-hoc evaluation."""
    return split_claims(answer_text)


def _merge_api_mode(existing: Optional[str], new_mode: str) -> Optional[str]:
    if existing == "token_fusion" or new_mode == "token_fusion":
        return "token_fusion"
    if existing is None:
        return new_mode
    return existing
