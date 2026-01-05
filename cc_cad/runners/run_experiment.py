"""Main runner for CC-CAD experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from cc_cad.decoding.cad import CADOpenGenerator, StubAPIClient, cad_api_generate
from cc_cad.decoding.cc_cad import CCCADDecoder, extract_claims
from cc_cad.entailment.nli import NLIScorer
from cc_cad.eval.citation_metrics import compute_citation_metrics
from cc_cad.eval.metrics import compute_claim_metrics, exact_match, f1_score
from cc_cad.retrieval.faiss_index import FaissIndex
from cc_cad.retrieval.retriever import Retriever
from cc_cad.datasets.nq_open import load_nq_open
from cc_cad.datasets.triviaqa import load_triviaqa
from cc_cad.datasets.hotpotqa import load_hotpotqa
from cc_cad.datasets.fever import load_fever


DATASET_LOADERS = {
    "nq": load_nq_open,
    "triviaqa": load_triviaqa,
    "hotpotqa": load_hotpotqa,
    "fever": load_fever,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_retriever(cfg: Dict) -> Retriever:
    index_dir = cfg.get("index_dir")
    wiki_path = cfg.get("wiki_path")
    model_name = cfg.get("embed_model", "intfloat/e5-large-v2")
    index = FaissIndex(model_name=model_name)
    if index_dir and Path(index_dir, "index.faiss").exists():
        index.load(index_dir)
    else:
        if not wiki_path:
            raise ValueError("wiki_path is required to build FAISS index")
        index.build_from_jsonl(wiki_path)
        if index_dir:
            index.save(index_dir)
    return Retriever(index)


def generate_rag_answer(
    generator: CADOpenGenerator,
    question: str,
    evidence: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> str:
    prompt = f"Question: {question}\nProvide a short factual answer." 
    result = generator.generate(
        prompt=prompt,
        evidence=evidence,
        alpha=0.0,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        mode="greedy",
    )
    return result.text.strip()


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = load_config(args.config)
    retriever = build_retriever(cfg)
    nli = NLIScorer(model_name=cfg.get("nli_model", "roberta-large-mnli"), device=args.device)

    if args.mode == "open":
        model_name = cfg.get("model_name", args.model)
        generator = CADOpenGenerator(model_name_or_path=model_name, device=args.device)
        api_client = None
    else:
        generator = None
        api_client = StubAPIClient()

    dataset_loader = DATASET_LOADERS[args.dataset]
    dataset = dataset_loader(split=cfg.get("split", "validation"), local_path=args.local_dataset)
    if args.max_examples:
        dataset = dataset[: args.max_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    trace_path = output_dir / "trace.jsonl"
    results_path = output_dir / "results.jsonl"

    aggregate = {
        "em": [],
        "f1": [],
        "claim_f": [],
        "hall_r": [],
        "cite_p": [],
        "cite_r": [],
        "latency_sec": [],
        "regens": [],
        "cad_api_mode": [],
    }

    with predictions_path.open("w", encoding="utf-8") as pred_handle, trace_path.open(
        "w", encoding="utf-8"
    ) as trace_handle, results_path.open("w", encoding="utf-8") as results_handle:
        for example in dataset:
            question = example["question"]
            answers = example.get("answers", [])
            start_time = time.time()
            passages = retriever.retrieve(question, top_k=args.top_k)
            evidence = Retriever.format_evidence(passages)
            evidence_map = {p.pid: p.text for p in passages}

            claims = []
            citations = []
            entailment_scores = []
            answer_text = ""

            if args.method == "cc_cad":
                decoder = CCCADDecoder(
                    retriever=retriever,
                    nli=nli,
                    open_generator=generator,
                    api_client=api_client,
                )
                record = decoder.generate(
                    question=question,
                    top_k=args.top_k,
                    alpha=args.alpha,
                    threshold=args.threshold,
                    max_claims=args.max_claims,
                    mode=args.mode,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    use_entailment=not args.no_entailment_gating,
                )
                answer_text = record.answer
                cad_api_mode = record.cad_api_mode
                regen_count = sum(1 for c in record.claims if c.regenerated)
                for claim_rec in record.claims:
                    claims.append(claim_rec.claim)
                    citations.append(claim_rec.citation)
                    entailment_scores.append(claim_rec.entailment_score)
                    trace_handle.write(
                        json.dumps(
                            {
                                "id": example["id"],
                                **asdict(claim_rec),
                            }
                        )
                        + "\n"
                    )
            else:
                cad_api_mode = None
                regen_count = 0
                if args.mode == "open":
                    if args.method == "cad":
                        result = generator.generate(
                            prompt=f"Question: {question}\nProvide a short factual answer.",
                            evidence=evidence,
                            alpha=args.alpha,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            mode="greedy",
                        )
                        answer_text = result.text.strip()
                    else:
                        answer_text = generate_rag_answer(
                            generator,
                            question,
                            evidence,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                else:
                    prompt = f"Question: {question}\nProvide a short factual answer."
                    if args.method == "cad":
                        result = cad_api_generate(api_client, prompt, evidence, args.alpha)
                        answer_text = result.text.strip()
                        cad_api_mode = result.approximation
                    else:
                        answer_text = api_client.generate_with_context(prompt, evidence).text.strip()

                claims = extract_claims(answer_text)
                processed_claims: List[str] = []
                for claim in claims:
                    ent = nli.score_claim(claim, evidence_map.items())
                    if args.method == "rag_posthoc" and ent.score < args.threshold:
                        claim = "Not supported by evidence."
                        citation = ""
                    else:
                        citation = f"[{ent.passage_id}]" if ent.passage_id else ""
                    processed_claims.append(claim)
                    citations.append(citation)
                    entailment_scores.append(ent.score)
                    trace_handle.write(
                        json.dumps(
                            {
                                "id": example["id"],
                                "claim": claim,
                                "citation": citations[-1],
                                "entailment_score": ent.score,
                                "regenerated": False,
                            }
                        )
                        + "\n"
                    )
                if args.method == "rag_posthoc":
                    answer_text = " ".join(processed_claims).strip()
                    claims = processed_claims

            latency = time.time() - start_time

            em = exact_match(answer_text, answers)
            f1 = f1_score(answer_text, answers)
            claim_metrics = compute_claim_metrics(claims, evidence_map, nli, args.threshold)
            citation_metrics = compute_citation_metrics(
                claims, citations, evidence_map, nli, args.threshold
            )

            aggregate["em"].append(em)
            aggregate["f1"].append(f1)
            aggregate["claim_f"].append(claim_metrics["claim_f"])
            aggregate["hall_r"].append(claim_metrics["hall_r"])
            aggregate["cite_p"].append(citation_metrics["cite_p"])
            aggregate["cite_r"].append(citation_metrics["cite_r"])
            aggregate["latency_sec"].append(latency)
            aggregate["regens"].append(regen_count)
            if cad_api_mode:
                aggregate["cad_api_mode"].append(cad_api_mode)

            results_handle.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "em": em,
                        "f1": f1,
                        "claim_f": claim_metrics["claim_f"],
                        "hall_r": claim_metrics["hall_r"],
                        "cite_p": citation_metrics["cite_p"],
                        "cite_r": citation_metrics["cite_r"],
                        "latency_sec": latency,
                    }
                )
                + "\n"
            )

            pred_handle.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "question": question,
                        "answer": answer_text,
                        "claims": claims,
                        "citations": citations,
                    }
                )
                + "\n"
            )

    metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in aggregate.items() if k != "cad_api_mode"}
    metrics["avg_regens"] = float(np.mean(aggregate["regens"])) if aggregate["regens"] else 0.0
    if args.mode == "api" and args.method in {"cad", "cc_cad"}:
        if aggregate["cad_api_mode"]:
            metrics["cad_api_mode"] = max(set(aggregate["cad_api_mode"]), key=aggregate["cad_api_mode"].count)
        else:
            metrics["cad_api_mode"] = "unknown"
    metrics.update(
        {
            "dataset": args.dataset,
            "method": args.method,
            "mode": args.mode,
            "examples": len(dataset),
            "alpha": args.alpha,
            "threshold": args.threshold,
            "top_k": args.top_k,
        }
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS.keys())
    parser.add_argument("--mode", required=True, choices=["open", "api"])
    parser.add_argument(
        "--method", required=True, choices=["rag", "cad", "cc_cad", "rag_posthoc"]
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max_claims", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_entailment_gating", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--output_dir", default="results/run")
    parser.add_argument("--local_dataset", default=None)
    args = parser.parse_args()

    if args.max_examples == 0:
        args.max_examples = None

    run_experiment(args)


if __name__ == "__main__":
    main()
