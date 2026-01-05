"""Streamlit UI for CC-CAD QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import yaml

from cc_cad.decoding.cad import CADOpenGenerator, StubAPIClient, cad_api_generate
from cc_cad.decoding.cc_cad import CCCADDecoder
from cc_cad.entailment.nli import NLIScorer
from cc_cad.retrieval.faiss_index import FaissIndex
from cc_cad.retrieval.retriever import Retriever


@st.cache_resource
def load_config(config_path: str) -> Dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@st.cache_resource
def load_retriever(config_path: str) -> Retriever:
    cfg = load_config(config_path)
    index_dir = cfg.get("index_dir")
    wiki_path = cfg.get("wiki_path")
    embed_model = cfg.get("embed_model", "intfloat/e5-large-v2")
    index = FaissIndex(model_name=embed_model)
    if index_dir and Path(index_dir, "index.faiss").exists():
        index.load(index_dir)
    else:
        if not wiki_path:
            raise ValueError("wiki_path is required to build FAISS index")
        index.build_from_jsonl(wiki_path)
        if index_dir:
            index.save(index_dir)
    return Retriever(index)


@st.cache_resource
def load_generator(model_name: str, device: str) -> CADOpenGenerator:
    return CADOpenGenerator(model_name_or_path=model_name, device=device)


@st.cache_resource
def load_nli(model_name: str, device: str) -> NLIScorer:
    return NLIScorer(model_name=model_name, device=device)


def run_open_method(
    question: str,
    method: str,
    retriever: Retriever,
    generator: CADOpenGenerator,
    nli: NLIScorer,
    top_k: int,
    alpha: float,
    threshold: float,
    max_claims: int,
) -> Dict:
    passages = retriever.retrieve(question, top_k=top_k)
    evidence = Retriever.format_evidence(passages)

    if method == "rag":
        result = generator.generate(
            prompt=f"Question: {question}\nProvide a short factual answer.",
            evidence=evidence,
            alpha=0.0,
            max_new_tokens=128,
            mode="greedy",
        )
        return {"answer": result.text.strip(), "claims": [], "citations": []}
    if method == "cad":
        result = generator.generate(
            prompt=f"Question: {question}\nProvide a short factual answer.",
            evidence=evidence,
            alpha=alpha,
            max_new_tokens=128,
            mode="greedy",
        )
        return {"answer": result.text.strip(), "claims": [], "citations": []}

    decoder = CCCADDecoder(
        retriever=retriever,
        nli=nli,
        open_generator=generator,
        api_client=None,
    )
    record = decoder.generate(
        question=question,
        top_k=top_k,
        alpha=alpha,
        threshold=threshold,
        max_claims=max_claims,
        mode="open",
    )
    return {
        "answer": record.answer,
        "claims": [c.claim for c in record.claims],
        "citations": [c.citation for c in record.claims],
    }


def run_api_method(
    question: str,
    method: str,
    retriever: Retriever,
    top_k: int,
    alpha: float,
    threshold: float,
    max_claims: int,
) -> Dict:
    passages = retriever.retrieve(question, top_k=top_k)
    evidence = Retriever.format_evidence(passages)
    client = StubAPIClient()

    if method == "cad":
        result = cad_api_generate(client, f"Question: {question}", evidence, alpha)
        return {"answer": result.text.strip(), "claims": [], "citations": []}

    decoder = CCCADDecoder(
        retriever=retriever,
        nli=NLIScorer(model_name="mock"),
        open_generator=None,
        api_client=client,
    )
    record = decoder.generate(
        question=question,
        top_k=top_k,
        alpha=alpha,
        threshold=threshold,
        max_claims=max_claims,
        mode="api",
    )
    return {
        "answer": record.answer,
        "claims": [c.claim for c in record.claims],
        "citations": [c.citation for c in record.claims],
    }


def main() -> None:
    st.title("CC-CAD QA")
    st.markdown("Interactive CC-CAD demo over a local Wikipedia FAISS index.")

    with st.sidebar:
        config_path = st.text_input("Config path", value="configs/nq.yaml")
        method = st.selectbox("Method", options=["rag", "cad", "cc_cad"])
        mode = st.selectbox("Mode", options=["open", "api"])
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5)
        alpha = st.slider("Alpha", min_value=0.0, max_value=2.0, value=0.5, step=0.05)
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
        max_claims = st.number_input("Max claims", min_value=1, max_value=10, value=5)
        device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
        model_override = st.text_input("Model override (optional)", value="")

    question = st.text_area("Question", value="Who wrote The Old Man and the Sea?")
    run_button = st.button("Run")

    if run_button and question.strip():
        cfg = load_config(config_path)
        retriever = load_retriever(config_path)

        if mode == "open":
            model_name = model_override or cfg.get("model_name", "meta-llama/Llama-2-13b-chat-hf")
            generator = load_generator(model_name, device)
            nli = load_nli(cfg.get("nli_model", "roberta-large-mnli"), device)
            result = run_open_method(
                question,
                method,
                retriever,
                generator,
                nli,
                top_k,
                alpha,
                threshold,
                max_claims,
            )
        else:
            result = run_api_method(
                question,
                method,
                retriever,
                top_k,
                alpha,
                threshold,
                max_claims,
            )

        st.subheader("Answer")
        st.write(result["answer"])
        if result["claims"]:
            st.subheader("Claims")
            for claim, citation in zip(result["claims"], result["citations"]):
                st.write(f"- {claim} {citation}")
        st.subheader("Raw output")
        st.code(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
