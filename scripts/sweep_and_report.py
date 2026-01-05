"""Run the full CC-CAD experimental suite and write results to results/."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_cad.datasets.fever import load_fever
from cc_cad.datasets.hotpotqa import load_hotpotqa
from cc_cad.datasets.nq_open import load_nq_open
from cc_cad.datasets.triviaqa import load_triviaqa
from cc_cad.retrieval.faiss_index import FaissIndex
from cc_cad.retrieval.retriever import Retriever

DATASET_LOADERS = {
    "nq": load_nq_open,
    "triviaqa": load_triviaqa,
    "hotpotqa": load_hotpotqa,
    "fever": load_fever,
}

TOP_K = {
    "nq": 5,
    "triviaqa": 5,
    "hotpotqa": 8,
    "fever": 5,
}


def run_command(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def run_experiment(
    output_dir: Path,
    dataset: str,
    mode: str,
    method: str,
    config_path: Path,
    n_eval: int,
    top_k: int,
    seed: int,
    device: str,
    model: Optional[str],
    alpha: Optional[float] = None,
    threshold: Optional[float] = None,
    max_claims: Optional[int] = None,
    no_entailment_gating: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "cc_cad.runners.run_experiment",
        "--dataset",
        dataset,
        "--mode",
        mode,
        "--method",
        method,
        "--config",
        str(config_path),
        "--max_examples",
        str(n_eval),
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
        "--top_k",
        str(top_k),
        "--device",
        device,
    ]
    if model:
        cmd += ["--model", model]
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    if max_claims is not None:
        cmd += ["--max_claims", str(max_claims)]
    if no_entailment_gating:
        cmd += ["--no_entailment_gating"]
    run_command(cmd)
    return cmd


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_retriever(config_path: Path) -> Retriever:
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


def load_predictions(path: Path) -> Dict[str, str]:
    predictions = {}
    if not path.exists():
        return predictions
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            predictions[str(row["id"])]=row.get("answer", "")
    return predictions


def format_snippets(passages: List) -> str:
    snippets = []
    for passage in passages[:3]:
        text = passage.text.strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        snippets.append(f"[{passage.pid}] {text}")
    return " | ".join(snippets)


def write_manifest(path: Path, manifest: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def get_git_hash() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_hardware_info() -> Dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "unknown"
    return info


def run_table2(
    base_dir: Path, config_dir: Path, n_eval: int, seed: int, device: str, model: Optional[str]
) -> List[Dict]:
    runs = []
    for dataset in ["nq", "triviaqa", "hotpotqa"]:
        top_k = TOP_K[dataset]
        for method in ["rag", "rag_posthoc", "cad", "cc_cad"]:
            output_dir = base_dir / "table2" / dataset / method
            cmd = run_experiment(
                output_dir=output_dir,
                dataset=dataset,
                mode="open",
                method=method,
                config_path=config_dir / f"{dataset}.yaml",
                n_eval=n_eval,
                top_k=top_k,
                seed=seed,
                device=device,
                model=model,
                alpha=0.5 if method in {"cad", "cc_cad"} else None,
                threshold=0.75 if method == "cc_cad" else None,
                max_claims=5 if method == "cc_cad" else None,
            )
            runs.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mode": "open",
                    "output_dir": str(output_dir),
                    "cmd": cmd,
                }
            )
    return runs


def run_table3(
    base_dir: Path, config_dir: Path, n_eval: int, seed: int, device: str, model: Optional[str]
) -> List[Dict]:
    runs = []
    for dataset in ["nq", "hotpotqa"]:
        top_k = TOP_K[dataset]
        for method in ["rag", "cad", "cc_cad"]:
            output_dir = base_dir / "table3" / dataset / method
            cmd = run_experiment(
                output_dir=output_dir,
                dataset=dataset,
                mode="api",
                method=method,
                config_path=config_dir / f"{dataset}.yaml",
                n_eval=n_eval,
                top_k=top_k,
                seed=seed,
                device=device,
                model=model,
                alpha=0.5 if method in {"cad", "cc_cad"} else None,
                threshold=0.75 if method == "cc_cad" else None,
                max_claims=5 if method == "cc_cad" else None,
            )
            runs.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mode": "api",
                    "output_dir": str(output_dir),
                    "cmd": cmd,
                }
            )
    return runs


def run_table4(
    base_dir: Path, config_dir: Path, n_eval: int, seed: int, device: str, model: Optional[str]
) -> List[Dict]:
    dataset = "hotpotqa"
    top_k = TOP_K[dataset]
    variants = [
        ("full", {"alpha": 0.5, "threshold": 0.75, "max_claims": 5, "no_entailment_gating": False}),
        ("alpha0", {"alpha": 0.0, "threshold": 0.75, "max_claims": 5, "no_entailment_gating": False}),
        ("no_entailment", {"alpha": 0.5, "threshold": 0.75, "max_claims": 5, "no_entailment_gating": True}),
        ("threshold_plus", {"alpha": 0.5, "threshold": 0.85, "max_claims": 5, "no_entailment_gating": False}),
        ("threshold_minus", {"alpha": 0.5, "threshold": 0.65, "max_claims": 5, "no_entailment_gating": False}),
    ]
    runs = []
    for label, params in variants:
        output_dir = base_dir / "table4" / dataset / label
        cmd = run_experiment(
            output_dir=output_dir,
            dataset=dataset,
            mode="open",
            method="cc_cad",
            config_path=config_dir / f"{dataset}.yaml",
            n_eval=n_eval,
            top_k=top_k,
            seed=seed,
            device=device,
            model=model,
            alpha=params["alpha"],
            threshold=params["threshold"],
            max_claims=params["max_claims"],
            no_entailment_gating=params["no_entailment_gating"],
        )
        runs.append(
            {
                "dataset": dataset,
                "method": "cc_cad",
                "variant": label,
                "output_dir": str(output_dir),
                "cmd": cmd,
            }
        )
    return runs


def run_fig4(
    base_dir: Path, config_dir: Path, n_eval: int, seed: int, device: str, model: Optional[str]
) -> None:
    dataset = "nq"
    top_k = TOP_K[dataset]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    records = []
    for threshold in thresholds:
        output_dir = base_dir / "fig4" / f"thr_{threshold:.2f}"
        run_experiment(
            output_dir=output_dir,
            dataset=dataset,
            mode="open",
            method="cc_cad",
            config_path=config_dir / f"{dataset}.yaml",
            n_eval=n_eval,
            top_k=top_k,
            seed=seed,
            device=device,
            model=model,
            alpha=0.5,
            threshold=threshold,
            max_claims=5,
        )
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            records.append(
                {
                    "threshold": threshold,
                    "em": metrics.get("em", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "hall_r": metrics.get("hall_r", 0.0),
                    "cite_p": metrics.get("cite_p", 0.0),
                    "cite_r": metrics.get("cite_r", 0.0),
                }
            )

    output_csv = Path("results/fig4_tradeoff.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys() if records else [])
        writer.writeheader()
        writer.writerows(records)


def run_human_eval(
    base_dir: Path, config_dir: Path, n_eval: int, seed: int, device: str, model: Optional[str]
) -> None:
    rng = random.Random(seed)
    output_root = base_dir / "human_eval"
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = ["nq", "hotpotqa", "fever"]
    for dataset in datasets:
        top_k = TOP_K[dataset]
        for method in ["rag", "cad", "cc_cad"]:
            output_dir = output_root / dataset / method
            run_experiment(
                output_dir=output_dir,
                dataset=dataset,
                mode="open",
                method=method,
                config_path=config_dir / f"{dataset}.yaml",
                n_eval=n_eval,
                top_k=top_k,
                seed=seed,
                device=device,
                model=model,
                alpha=0.5 if method in {"cad", "cc_cad"} else None,
                threshold=0.75 if method == "cc_cad" else None,
                max_claims=5 if method == "cc_cad" else None,
            )

    items_path = Path("results/human_eval/items.csv")
    items_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "dataset",
        "question",
        "evidence_snippets",
        "answer_rag",
        "answer_cad",
        "answer_cc_cad",
    ]

    with items_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            data = DATASET_LOADERS[dataset](split="validation")
            ids = [str(item["id"]) for item in data]
            rng.shuffle(ids)
            sample_ids = ids[:n_eval]
            question_map = {str(item["id"]): item["question"] for item in data}

            retriever = build_retriever(config_dir / f"{dataset}.yaml")
            preds_rag = load_predictions(output_root / dataset / "rag" / "predictions.jsonl")
            preds_cad = load_predictions(output_root / dataset / "cad" / "predictions.jsonl")
            preds_cc = load_predictions(output_root / dataset / "cc_cad" / "predictions.jsonl")

            for qid in sample_ids:
                question = question_map.get(qid, "")
                passages = retriever.retrieve(question, top_k=TOP_K[dataset])
                evidence = format_snippets(passages)
                writer.writerow(
                    {
                        "id": qid,
                        "dataset": dataset,
                        "question": question,
                        "evidence_snippets": evidence,
                        "answer_rag": preds_rag.get(qid, ""),
                        "answer_cad": preds_cad.get(qid, ""),
                        "answer_cc_cad": preds_cc.get(qid, ""),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table2_n", type=int, default=500)
    parser.add_argument("--table3_n", type=int, default=200)
    parser.add_argument("--table4_n", type=int, default=500)
    parser.add_argument("--fig4_n", type=int, default=500)
    parser.add_argument("--human_eval_n", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("results") / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(args.config_dir)

    manifest = {
        "timestamp": timestamp,
        "git_hash": get_git_hash(),
        "hardware": get_hardware_info(),
        "seed": args.seed,
        "configs": {},
        "runs": [],
    }

    for cfg_path in config_dir.glob("*.yaml"):
        manifest["configs"][cfg_path.stem] = load_config(cfg_path)

    manifest["runs"].extend(
        run_table2(base_dir, config_dir, args.table2_n, args.seed, args.device, args.model)
    )
    manifest["runs"].extend(
        run_table3(base_dir, config_dir, args.table3_n, args.seed, args.device, args.model)
    )
    manifest["runs"].extend(
        run_table4(base_dir, config_dir, args.table4_n, args.seed, args.device, args.model)
    )
    run_fig4(base_dir, config_dir, args.fig4_n, args.seed, args.device, args.model)
    run_human_eval(base_dir, config_dir, args.human_eval_n, args.seed, args.device, args.model)

    write_manifest(base_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
