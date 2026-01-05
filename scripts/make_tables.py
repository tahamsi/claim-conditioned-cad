"""Generate tables from metrics and configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_metrics(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in results_dir.rglob("metrics.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return pd.DataFrame(rows)


def load_configs(config_dir: Path) -> pd.DataFrame:
    rows = []
    for path in config_dir.glob("*.yaml"):
        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        rows.append(
            {
                "dataset": cfg.get("dataset", path.stem),
                "split": cfg.get("split", "validation"),
                "index_dir": cfg.get("index_dir", ""),
                "wiki_path": cfg.get("wiki_path", ""),
                "embed_model": cfg.get("embed_model", "intfloat/e5-large-v2"),
                "nli_model": cfg.get("nli_model", "roberta-large-mnli"),
            }
        )
    return pd.DataFrame(rows)


def write_table_markdown(df: pd.DataFrame, title: str, path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {title}\n\n")
        handle.write(df.to_markdown(index=False))
        handle.write("\n\n")


def make_human_eval_template(metrics: pd.DataFrame, output_csv: Path) -> None:
    fieldnames = [
        "dataset",
        "method",
        "mode",
        "example_id",
        "question",
        "answer",
        "supported",
        "notes",
    ]
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "mode": row.get("mode", ""),
                "example_id": "",
                "question": "",
                "answer": "",
                "supported": "",
                "notes": "",
            }
        )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base = Path(".")
    results_dir = base / "results"
    config_dir = base / "configs"
    tables_path = results_dir / "tables.md"
    tables_path.write_text("", encoding="utf-8")

    configs = load_configs(config_dir)
    metrics = load_metrics(results_dir)

    if not configs.empty:
        write_table_markdown(configs, "Table 1: Dataset Settings", tables_path)

    if not metrics.empty:
        open_results = metrics[metrics["mode"] == "open"]
        api_results = metrics[metrics["mode"] == "api"]
        if not open_results.empty:
            write_table_markdown(
                open_results[["dataset", "method", "em", "f1", "claim_f", "hall_r"]],
                "Table 2: Open-Weight Results",
                tables_path,
            )
        if not api_results.empty:
            write_table_markdown(
                api_results[["dataset", "method", "em", "f1", "claim_f", "hall_r"]],
                "Table 3: API Results",
                tables_path,
            )
        ablations = metrics[["dataset", "method", "alpha", "threshold", "f1", "hall_r"]]
        if not ablations.empty:
            write_table_markdown(ablations, "Table 4: Ablations", tables_path)

    human_csv = results_dir / "table5_human_eval.csv"
    make_human_eval_template(metrics, human_csv)


if __name__ == "__main__":
    main()
