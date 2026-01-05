"""Summarize human evaluation annotations for Table 5."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> None:
    input_path = Path("results/human_eval/annotations.csv")
    if not input_path.exists():
        raise FileNotFoundError("results/human_eval/annotations.csv not found")

    totals: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row.get("dataset", "")
            if not dataset:
                continue
            for method_key, col in {
                "rag": "support_rag",
                "cad": "support_cad",
                "cc_cad": "support_cc_cad",
            }.items():
                value = row.get(col, "").strip().lower()
                if value in {"1", "yes", "true"}:
                    totals[dataset][method_key].append(1)
                elif value in {"0", "no", "false"}:
                    totals[dataset][method_key].append(0)

    print("dataset,method,supported_rate,count")
    for dataset, methods in totals.items():
        for method, values in methods.items():
            if not values:
                continue
            rate = sum(values) / max(len(values), 1)
            print(f"{dataset},{method},{rate:.3f},{len(values)}")


if __name__ == "__main__":
    main()
