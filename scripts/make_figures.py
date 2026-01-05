"""Generate figures from results metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in results_dir.rglob("metrics.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return pd.DataFrame(rows)


def main() -> None:
    results_dir = Path("results")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(results_dir)
    if metrics.empty:
        return

    # Figure 4: accuracy vs hallucination tradeoff vs threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in metrics.groupby("method"):
        ax.scatter(group["hall_r"], group["f1"], label=method)
    ax.set_xlabel("Hallucination Rate (Hall-R)")
    ax.set_ylabel("F1")
    ax.set_title("Accuracy vs Hallucination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_tradeoff.png")
    plt.close(fig)

    # Bar chart for Cite-P/Cite-R across methods
    fig, ax = plt.subplots(figsize=(6, 4))
    agg = metrics.groupby("method").mean(numeric_only=True)
    x = np.arange(len(agg.index))
    width = 0.35
    ax.bar(x - width / 2, agg["cite_p"], width=width, label="Cite-P")
    ax.bar(x + width / 2, agg["cite_r"], width=width, label="Cite-R")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("Score")
    ax.set_title("Citation Precision/Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_citations.png")
    plt.close(fig)

    # Latency breakdown
    fig, ax = plt.subplots(figsize=(6, 4))
    lat = metrics.groupby("method")["latency_sec"].mean()
    ax.bar(lat.index, lat.values)
    ax.set_ylabel("Seconds")
    ax.set_title("Average Latency")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_latency.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
