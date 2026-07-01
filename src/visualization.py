from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt


def plot_similarity_by_label(
    df: pd.DataFrame,
    score_column: str,
    output_path: str | None = None,
):
    """Build a boxplot of cosine similarity values grouped by NLI label."""
    label_order = ["entailment", "neutral", "contradiction"]
    labels = [label for label in label_order if label in set(df["label"])]
    labels += [label for label in sorted(set(df["label"])) if label not in labels]

    data = [df.loc[df["label"] == label, score_column].dropna() for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"Распределение cosine similarity по классам ({score_column})")
    ax.set_xlabel("Класс NLI")
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig
