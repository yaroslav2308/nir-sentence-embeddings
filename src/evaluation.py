from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def aggregate_by_label(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Aggregate a similarity score by NLI label."""
    return (
        df.groupby("label")[score_column]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


def compute_delta_entailment_contradiction(df: pd.DataFrame, score_column: str) -> float:
    """Return mean(entailment score) minus mean(contradiction score)."""
    means = df.groupby("label")[score_column].mean()
    if "entailment" not in means or "contradiction" not in means:
        return np.nan
    return float(means["entailment"] - means["contradiction"])


def compute_auc_entailment_vs_contradiction(df: pd.DataFrame, score_column: str) -> float:
    """Compute ROC-AUC for entailment versus contradiction using a similarity score."""
    subset = df[df["label"].isin(["entailment", "contradiction"])].copy()
    if subset["label"].nunique() < 2 or subset.empty:
        return np.nan

    y_true = subset["label"].map({"contradiction": 0, "entailment": 1})
    try:
        return float(roc_auc_score(y_true, subset[score_column]))
    except ValueError:
        return np.nan


def get_high_similarity_contradictions(
    df: pd.DataFrame,
    score_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return contradiction pairs with the highest similarity values."""
    return (
        df[df["label"] == "contradiction"]
        .sort_values(score_column, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_low_similarity_entailments(
    df: pd.DataFrame,
    score_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return entailment pairs with the lowest similarity values."""
    return (
        df[df["label"] == "entailment"]
        .sort_values(score_column, ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )
