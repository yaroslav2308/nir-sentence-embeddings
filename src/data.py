from __future__ import annotations

import re

import pandas as pd


def _extract_label_names(features: object) -> list[str] | None:
    """Extract class names from a Hugging Face features object when available."""
    label_feature = None

    if hasattr(features, "__getitem__"):
        try:
            label_feature = features["label"]
        except (KeyError, TypeError):
            label_feature = None

    names = getattr(label_feature, "names", None)
    if names:
        return list(names)
    return None


def _normalize_label(value: object, label_names: list[str] | None) -> str:
    """Convert a dataset label value to a readable class name."""
    if isinstance(value, str):
        return value

    if label_names is not None:
        try:
            return label_names[int(value)]
        except (ValueError, TypeError, IndexError):
            pass

    fallback = {0: "entailment", 1: "neutral", 2: "contradiction"}
    try:
        return fallback[int(value)]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Cannot normalize label value: {value!r}") from exc


def load_xnli_ru(split: str = "validation") -> pd.DataFrame:
    """Load the Russian part of XNLI and return premise, hypothesis, label columns."""
    from datasets import load_dataset

    dataset = load_dataset("xnli", "ru", split=split)
    label_names = _extract_label_names(dataset.features)

    df = dataset.to_pandas()
    required_columns = ["premise", "hypothesis", "label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"XNLI dataset is missing columns: {missing_columns}")

    result = df[required_columns].copy()
    result["label"] = result["label"].apply(lambda value: _normalize_label(value, label_names))
    return result


def sample_balanced(
    df: pd.DataFrame,
    n_per_class: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample up to n_per_class examples for each label without failing on small classes."""
    parts = []
    for _, group in df.groupby("label", sort=True):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=random_state))

    if not parts:
        return df.copy()

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def _token_set(text: object) -> set[str]:
    """Return a simple lower-case token set for lexical overlap calculation."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _jaccard_similarity(left: object, right: object) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple length and lexical overlap features for sentence pairs."""
    result = df.copy()
    result["premise_len_words"] = result["premise"].fillna("").astype(str).str.split().str.len()
    result["hypothesis_len_words"] = (
        result["hypothesis"].fillna("").astype(str).str.split().str.len()
    )
    result["lexical_overlap"] = [
        _jaccard_similarity(premise, hypothesis)
        for premise, hypothesis in zip(result["premise"], result["hypothesis"])
    ]
    return result
