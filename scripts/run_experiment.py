from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import add_text_stats, load_xnli_ru, sample_balanced
from src.evaluation import (
    aggregate_by_label,
    compute_auc_entailment_vs_contradiction,
    compute_delta_entailment_contradiction,
    get_high_similarity_contradictions,
)
from src.models import compute_sentence_transformer_similarity, compute_tfidf_similarity
from src.visualization import plot_similarity_by_label


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Эксперимент по оценке similarity для русскоязычных NLI-пар."
    )
    parser.add_argument("--n-per-class", type=int, default=300, help="Число примеров на класс.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size для модели.")
    parser.add_argument("--split", type=str, default="validation", help="Split XNLI.")
    return parser.parse_args()


def print_metric_report(df: pd.DataFrame, score_column: str) -> None:
    print(f"\nАгрегация по классам для {score_column}:")
    print(aggregate_by_label(df, score_column).to_string(index=False))

    delta = compute_delta_entailment_contradiction(df, score_column)
    auc = compute_auc_entailment_vs_contradiction(df, score_column)
    print(f"Delta entailment-contradiction для {score_column}: {delta:.4f}")
    print(f"ROC-AUC entailment vs contradiction для {score_column}: {auc:.4f}")


def main() -> None:
    args = parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print(f"Загружаем XNLI Russian, split={args.split!r}...")
    df = load_xnli_ru(split=args.split)
    print(f"Загружено строк: {len(df)}")

    print(f"Берем balanced sample: до {args.n_per_class} примеров на класс...")
    df = sample_balanced(df, n_per_class=args.n_per_class)
    print("Распределение классов:")
    print(df["label"].value_counts().to_string())

    print("Добавляем простые текстовые признаки...")
    df = add_text_stats(df)

    print("Считаем TF-IDF cosine similarity...")
    df["tfidf_similarity"] = compute_tfidf_similarity(df)

    print(f"Считаем SentenceTransformer cosine similarity: {args.model}")
    df["st_similarity"] = compute_sentence_transformer_similarity(
        df,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    print_metric_report(df, "tfidf_similarity")
    print_metric_report(df, "st_similarity")

    results_path = output_dir / "results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nCSV с результатами сохранен: {results_path}")

    figure_path = output_dir / "similarity_by_label.png"
    fig = plot_similarity_by_label(df, "st_similarity", output_path=str(figure_path))
    fig.clf()
    print(f"График сохранен: {figure_path}")

    errors_path = output_dir / "high_similarity_contradictions.csv"
    high_similarity_contradictions = get_high_similarity_contradictions(df, "st_similarity", top_n=20)
    high_similarity_contradictions.to_csv(errors_path, index=False)
    print(f"Top contradiction-пары с высокой similarity сохранены: {errors_path}")


if __name__ == "__main__":
    main()
