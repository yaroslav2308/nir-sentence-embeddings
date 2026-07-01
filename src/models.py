from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Recommended sentence embedding models for experiments:
# - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: fast multilingual model.
# - sentence-transformers/paraphrase-multilingual-mpnet-base-v2: heavier multilingual model.
# - intfloat/multilingual-e5-base: additional model, if needed.
# - ai-forever/ru-en-RoSBERTa: Russian-oriented model, if it runs correctly through
#   sentence-transformers or transformers in the current environment.


def cosine_sim_matrix(a: np.ndarray | sparse.spmatrix, b: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Compute a cosine similarity matrix for two embedding matrices."""
    return cosine_similarity(a, b)


def compute_tfidf_similarity(df: pd.DataFrame) -> pd.Series:
    """Compute pairwise TF-IDF cosine similarity for premise-hypothesis pairs."""
    premises = df["premise"].fillna("").astype(str).tolist()
    hypotheses = df["hypothesis"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(premises + hypotheses)

    premise_vectors = vectorizer.transform(premises)
    hypothesis_vectors = vectorizer.transform(hypotheses)
    similarities = cosine_similarity(premise_vectors, hypothesis_vectors).diagonal()

    return pd.Series(similarities, index=df.index, name="tfidf_similarity")


def compute_sentence_transformer_similarity(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int = 32,
) -> pd.Series:
    """Compute pairwise cosine similarity using a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    premises = df["premise"].fillna("").astype(str).tolist()
    hypotheses = df["hypothesis"].fillna("").astype(str).tolist()

    model = SentenceTransformer(model_name)
    premise_embeddings = model.encode(
        premises,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    hypothesis_embeddings = model.encode(
        hypotheses,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarities = np.sum(premise_embeddings * hypothesis_embeddings, axis=1)
    return pd.Series(similarities, index=df.index, name=f"st_similarity__{model_name}")
