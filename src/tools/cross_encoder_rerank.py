import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List

from langchain.schema import Document


def rerank_with_cross_encoder(
    query: str, passages: List[Document], n_similar_docs: int
) -> List[Document]:

    model = CrossEncoder("BAAI/bge-reranker-large")
    scores = model.predict([(query, p.page_content) for p in passages])
    # Convert to [0,1] so we might could use a cutoff later.
    scores = 1.0 / (1 + np.exp(-np.array(scores)))
    # get top k scoring passages
    sorted_pairs = sorted(
        zip(scores, passages),
        key=lambda x: x[0],
        reverse=True,
    )
    passages = [
        d for _, d in sorted_pairs[: n_similar_docs]
    ]
    return passages

