"""Embedding and reranking, using Hugging Face's hosted API by default.

EMBED_BACKEND=hf_api   -> huggingface_hub InferenceClient (free tier, rate limited, needs HF_TOKEN)
EMBED_BACKEND=local    -> sentence-transformers on your machine
RERANK_BACKEND=local   -> small cross-encoder on CPU (default; free hosted rerankers are unreliable)
RERANK_BACKEND=hf_api  -> hosted cross-encoder via the text-classification task
"""
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from .config import get_settings


@lru_cache
def _hf_client():
    from huggingface_hub import InferenceClient

    s = get_settings()
    kwargs = {"token": s.hf_token or None}
    if s.hf_provider:
        kwargs["provider"] = s.hf_provider
    return InferenceClient(**kwargs)


@lru_cache
def _local_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model)


@lru_cache
def _local_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().rerank_model)


def _retry(fn, attempts: int = 5):
    """Free tiers rate-limit and cold-start models; back off and retry."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # Bad token / unknown model will not fix itself: fail immediately with the real error.
            if status in (400, 401, 403, 404) or i == attempts - 1:
                raise
            time.sleep(2 ** i)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim > 1:  # token-level output: use [CLS]-less mean as a fallback
        v = v[0] if v.shape[0] == 1 else v.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n else v


def _hf_embed_one(text: str) -> np.ndarray:
    s = get_settings()
    out = _retry(lambda: _hf_client().feature_extraction(text, model=s.embed_model))
    return _normalize(np.array(out))


# bge models expect this prefix on queries (not on passages) for retrieval.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_passages(texts: list[str]) -> np.ndarray:
    s = get_settings()
    if s.embed_backend == "local":
        return _local_embedder().encode(texts, normalize_embeddings=True, batch_size=32)
    with ThreadPoolExecutor(max_workers=4) as ex:  # modest parallelism to respect rate limits
        return np.stack(list(ex.map(_hf_embed_one, texts)))


def embed_query(text: str) -> np.ndarray:
    s = get_settings()
    q = _QUERY_PREFIX + text
    if s.embed_backend == "local":
        return _local_embedder().encode(q, normalize_embeddings=True)
    return _hf_embed_one(q)


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    s = get_settings()
    if s.rerank_backend == "hf_api":
        def one(p: str) -> float:
            res = _retry(lambda: _hf_client().text_classification(f"{query} [SEP] {p}", model=s.rerank_model))
            return float(res[0].score)

        with ThreadPoolExecutor(max_workers=4) as ex:
            return list(ex.map(one, passages))
    scores = _local_reranker().predict([(query, p) for p in passages])
    return [float(x) for x in scores]
