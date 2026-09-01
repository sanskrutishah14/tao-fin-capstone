"""
Hybrid retriever: FAISS (semantic) + BM25 (keyword) fused via
Reciprocal Rank Fusion (RRF), with a cross-encoder reranking pass.

RRF is a simple, effective rank-level fusion that doesn't require
calibrating or normalizing scores across two very different scoring
systems (cosine similarity vs. BM25 term-frequency scores):

    RRF_score(d) = sum over each retriever i of  1 / (k + rank_i(d))

where k is a smoothing constant (default 60) and rank_i(d) is the
rank of document d in retrieval system i (0 if it didn't appear).

Pipeline:
    query -> FAISS top-N + BM25 top-N -> RRF merge -> rerank -> top-K
"""

from collections import defaultdict
from typing import Dict, List, Optional

from app.services.bm25_index import BM25Index
from app.services.embeddings import EmbeddingService
from app.services.reranker import Reranker
from app.services.vector_store import FAISSVectorStore


class HybridRetriever:
    """
    Combines FAISS semantic search and BM25 keyword search via RRF,
    then (optionally) reranks the fused shortlist with a cross-encoder.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_index: BM25Index,
        embedding_service: EmbeddingService,
        reranker: Optional[Reranker] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        faiss_candidates: int = 20,
        bm25_candidates: int = 20,
        use_reranker: bool = True,
    ) -> List[Dict]:
        """
        Full hybrid retrieval pipeline:

        1. FAISS semantic search  -> faiss_candidates
        2. BM25 keyword search    -> bm25_candidates
        3. RRF fusion of the two
        4. Cross-encoder rerank of the fused shortlist (if enabled)
        5. Return top_k, in the same {"score":, "document":} shape
           the rest of the app (rag_pipeline.build_prompt, etc.)
           already expects.
        """
        query_embedding = self.embedding_service.embed_query(question)
        faiss_results = self.vector_store.search(query_embedding, top_k=faiss_candidates)

        bm25_results = self.bm25_index.search(question, top_k=bm25_candidates)

        fused = self._rrf_fuse(faiss_results, bm25_results)

        # Rerank a somewhat wider pool than top_k so the cross-encoder
        # actually gets to correct FAISS/BM25 mistakes, not just
        # reorder an already-narrow list.
        rerank_pool = fused[: max(top_k * 3, 15)]

        if use_reranker and self.reranker is not None and rerank_pool:
            return self.reranker.rerank(question, rerank_pool, top_k=top_k)

        return fused[:top_k]

    def _rrf_fuse(
        self,
        faiss_results: List[Dict],
        bm25_results: List[Dict],
    ) -> List[Dict]:
        """Reciprocal Rank Fusion across the FAISS and BM25 result lists."""
        rrf_scores: Dict[int, float] = defaultdict(float)
        lookup: Dict[int, Dict] = {}

        for rank, result in enumerate(faiss_results):
            idx = result["index"]
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank + 1)
            lookup[idx] = result["document"]

        for rank, result in enumerate(bm25_results):
            idx = result["index"]
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank + 1)
            lookup[idx] = result["document"]

        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        return [
            {"score": rrf_scores[idx], "index": idx, "document": lookup[idx]}
            for idx in sorted_indices
        ]