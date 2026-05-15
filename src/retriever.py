from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import logging
from sentence_transformers import CrossEncoder
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "DiTy/cross-encoder-russian-msmarco"):
        self.model = CrossEncoder(
            model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Reranker {model_name} загружен")

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        # Сортируем по убыванию score
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


class AdvancedHybridRetriever:
    def __init__(self, vectorstore, documents: List[Document], reranker=None):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = BM25Okapi([doc.page_content.split() for doc in documents])
        self.reranker = reranker or Reranker()

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20

    def _vector_search(self, query: str, k: int = 20) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int = 20) -> List[Document]:
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_idx]

    def _reciprocal_rank_fusion(self, results: List[List[Document]], k: int = 60) -> List[Document]:
        """RRF fusion"""
        doc_score = {}
        for rank_list in results:
            for rank, doc in enumerate(rank_list):
                doc_id = id(doc)
                if doc_id not in doc_score:
                    doc_score[doc_id] = 0
                doc_score[doc_id] += 1.0 / (rank + 60)

        sorted_docs = sorted(
            [(d, doc_score[id(d)]) for d_list in results for d in d_list],
            key=lambda x: x[1], reverse=True
        )
        unique = {}
        for doc, score in sorted_docs:
            doc_key = id(doc)
            if doc_key not in unique:
                unique[doc_key] = doc
        return list(unique.values())[:k]

    def retrieve(self, query: str, k: int = 5, top_k_candidates: int = 20) -> List[Document]:
        logger.info(f"Retrieve для запроса: {query[:100]}...")

        vector_docs = self._vector_search(query, top_k_candidates)
        bm25_docs = self._bm25_search(query, top_k_candidates)

        candidates = self._reciprocal_rank_fusion(
            [vector_docs, bm25_docs], k=top_k_candidates * 2)

        reranked = self.reranker.rerank(query, candidates, top_k=k)

        logger.info(f"Возвращено {len(reranked)} чанков после rerank")
        return reranked
