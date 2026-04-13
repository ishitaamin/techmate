# backend/database/vector_store.py
import os
import json
import logging
import asyncio
import numpy as np
import faiss
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger("techmate.vector_store")

class VectorStore:
    def __init__(self, index_path: str = "data/faiss.index", chunks_path: str = "data/faiss_chunks.json"):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.vector_index = None
        self.chunk_texts: List[str] = []
        
        # Lazy loaders for our models
        self._embedding_model = None 
        self._reranker_model = None
        self._load_from_disk()

    @property
    def model(self):
        """Lazy loader for the fast Bi-encoder (Retrieval)."""
        if self._embedding_model is None:
            logger.info("Loading SentenceTransformer (Retrieval) model...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    @property
    def reranker(self):
        """Lazy loader for the accurate Cross-Encoder (Re-ranking)."""
        if self._reranker_model is None:
            logger.info("Loading CrossEncoder (Re-ranking) model...")
            # ms-marco is specifically trained for search and Q&A relevance
            self._reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker_model

    def _load_from_disk(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            try:
                self.vector_index = faiss.read_index(self.index_path)
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunk_texts = json.load(f)
                logger.info(f"Loaded FAISS index with {len(self.chunk_texts)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_index = None
                self.chunk_texts = []

    def save_to_disk(self):
        if self.vector_index is None:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        try:
            faiss.write_index(self.vector_index, self.index_path)
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunk_texts, f, ensure_ascii=False, indent=2)
            logger.info("Successfully saved FAISS index to disk.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def add_texts(self, chunks: List[str], persist: bool = True):
        if not chunks: return
        try:
            embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            embeddings = np.asarray(embeddings).astype("float32")
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, axis=0)
            
            dim = embeddings.shape[1]
            if self.vector_index is None:
                self.vector_index = faiss.IndexFlatL2(dim)
                logger.info(f"Created new FAISS index (dim={dim}).")

            self.vector_index.add(embeddings)
            self.chunk_texts.extend(chunks)
            if persist: self.save_to_disk()
        except Exception as e:
            logger.error(f"Failed to add texts to vector store: {e}")

    async def add_texts_async(self, chunks: List[str], persist: bool = True):
        await asyncio.to_thread(self.add_texts, chunks, persist)

    def search_and_rerank(self, query: str, retrieve_top_k: int = 10, final_top_k: int = 3) -> List[str]:
        """Retrieves a wide net of chunks, then uses a CrossEncoder to keep only the absolute best."""
        if self.vector_index is None or not self.chunk_texts:
            return []

        try:
            # 1. RETRIEVE: Get top 10 fast matches
            q_emb = self.model.encode([query], convert_to_numpy=True)
            q_emb = np.asarray(q_emb).astype("float32")
            if q_emb.ndim == 1: q_emb = np.expand_dims(q_emb, axis=0)

            distances, indices = self.vector_index.search(q_emb, retrieve_top_k)
            
            retrieved_chunks = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunk_texts):
                    retrieved_chunks.append(self.chunk_texts[idx])

            if not retrieved_chunks:
                return []

            # 2. RE-RANK: Score the query against each retrieved chunk
            # Create pairs: [["query", "chunk1"], ["query", "chunk2"], ...]
            cross_inp = [[query, chunk] for chunk in retrieved_chunks]
            scores = self.reranker.predict(cross_inp)

            # Sort chunks by their CrossEncoder score (highest first)
            scored_chunks = list(zip(scores, retrieved_chunks))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            # Return only the top N chunks
            best_chunks = [chunk for score, chunk in scored_chunks[:final_top_k]]
            logger.info(f"Re-ranked top {retrieve_top_k} down to {final_top_k} high-quality chunks.")
            return best_chunks

        except Exception as e:
            logger.error(f"FAISS search/rerank failed: {e}")
            return []
            
    async def search_and_rerank_async(self, query: str, retrieve_top_k: int = 10, final_top_k: int = 3) -> List[str]:
        return await asyncio.to_thread(self.search_and_rerank, query, retrieve_top_k, final_top_k)