"""
Sentence-level mining module for SM-UMT.
Implements TopK+BM25 selection strategy for ICL examples.
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

from .bm25 import BM25, select_with_bm25


class SentenceMiner:
    """
    Sentence-level mining for selecting ICL examples.
    
    This is Stage 2 of the SM-UMT pipeline:
    1. Compute sentence embeddings
    2. TopK similarity selection (top 20)
    3. BM25 re-ranking
    4. Filtering with threshold τ=0.90
    5. Select final k=8 ICL examples
    """
    
    def __init__(self, config, embedding_model=None):
        """
        Initialize SentenceMiner.
        
        Args:
            config: Configuration object
            embedding_model: Pre-loaded embedding model (optional)
        """
        self.config = config
        self.encoder = None
        self._embedding_model = embedding_model
        
        # Cache for embeddings
        self.embedding_cache = {}
    
    def _load_encoder(self):
        """Lazy load the sentence encoder."""
        if self.encoder is None:
            if self._embedding_model is not None:
                self.encoder = self._embedding_model
            else:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.config.embedding_model}")
                self.encoder = SentenceTransformer(self.config.embedding_model)
    
    def compute_embeddings(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute sentence embeddings.
        
        Args:
            sentences: List of sentences
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings (N x D)
        """
        self._load_encoder()
        
        embeddings = self.encoder.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def compute_similarity(
        self,
        query_emb: np.ndarray,
        corpus_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus.
        
        Args:
            query_emb: Query embedding (D,) or (1, D)
            corpus_embs: Corpus embeddings (N, D)
        
        Returns:
            Similarity scores (N,)
        """
        # Handle empty corpus
        if corpus_embs is None or len(corpus_embs) == 0:
            return np.array([])
        
        # Ensure 2D arrays
        query_emb = np.atleast_2d(query_emb)
        corpus_embs = np.atleast_2d(corpus_embs)
        
        # Handle single embedding case
        if corpus_embs.ndim == 1:
            corpus_embs = corpus_embs.reshape(1, -1)
        
        # Normalize embeddings (with safety for zero vectors)
        query_norm_val = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_norm_val = np.where(query_norm_val == 0, 1, query_norm_val)
        query_norm = query_emb / query_norm_val
        
        corpus_norm_val = np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        corpus_norm_val = np.where(corpus_norm_val == 0, 1, corpus_norm_val)
        corpus_norm = corpus_embs / corpus_norm_val
        
        # Compute cosine similarity
        similarities = np.dot(corpus_norm, query_norm.T).flatten()
        
        return similarities
    
    def topk_similarity_selection(
        self,
        query_emb: np.ndarray,
        corpus_embs: np.ndarray,
        corpus: List[str],
        k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Select top-K sentences by embedding similarity.
        
        Args:
            query_emb: Query sentence embedding
            corpus_embs: Corpus sentence embeddings
            corpus: List of corpus sentences
            k: Number of top candidates to select
        
        Returns:
            List of (sentence, similarity_score) tuples
        """
        similarities = self.compute_similarity(query_emb, corpus_embs)
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (corpus[i], similarities[i])
            for i in top_indices
        ]
        
        return results
    
    def filter_by_threshold(
        self,
        candidates: List[Tuple[str, float]],
        threshold: float = 0.90
    ) -> List[Tuple[str, float]]:
        """
        Filter candidates by similarity threshold.
        
        Args:
            candidates: List of (sentence, score) tuples
            threshold: Minimum similarity score
        
        Returns:
            Filtered list of candidates
        """
        return [
            (sent, score) for sent, score in candidates
            if score >= threshold
        ]
    
    def select_icl_examples(
        self,
        query: str,
        parallel_corpus: List[Tuple[str, str]],
        corpus_embeddings: np.ndarray = None,
        k: int = 8
    ) -> List[Tuple[str, str]]:
        """
        Select ICL examples using TopK+BM25 strategy.
        
        Args:
            query: Query sentence (source language)
            parallel_corpus: List of (source, target) parallel pairs
            corpus_embeddings: Pre-computed source embeddings (optional)
            k: Number of ICL examples to select
        
        Returns:
            List of (source, target) ICL examples
        """
        # Handle empty parallel corpus
        if not parallel_corpus or len(parallel_corpus) == 0:
            return []
        
        source_sentences = [s for s, t in parallel_corpus]
        target_sentences = [t for s, t in parallel_corpus]
        
        # Step 1: Compute embeddings if not provided
        if corpus_embeddings is None:
            corpus_embeddings = self.compute_embeddings(source_sentences)
        
        # Compute query embedding
        query_emb = self.compute_embeddings([query], show_progress=False)[0]
        
        # Step 2: TopK similarity selection (top_n candidates)
        top_candidates = self.topk_similarity_selection(
            query_emb,
            corpus_embeddings,
            source_sentences,
            k=self.config.top_n
        )
        
        # Step 3: Filter by threshold
        filtered_candidates = self.filter_by_threshold(
            top_candidates,
            threshold=self.config.tau
        )
        
        # If no candidates pass threshold, use top candidates without filtering
        if len(filtered_candidates) < k:
            filtered_candidates = top_candidates
        
        # Step 4: BM25 re-ranking
        candidate_sentences = [s for s, _ in filtered_candidates]
        
        if len(candidate_sentences) > k:
            # Use BM25 to select final k examples
            bm25 = BM25(
                corpus=candidate_sentences,
                k1=self.config.bm25_k1,
                b=self.config.bm25_b
            )
            top_k = bm25.get_top_k(query, k=k)
            selected_sources = [candidate_sentences[idx] for idx, _ in top_k]
        else:
            selected_sources = candidate_sentences[:k]
        
        # Step 5: Get corresponding target sentences
        source_to_target = dict(zip(source_sentences, target_sentences))
        icl_examples = [
            (src, source_to_target[src])
            for src in selected_sources
            if src in source_to_target
        ]
        
        return icl_examples
    
    def mine_sentence_examples(
        self,
        test_sentences: List[str],
        parallel_corpus: List[Tuple[str, str]],
        k: int = 8,
        show_progress: bool = True
    ) -> List[List[Tuple[str, str]]]:
        """
        Mine ICL examples for multiple test sentences.
        
        Args:
            test_sentences: List of test sentences to translate
            parallel_corpus: Synthetic parallel corpus
            k: Number of ICL examples per test sentence
            show_progress: Whether to show progress bar
        
        Returns:
            List of ICL example lists (one per test sentence)
        """
        # Pre-compute corpus embeddings
        source_sentences = [s for s, t in parallel_corpus]
        print("Computing corpus embeddings...")
        corpus_embeddings = self.compute_embeddings(source_sentences)
        
        # Mine examples for each test sentence
        all_examples = []
        iterator = tqdm(test_sentences, desc="Mining ICL examples") if show_progress else test_sentences
        
        for sentence in iterator:
            examples = self.select_icl_examples(
                sentence,
                parallel_corpus,
                corpus_embeddings,
                k=k
            )
            all_examples.append(examples)
        
        return all_examples


def mine_icl_examples(
    query: str,
    parallel_corpus: List[Tuple[str, str]],
    config,
    embedding_model=None
) -> List[Tuple[str, str]]:
    """
    Convenience function for mining ICL examples.
    
    Args:
        query: Query sentence
        parallel_corpus: Synthetic parallel corpus
        config: Configuration object
        embedding_model: Optional pre-loaded embedding model
    
    Returns:
        List of (source, target) ICL examples
    """
    miner = SentenceMiner(config, embedding_model)
    
    return miner.select_icl_examples(
        query,
        parallel_corpus,
        k=config.k
    )
