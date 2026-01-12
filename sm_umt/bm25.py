"""
BM25 implementation for SM-UMT.
Used for re-ranking in-context example candidates.
"""

import math
from typing import List, Tuple
from collections import Counter


class BM25:
    """
    BM25 (Best Matching 25) ranking algorithm.
    
    Used in the TopK+BM25 selection strategy to re-rank
    candidate ICL examples based on lexical similarity.
    """
    
    def __init__(
        self,
        corpus: List[str] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25.
        
        Args:
            corpus: List of documents to index
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        
        self.corpus = []
        self.tokenized_corpus = []
        self.doc_lens = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.idf = {}
        self.N = 0
        
        if corpus:
            self.fit(corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()
    
    def fit(self, corpus: List[str]):
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            corpus: List of documents
        """
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.N = len(corpus)
        
        # Calculate document lengths
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0
        
        # Calculate document frequencies
        self.doc_freqs = Counter()
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            self.doc_freqs.update(unique_terms)
        
        # Calculate IDF for each term
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # IDF with smoothing
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def _score_document(
        self,
        query_terms: List[str],
        doc_idx: int
    ) -> float:
        """
        Calculate BM25 score for a single document.
        
        Args:
            query_terms: Tokenized query
            doc_idx: Index of document in corpus
        
        Returns:
            BM25 score
        """
        doc = self.tokenized_corpus[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        term_freqs = Counter(doc)
        
        for term in query_terms:
            if term not in self.idf:
                continue
            
            tf = term_freqs.get(term, 0)
            idf = self.idf[term]
            
            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def score(self, query: str, document: str) -> float:
        """
        Calculate BM25 score between query and document.
        
        Args:
            query: Query string
            document: Document string
        
        Returns:
            BM25 score
        """
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(document)
        
        # Create a mini-corpus with just this document
        doc_len = len(doc_terms)
        term_freqs = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                # Use a default IDF for unseen terms
                idf = math.log((self.N + 1) / 1)
            else:
                idf = self.idf[term]
            
            tf = term_freqs.get(term, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            if denominator > 0:
                score += idf * (numerator / denominator)
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents in corpus.
        
        Args:
            query: Query string
        
        Returns:
            List of scores for each document
        """
        query_terms = self._tokenize(query)
        scores = [
            self._score_document(query_terms, i) 
            for i in range(self.N)
        ]
        return scores
    
    def get_top_k(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top-K documents by BM25 score.
        
        Args:
            query: Query string
            k: Number of documents to return
        
        Returns:
            List of (doc_index, score) tuples
        """
        scores = self.get_scores(query)
        
        # Get indices sorted by score (descending)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:k]
    
    def rank_documents(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rank a list of documents by BM25 score.
        
        Args:
            query: Query string
            documents: List of documents to rank
        
        Returns:
            List of (document, score) tuples sorted by score
        """
        # Temporarily fit on these documents
        original_corpus = self.corpus
        original_N = self.N
        
        self.fit(documents)
        scores = self.get_scores(query)
        
        # Restore original corpus
        self.corpus = original_corpus
        self.N = original_N
        
        # Pair documents with scores and sort
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked


def select_with_bm25(
    query: str,
    candidates: List[str],
    k: int = 8,
    k1: float = 1.5,
    b: float = 0.75
) -> List[str]:
    """
    Select top-K candidates using BM25 ranking.
    
    Args:
        query: Query string
        candidates: List of candidate strings
        k: Number of candidates to select
        k1: BM25 k1 parameter
        b: BM25 b parameter
    
    Returns:
        Top-K candidates by BM25 score
    """
    bm25 = BM25(corpus=candidates, k1=k1, b=b)
    top_k = bm25.get_top_k(query, k=k)
    
    return [candidates[idx] for idx, _ in top_k]
