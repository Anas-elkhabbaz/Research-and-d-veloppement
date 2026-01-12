"""
Configuration module for SM-UMT.
Contains all hyperparameters from the paper.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """
    Configuration class with hyperparameters from the paper.
    
    Key hyperparameters:
    - kwp: Number of word pairs for word-level ICL (default: 10)
    - k: Number of sentence-level ICL examples (default: 8)
    - tau: Similarity threshold for filtering (default: 0.90)
    - top_n: Top N candidates before BM25 selection (default: 20)
    """
    
    # Word-level mining parameters
    kwp: int = 10  # Number of word translations to generate per source word
    
    # Sentence-level mining parameters
    k: int = 8  # Number of ICL examples for sentence translation
    tau: float = 0.90  # Similarity threshold for filtering
    top_n: int = 20  # Top N candidates before BM25 selection
    
    # Language settings
    src_lang: str = "fra"  # Source language code (ISO 639-3)
    tgt_lang: str = "eng"  # Target language code (ISO 639-3)
    
    # Supported language pairs
    supported_langs: dict = field(default_factory=lambda: {
        "fra": "French",
        "eng": "English", 
        "arb": "Arabic"
    })
    
    # LLM settings (Gemini) - using 2.5-flash which has separate quota
    llm_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 256
    
    # Embedding model (for sentence similarity)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    # Evaluation
    batch_size: int = 8
    max_samples: int = 100  # For quick testing
    
    def get_lang_name(self, code: str) -> str:
        """Get full language name from ISO code."""
        return self.supported_langs.get(code, code)
    
    def validate(self):
        """Validate configuration."""
        assert self.kwp > 0, "kwp must be positive"
        assert self.k > 0, "k must be positive"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"
        assert self.top_n >= self.k, "top_n must be >= k"
        assert self.src_lang in self.supported_langs, f"Unsupported source language: {self.src_lang}"
        assert self.tgt_lang in self.supported_langs, f"Unsupported target language: {self.tgt_lang}"


# Pre-configured language pairs
LANGUAGE_PAIRS = [
    ("fra", "eng"),  # French -> English
    ("eng", "fra"),  # English -> French
    ("arb", "eng"),  # Arabic -> English
    ("eng", "arb"),  # English -> Arabic
]
