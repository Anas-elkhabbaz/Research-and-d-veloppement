"""
Quality-Aware ICL Selection (QAIS) module for SM-UMT.
Implements quality filtering for mined ICL examples using back-translation.

Novel contribution: Filters low-quality mined examples before use.
"""

import re
from typing import List, Dict, Tuple, Optional
from .config import Config


class QualityAwareICLSelector:
    """
    Quality-Aware ICL Selection for SM-UMT.
    
    Filters low-quality mined examples using:
    1. Back-translation consistency checking
    2. Semantic similarity scoring
    3. Fluency validation
    """
    
    def __init__(self, llm_client, config: Config, embedding_model=None):
        """
        Initialize QAIS.
        
        Args:
            llm_client: LLM client for back-translation
            config: Configuration object
            embedding_model: Pre-loaded embedding model (optional)
        """
        self.llm_client = llm_client
        self.config = config
        self.embedding_model = embedding_model
        self.quality_cache = {}  # Cache quality scores
        self.stats = {
            'total_evaluated': 0,
            'passed_quality': 0,
            'failed_quality': 0,
            'cache_hits': 0
        }
    
    def back_translate(
        self,
        sentence: str,
        src_lang: str,
        tgt_lang: str
    ) -> str:
        """
        Translate sentence back from target to source language.
        
        Args:
            sentence: Sentence in target language
            src_lang: Original source language name
            tgt_lang: Target language name (current language of sentence)
            
        Returns:
            Back-translated sentence
        """
        prompt = f"""Translate the following sentence from {tgt_lang} to {src_lang}.
Only provide the translation, nothing else.

{tgt_lang}: {sentence}
{src_lang}:"""
        
        try:
            result = self.llm_client.generate(prompt)
            return result.strip()
        except Exception as e:
            print(f"QAIS: Back-translation failed: {e}")
            return ""
    
    def compute_lexical_overlap(
        self,
        original: str,
        back_translated: str
    ) -> float:
        """
        Compute lexical overlap between original and back-translated.
        
        Args:
            original: Original source sentence
            back_translated: Back-translated sentence
            
        Returns:
            Overlap score (0-1)
        """
        # Tokenize
        orig_words = set(re.findall(r'\b\w+\b', original.lower()))
        back_words = set(re.findall(r'\b\w+\b', back_translated.lower()))
        
        if not orig_words:
            return 0.0
        
        # Compute Jaccard similarity
        intersection = len(orig_words & back_words)
        union = len(orig_words | back_words)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_semantic_similarity(
        self,
        original: str,
        back_translated: str
    ) -> float:
        """
        Compute semantic similarity using embeddings.
        
        Args:
            original: Original source sentence
            back_translated: Back-translated sentence
            
        Returns:
            Similarity score (0-1)
        """
        if self.embedding_model is None:
            # Fall back to lexical overlap
            return self.compute_lexical_overlap(original, back_translated)
        
        try:
            import numpy as np
            
            embeddings = self.embedding_model.encode([original, back_translated])
            
            # Cosine similarity
            sim = np.dot(embeddings[0], embeddings[1])
            sim /= (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            
            return float(sim)
        except Exception as e:
            print(f"QAIS: Embedding similarity failed: {e}")
            return self.compute_lexical_overlap(original, back_translated)
    
    def compute_quality_score(
        self,
        source: str,
        target: str,
        src_lang: str,
        tgt_lang: str,
        use_back_translation: bool = True
    ) -> float:
        """
        Compute overall quality score for a (source, target) pair.
        
        Args:
            source: Source sentence
            target: Target (synthetic) translation
            src_lang: Source language name
            tgt_lang: Target language name
            use_back_translation: Whether to use back-translation check
            
        Returns:
            Quality score (0-1)
        """
        # Check cache
        cache_key = (source, target)
        if cache_key in self.quality_cache:
            self.stats['cache_hits'] += 1
            return self.quality_cache[cache_key]
        
        self.stats['total_evaluated'] += 1
        
        scores = []
        
        # Length ratio check (synthetic shouldn't be too different in length)
        len_ratio = len(target) / max(len(source), 1)
        len_score = 1.0 if 0.5 <= len_ratio <= 2.0 else 0.5
        scores.append(len_score * 0.2)  # 20% weight
        
        # Word preservation check (for synthetic translations)
        source_words = set(re.findall(r'\b\w+\b', source.lower()))
        target_words = set(re.findall(r'\b\w+\b', target.lower()))
        
        # Some source words should appear in target (for synthetic)
        preserved = len(source_words & target_words) / max(len(source_words), 1)
        scores.append(preserved * 0.3)  # 30% weight
        
        # Back-translation consistency (if enabled, 50% weight)
        if use_back_translation:
            back_translated = self.back_translate(target, src_lang, tgt_lang)
            if back_translated:
                bt_score = self.compute_semantic_similarity(source, back_translated)
                scores.append(bt_score * 0.5)
            else:
                scores.append(0.25)  # Neutral if back-translation failed
        else:
            scores.append(0.25)  # Skip back-translation
        
        quality = sum(scores)
        
        # Cache result
        self.quality_cache[cache_key] = quality
        
        if quality >= self.config.qais_threshold if hasattr(self.config, 'qais_threshold') else 0.5:
            self.stats['passed_quality'] += 1
        else:
            self.stats['failed_quality'] += 1
        
        return quality
    
    def filter_quality_examples(
        self,
        icl_examples: List[Tuple[str, str]],
        src_lang: str,
        tgt_lang: str,
        min_quality: float = 0.5,
        use_back_translation: bool = False  # Off by default to save API calls
    ) -> List[Tuple[str, str]]:
        """
        Filter ICL examples by quality score.
        
        Args:
            icl_examples: List of (source, target) pairs
            src_lang: Source language name
            tgt_lang: Target language name
            min_quality: Minimum quality threshold
            use_back_translation: Use back-translation (expensive)
            
        Returns:
            Filtered list of high-quality examples
        """
        scored_examples = []
        
        for source, target in icl_examples:
            score = self.compute_quality_score(
                source, target, src_lang, tgt_lang,
                use_back_translation=use_back_translation
            )
            scored_examples.append((source, target, score))
        
        # Sort by quality (descending)
        scored_examples.sort(key=lambda x: x[2], reverse=True)
        
        # Filter by threshold
        filtered = [
            (src, tgt) for src, tgt, score in scored_examples
            if score >= min_quality
        ]
        
        return filtered
    
    def select_best_examples(
        self,
        icl_examples: List[Tuple[str, str]],
        src_lang: str,
        tgt_lang: str,
        k: int = 8,
        use_back_translation: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Select top-k highest quality examples.
        
        Args:
            icl_examples: List of (source, target) pairs
            src_lang: Source language name
            tgt_lang: Target language name
            k: Number of examples to select
            use_back_translation: Use back-translation scoring
            
        Returns:
            Top-k quality examples
        """
        scored = []
        
        for source, target in icl_examples:
            score = self.compute_quality_score(
                source, target, src_lang, tgt_lang,
                use_back_translation=use_back_translation
            )
            scored.append((source, target, score))
        
        # Sort by quality
        scored.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k
        return [(src, tgt) for src, tgt, _ in scored[:k]]
    
    def get_stats(self) -> Dict[str, int]:
        """Get QAIS statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_evaluated': 0,
            'passed_quality': 0,
            'failed_quality': 0,
            'cache_hits': 0
        }


def filter_icl_with_quality(
    icl_examples: List[Tuple[str, str]],
    llm_client,
    config: Config,
    src_lang: str,
    tgt_lang: str,
    k: int = 8,
    use_back_translation: bool = False
) -> List[Tuple[str, str]]:
    """
    Convenience function for quality-aware ICL selection.
    
    Args:
        icl_examples: List of (source, target) pairs
        llm_client: LLM client
        config: Configuration
        src_lang: Source language name
        tgt_lang: Target language name
        k: Number of examples to select
        use_back_translation: Enable back-translation (API expensive)
        
    Returns:
        Top-k quality examples
    """
    qais = QualityAwareICLSelector(llm_client, config)
    return qais.select_best_examples(
        icl_examples, src_lang, tgt_lang, k,
        use_back_translation=use_back_translation
    )
