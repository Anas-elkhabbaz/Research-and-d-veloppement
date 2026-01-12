"""
Word-level mining module for SM-UMT.
Extracts words and generates word-level translations using LLM.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import Counter
from tqdm import tqdm

from .prompts import (
    create_word_translation_prompt,
    create_word_extraction_prompt,
    create_batch_word_translation_prompt
)


class WordMiner:
    """
    Word-level mining for generating synthetic parallel data.
    
    This is Stage 1 of the SM-UMT pipeline:
    1. Extract key content words from source sentences
    2. Translate words using LLM with ICL
    3. Create synthetic word-by-word parallel data
    """
    
    def __init__(self, llm_client, config):
        """
        Initialize WordMiner.
        
        Args:
            llm_client: LLM client for generating translations
            config: Configuration object
        """
        self.llm_client = llm_client
        self.config = config
        self.word_cache = {}  # Cache for word translations
    
    def extract_words_simple(self, sentences: List[str]) -> List[str]:
        """
        Extract unique words from sentences using simple tokenization.
        
        Args:
            sentences: List of source sentences
        
        Returns:
            List of unique words sorted by frequency
        """
        word_counter = Counter()
        
        for sentence in sentences:
            # Simple tokenization: split on whitespace and punctuation
            words = re.findall(r'\b\w+\b', sentence.lower())
            word_counter.update(words)
        
        # Filter out very short words and return sorted by frequency
        words = [w for w, c in word_counter.most_common() if len(w) > 2]
        return words
    
    def extract_words_llm(
        self,
        sentences: List[str],
        src_lang: str,
        max_words: int = 100
    ) -> List[str]:
        """
        Extract key content words using LLM.
        
        Args:
            sentences: List of source sentences
            src_lang: Source language name
            max_words: Maximum number of words to extract
        
        Returns:
            List of unique content words
        """
        all_words = set()
        
        for sentence in tqdm(sentences[:min(len(sentences), 20)], desc="Extracting words"):
            prompt = create_word_extraction_prompt(sentence, src_lang)
            response = self.llm_client.generate(prompt)
            
            # Parse comma-separated words
            words = [w.strip().lower() for w in response.split(',')]
            words = [w for w in words if w and len(w) > 2]
            all_words.update(words)
            
            if len(all_words) >= max_words:
                break
        
        return list(all_words)[:max_words]
    
    def translate_word(
        self,
        word: str,
        src_lang: str,
        tgt_lang: str,
        examples: List[Tuple[str, str]] = None
    ) -> str:
        """
        Translate a single word using LLM.
        
        Args:
            word: Source word
            src_lang: Source language name
            tgt_lang: Target language name
            examples: Optional ICL examples
        
        Returns:
            Translated word
        """
        # Check cache first
        cache_key = (word, src_lang, tgt_lang)
        if cache_key in self.word_cache:
            return self.word_cache[cache_key]
        
        prompt = create_word_translation_prompt(word, src_lang, tgt_lang, examples)
        translation = self.llm_client.generate(prompt)
        
        # Clean up translation
        translation = translation.strip().lower()
        # Remove any extra text after the first word
        translation = translation.split()[0] if translation else word
        
        # Cache and return
        self.word_cache[cache_key] = translation
        return translation
    
    def translate_words_batch(
        self,
        words: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 10
    ) -> Dict[str, str]:
        """
        Translate multiple words in batches.
        
        Args:
            words: List of source words
            src_lang: Source language name
            tgt_lang: Target language name
            batch_size: Number of words per batch
        
        Returns:
            Dictionary mapping source words to translations
        """
        translations = {}
        
        for i in tqdm(range(0, len(words), batch_size), desc="Translating words"):
            batch = words[i:i+batch_size]
            
            prompt = create_batch_word_translation_prompt(batch, src_lang, tgt_lang)
            response = self.llm_client.generate(prompt)
            
            # Parse numbered responses
            lines = response.strip().split('\n')
            for j, line in enumerate(lines):
                if j < len(batch):
                    # Extract translation after the number
                    match = re.search(r'\d+\.\s*(.+)', line)
                    if match:
                        translation = match.group(1).strip()
                    else:
                        translation = line.strip()
                    
                    translations[batch[j]] = translation
                    self.word_cache[(batch[j], src_lang, tgt_lang)] = translation
        
        return translations
    
    def mine_word_translations(
        self,
        sentences: List[str],
        src_lang: str,
        tgt_lang: str,
        use_llm_extraction: bool = False,
        max_words: int = 100
    ) -> Dict[str, str]:
        """
        Main method: Extract words and generate translations.
        
        Args:
            sentences: List of source sentences
            src_lang: Source language name
            tgt_lang: Target language name
            use_llm_extraction: Whether to use LLM for word extraction
            max_words: Maximum words to translate
        
        Returns:
            Dictionary mapping source words to target translations
        """
        print(f"Mining word translations: {src_lang} -> {tgt_lang}")
        
        # Step 1: Extract words
        if use_llm_extraction:
            words = self.extract_words_llm(sentences, src_lang, max_words)
        else:
            words = self.extract_words_simple(sentences)[:max_words]
        
        print(f"Extracted {len(words)} unique words")
        
        # Step 2: Translate words in batches
        translations = self.translate_words_batch(
            words, src_lang, tgt_lang, 
            batch_size=self.config.kwp
        )
        
        print(f"Generated {len(translations)} word translations")
        
        return translations
    
    def create_synthetic_parallel_sentence(
        self,
        sentence: str,
        word_translations: Dict[str, str]
    ) -> str:
        """
        Create a synthetic parallel sentence using word translations.
        
        This is a simple word-by-word replacement approach.
        For better quality, use the LLM-based approach.
        
        Args:
            sentence: Source sentence
            word_translations: Word translation dictionary
        
        Returns:
            Synthetic target sentence
        """
        words = re.findall(r'\b\w+\b', sentence.lower())
        translated_words = []
        
        for word in words:
            if word in word_translations:
                translated_words.append(word_translations[word])
            else:
                translated_words.append(word)  # Keep original if no translation
        
        return ' '.join(translated_words)
    
    def generate_synthetic_parallel_data(
        self,
        sentences: List[str],
        word_translations: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """
        Generate synthetic parallel data for all sentences.
        
        Args:
            sentences: List of source sentences
            word_translations: Word translation dictionary
        
        Returns:
            List of (source, synthetic_target) tuples
        """
        parallel_data = []
        
        for sentence in tqdm(sentences, desc="Creating synthetic parallel data"):
            synthetic = self.create_synthetic_parallel_sentence(sentence, word_translations)
            parallel_data.append((sentence, synthetic))
        
        return parallel_data


def mine_words(
    sentences: List[str],
    src_lang: str,
    tgt_lang: str,
    llm_client,
    config
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Convenience function for word-level mining.
    
    Args:
        sentences: List of source sentences
        src_lang: Source language name
        tgt_lang: Target language name
        llm_client: LLM client
        config: Configuration
    
    Returns:
        Tuple of (word_translations, synthetic_parallel_data)
    """
    miner = WordMiner(llm_client, config)
    
    # Mine word translations
    word_translations = miner.mine_word_translations(
        sentences, src_lang, tgt_lang,
        max_words=config.kwp * 10  # Get enough words
    )
    
    # Generate synthetic parallel data
    parallel_data = miner.generate_synthetic_parallel_data(
        sentences, word_translations
    )
    
    return word_translations, parallel_data
