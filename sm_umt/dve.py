"""
Dynamic Vocabulary Expansion (DVE) module for SM-UMT.
Implements on-the-fly word mining for unknown words during translation.

Novel contribution: Addresses the vocabulary gap limitation of static mining.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from .config import Config
from .prompts import create_word_translation_prompt


class DynamicVocabularyExpander:
    """
    Dynamic Vocabulary Expansion for SM-UMT.
    
    Identifies and translates unknown words on-the-fly during translation,
    addressing the limitation where words outside the mined set are dropped.
    """
    
    def __init__(self, llm_client, config: Config):
        """
        Initialize DVE.
        
        Args:
            llm_client: LLM client for translation
            config: Configuration object
        """
        self.llm_client = llm_client
        self.config = config
        self.word_cache = {}  # Cache translations to minimize API calls
        self.stats = {
            'unknown_found': 0,
            'successfully_translated': 0,
            'cache_hits': 0
        }
    
    def find_unknown_words(
        self,
        sentence: str,
        known_words: Dict[str, str]
    ) -> List[str]:
        """
        Identify words in sentence not in the known vocabulary.
        
        Args:
            sentence: Source sentence
            known_words: Dictionary of known word translations
            
        Returns:
            List of unknown words
        """
        # Tokenize sentence
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Find words not in dictionary (also check cache)
        unknown = []
        for word in words:
            if word not in known_words and word not in self.word_cache:
                # Skip very short words (articles, etc.)
                if len(word) > 2:
                    unknown.append(word)
        
        self.stats['unknown_found'] += len(unknown)
        return list(set(unknown))  # Remove duplicates
    
    def translate_unknown_word(
        self,
        word: str,
        context: str,
        src_lang: str,
        tgt_lang: str,
        existing_translations: Dict[str, str] = None
    ) -> Optional[str]:
        """
        Translate a single unknown word with context.
        
        Args:
            word: Word to translate
            context: Original sentence for context
            src_lang: Source language name
            tgt_lang: Target language name
            existing_translations: Existing translations for ICL examples
            
        Returns:
            Translated word or None if failed
        """
        # Check cache first
        if word in self.word_cache:
            self.stats['cache_hits'] += 1
            return self.word_cache[word]
        
        # Build ICL examples from existing translations
        examples = []
        if existing_translations:
            for src, tgt in list(existing_translations.items())[:5]:
                examples.append((src, tgt))
        
        # Create context-aware prompt
        prompt = f"""Translate the {src_lang} word "{word}" to {tgt_lang}.
Context sentence: "{context}"
Only provide the single word translation, nothing else.

"""
        if examples:
            prompt += "Examples:\n"
            for src_w, tgt_w in examples:
                prompt += f"{src_w} → {tgt_w}\n"
            prompt += "\n"
        
        prompt += f"Word: {word}\nTranslation:"
        
        try:
            translation = self.llm_client.generate(prompt)
            translation = translation.strip().lower()
            
            # Clean up response
            translation = re.sub(r'[^\w\s]', '', translation)
            translation = translation.split()[0] if translation.split() else word
            
            # Cache the result
            self.word_cache[word] = translation
            self.stats['successfully_translated'] += 1
            
            return translation
            
        except Exception as e:
            print(f"DVE: Failed to translate '{word}': {e}")
            return None
    
    def expand_vocabulary(
        self,
        sentence: str,
        known_words: Dict[str, str],
        src_lang: str,
        tgt_lang: str
    ) -> Dict[str, str]:
        """
        Expand vocabulary with translations for unknown words.
        
        Args:
            sentence: Source sentence
            known_words: Current word translation dictionary
            src_lang: Source language name
            tgt_lang: Target language name
            
        Returns:
            Updated dictionary with new translations added
        """
        unknown = self.find_unknown_words(sentence, known_words)
        
        if not unknown:
            return known_words
        
        # Create a copy to avoid modifying original
        expanded = dict(known_words)
        
        # Translate each unknown word
        for word in unknown:
            translation = self.translate_unknown_word(
                word, sentence, src_lang, tgt_lang, known_words
            )
            if translation:
                expanded[word] = translation
        
        return expanded
    
    def get_stats(self) -> Dict[str, int]:
        """Get DVE statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'unknown_found': 0,
            'successfully_translated': 0,
            'cache_hits': 0
        }


def translate_with_dve(
    sentence: str,
    word_dict: Dict[str, str],
    llm_client,
    config: Config,
    src_lang: str,
    tgt_lang: str
) -> Tuple[str, Dict[str, str]]:
    """
    Convenience function for DVE-enhanced word mining.
    
    Args:
        sentence: Source sentence
        word_dict: Existing word translations
        llm_client: LLM client
        config: Configuration
        src_lang: Source language name
        tgt_lang: Target language name
        
    Returns:
        Tuple of (sentence, expanded_word_dict)
    """
    dve = DynamicVocabularyExpander(llm_client, config)
    expanded = dve.expand_vocabulary(sentence, word_dict, src_lang, tgt_lang)
    return sentence, expanded
