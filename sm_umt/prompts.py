"""
Prompt templates for SM-UMT.
Contains prompts for word-level and sentence-level translation.
"""

from typing import List, Tuple


def create_word_translation_prompt(
    word: str,
    src_lang: str,
    tgt_lang: str,
    examples: List[Tuple[str, str]] = None
) -> str:
    """
    Create a prompt for word-level translation.
    
    Args:
        word: Source word to translate
        src_lang: Source language name
        tgt_lang: Target language name
        examples: Optional list of (source_word, target_word) examples
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a professional translator. Translate the following word from {src_lang} to {tgt_lang}.
Only provide the translation, nothing else.

"""
    
    if examples:
        prompt += "Examples:\n"
        for src_w, tgt_w in examples:
            prompt += f"{src_w} → {tgt_w}\n"
        prompt += "\n"
    
    prompt += f"Word: {word}\nTranslation:"
    
    return prompt


def create_word_extraction_prompt(sentence: str, src_lang: str) -> str:
    """
    Create a prompt to extract key content words from a sentence.
    
    Args:
        sentence: Source sentence
        src_lang: Source language name
    
    Returns:
        Formatted prompt string
    """
    return f"""Extract the key content words (nouns, verbs, adjectives, adverbs) from the following {src_lang} sentence.
Return only the words separated by commas, nothing else.

Sentence: {sentence}
Words:"""


def create_sentence_translation_prompt(
    sentence: str,
    src_lang: str,
    tgt_lang: str,
    icl_examples: List[Tuple[str, str]] = None
) -> str:
    """
    Create a prompt for sentence-level translation with ICL examples.
    
    Args:
        sentence: Source sentence to translate
        src_lang: Source language name
        tgt_lang: Target language name
        icl_examples: List of (source_sentence, target_sentence) ICL examples
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a professional translator. Translate the following sentence from {src_lang} to {tgt_lang}.
Only provide the translation, nothing else.

"""
    
    if icl_examples:
        prompt += "Examples:\n"
        for i, (src_sent, tgt_sent) in enumerate(icl_examples, 1):
            prompt += f"{src_lang}: {src_sent}\n{tgt_lang}: {tgt_sent}\n\n"
    
    prompt += f"{src_lang}: {sentence}\n{tgt_lang}:"
    
    return prompt


def create_synthetic_translation_prompt(
    sentence: str,
    word_translations: dict,
    src_lang: str,
    tgt_lang: str
) -> str:
    """
    Create a prompt for generating synthetic parallel sentence using word translations.
    
    Args:
        sentence: Source sentence
        word_translations: Dict mapping source words to translated words
        src_lang: Source language name
        tgt_lang: Target language name
    
    Returns:
        Formatted prompt string
    """
    word_mapping = "\n".join([f"  {src} → {tgt}" for src, tgt in word_translations.items()])
    
    return f"""Given the following word translations from {src_lang} to {tgt_lang}:
{word_mapping}

Create a natural {tgt_lang} translation of this {src_lang} sentence, using the word translations as guidance.
Only provide the translation, nothing else.

{src_lang} sentence: {sentence}
{tgt_lang} translation:"""


def create_batch_word_translation_prompt(
    words: List[str],
    src_lang: str,
    tgt_lang: str
) -> str:
    """
    Create a prompt for batch word translation.
    
    Args:
        words: List of source words to translate
        src_lang: Source language name
        tgt_lang: Target language name
    
    Returns:
        Formatted prompt string
    """
    words_str = "\n".join([f"{i+1}. {w}" for i, w in enumerate(words)])
    
    return f"""Translate the following words from {src_lang} to {tgt_lang}.
For each word, provide only the translation in the same numbered format.

Words to translate:
{words_str}

Translations:"""
