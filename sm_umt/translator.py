"""
Main translator module for SM-UMT.
Orchestrates the full translation pipeline.
"""

import os
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from .config import Config
from .llm_client import LLMClient
from .word_mining import WordMiner
from .sentence_mining import SentenceMiner
from .prompts import create_sentence_translation_prompt
from .evaluation import Evaluator, get_sample_data, load_flores200
from .utils import (
    ensure_dir,
    save_json,
    load_json,
    save_parallel_data,
    get_language_name,
    ResultLogger
)


class SMUMTTranslator:
    """
    Self-Mining UMT Translator.
    
    Main class that orchestrates the full SM-UMT pipeline:
    1. Word-level mining (Stage 1)
    2. Sentence-level mining with TopK+BM25 (Stage 2)
    3. Translation with mined ICL examples
    """
    
    def __init__(
        self,
        config: Config = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize SMUMTTranslator.
        
        Args:
            config: Configuration object
            api_key: Gemini API key (optional, uses env var if not provided)
        """
        self.config = config or Config()
        self.config.validate()
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            provider="gemini",
            api_key=api_key,
            model_name=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        # Initialize miners
        self.word_miner = WordMiner(self.llm_client, self.config)
        self.sentence_miner = SentenceMiner(self.config)
        
        # Cached data
        self.word_translations = {}
        self.synthetic_parallel = []
        self.corpus_embeddings = None
        
        # Results
        self.results = []
        self.evaluator = Evaluator()
    
    def mine_word_translations(
        self,
        source_sentences: List[str],
        src_lang: str = None,
        tgt_lang: str = None,
        max_words: int = 100
    ) -> Dict[str, str]:
        """
        Stage 1: Mine word-level translations.
        
        Args:
            source_sentences: Monolingual source sentences
            src_lang: Source language code
            tgt_lang: Target language code
            max_words: Maximum words to translate
        
        Returns:
            Dictionary of word translations
        """
        src_lang = src_lang or self.config.src_lang
        tgt_lang = tgt_lang or self.config.tgt_lang
        
        src_name = get_language_name(src_lang)
        tgt_name = get_language_name(tgt_lang)
        
        print(f"\n{'='*50}")
        print(f"Stage 1: Word-Level Mining ({src_name} -> {tgt_name})")
        print(f"{'='*50}")
        
        self.word_translations = self.word_miner.mine_word_translations(
            source_sentences,
            src_name,
            tgt_name,
            max_words=max_words
        )
        
        return self.word_translations
    
    def create_synthetic_parallel(
        self,
        source_sentences: List[str],
        word_translations: Dict[str, str] = None
    ) -> List[Tuple[str, str]]:
        """
        Create synthetic parallel data using word translations.
        
        Args:
            source_sentences: Source sentences
            word_translations: Word translation dictionary
        
        Returns:
            List of (source, synthetic_target) pairs
        """
        word_translations = word_translations or self.word_translations
        
        print("\nCreating synthetic parallel data...")
        
        self.synthetic_parallel = self.word_miner.generate_synthetic_parallel_data(
            source_sentences,
            word_translations
        )
        
        print(f"Created {len(self.synthetic_parallel)} synthetic pairs")
        
        return self.synthetic_parallel
    
    def mine_icl_examples(
        self,
        query: str,
        parallel_corpus: List[Tuple[str, str]] = None,
        k: int = None
    ) -> List[Tuple[str, str]]:
        """
        Stage 2: Mine ICL examples for a query sentence.
        
        Args:
            query: Query sentence to translate
            parallel_corpus: Parallel corpus for mining
            k: Number of ICL examples
        
        Returns:
            List of (source, target) ICL examples
        """
        parallel_corpus = parallel_corpus or self.synthetic_parallel
        k = k or self.config.k
        
        return self.sentence_miner.select_icl_examples(
            query,
            parallel_corpus,
            self.corpus_embeddings,
            k=k
        )
    
    def translate_sentence(
        self,
        sentence: str,
        icl_examples: List[Tuple[str, str]] = None,
        src_lang: str = None,
        tgt_lang: str = None
    ) -> str:
        """
        Translate a single sentence using ICL.
        
        Args:
            sentence: Source sentence
            icl_examples: ICL examples (mined if not provided)
            src_lang: Source language code
            tgt_lang: Target language code
        
        Returns:
            Translated sentence
        """
        src_lang = src_lang or self.config.src_lang
        tgt_lang = tgt_lang or self.config.tgt_lang
        
        src_name = get_language_name(src_lang)
        tgt_name = get_language_name(tgt_lang)
        
        # Mine ICL examples if not provided
        if icl_examples is None:
            icl_examples = self.mine_icl_examples(sentence)
        
        # Create prompt
        prompt = create_sentence_translation_prompt(
            sentence,
            src_name,
            tgt_name,
            icl_examples
        )
        
        # Generate translation
        translation = self.llm_client.generate(prompt)
        
        return translation.strip()
    
    def translate_batch(
        self,
        sentences: List[str],
        src_lang: str = None,
        tgt_lang: str = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            sentences: List of source sentences
            src_lang: Source language code
            tgt_lang: Target language code
            show_progress: Whether to show progress bar
        
        Returns:
            List of translated sentences
        """
        src_lang = src_lang or self.config.src_lang
        tgt_lang = tgt_lang or self.config.tgt_lang
        
        src_name = get_language_name(src_lang)
        tgt_name = get_language_name(tgt_lang)
        
        print(f"\n{'='*50}")
        print(f"Stage 2: Sentence Translation ({src_name} -> {tgt_name})")
        print(f"{'='*50}")
        
        # Pre-compute corpus embeddings
        if self.synthetic_parallel:
            source_sents = [s for s, t in self.synthetic_parallel]
            print("Computing corpus embeddings...")
            self.corpus_embeddings = self.sentence_miner.compute_embeddings(source_sents)
        
        translations = []
        iterator = tqdm(sentences, desc="Translating") if show_progress else sentences
        
        for sentence in iterator:
            translation = self.translate_sentence(
                sentence,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            translations.append(translation)
        
        return translations
    
    def run_pipeline(
        self,
        source_sentences: List[str],
        references: List[str] = None,
        src_lang: str = None,
        tgt_lang: str = None
    ) -> Dict:
        """
        Run the full SM-UMT pipeline.
        
        Args:
            source_sentences: Source sentences to translate
            references: Reference translations (for evaluation)
            src_lang: Source language code
            tgt_lang: Target language code
        
        Returns:
            Dictionary with translations and evaluation metrics
        """
        src_lang = src_lang or self.config.src_lang
        tgt_lang = tgt_lang or self.config.tgt_lang
        
        print(f"\n{'='*60}")
        print(f"SM-UMT Translation Pipeline")
        print(f"Source: {get_language_name(src_lang)}, Target: {get_language_name(tgt_lang)}")
        print(f"Sentences: {len(source_sentences)}")
        print(f"{'='*60}")
        
        # Stage 1: Word-level mining
        self.mine_word_translations(source_sentences, src_lang, tgt_lang)
        
        # Create synthetic parallel data
        self.create_synthetic_parallel(source_sentences)
        
        # Stage 2: Translate with mined ICL examples
        translations = self.translate_batch(source_sentences, src_lang, tgt_lang)
        
        # Prepare results
        result = {
            'translations': translations,
            'source_sentences': source_sentences,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'num_word_translations': len(self.word_translations),
            'num_synthetic_pairs': len(self.synthetic_parallel),
            'llm_stats': self.llm_client.get_stats()
        }
        
        # Evaluate if references provided
        if references:
            print("\nEvaluating translations...")
            eval_result = self.evaluator.compute_bleu(translations, references)
            result['evaluation'] = eval_result
            print(f"BLEU Score: {eval_result['bleu']:.2f}")
        
        return result
    
    def save_results(self, result: Dict, output_dir: str = None):
        """Save results to output directory."""
        output_dir = output_dir or self.config.output_dir
        ensure_dir(output_dir)
        
        # Save translations
        save_json(result, os.path.join(output_dir, "results.json"))
        
        # Save word translations
        save_json(
            self.word_translations,
            os.path.join(output_dir, "word_translations.json")
        )
        
        # Save synthetic parallel data
        save_parallel_data(
            self.synthetic_parallel,
            os.path.join(output_dir, "synthetic_parallel.tsv")
        )
        
        print(f"\nResults saved to {output_dir}")


def translate(
    sentence: str,
    src_lang: str = "fra",
    tgt_lang: str = "eng",
    api_key: str = None
) -> str:
    """
    Convenience function for quick translation.
    
    Args:
        sentence: Sentence to translate
        src_lang: Source language code
        tgt_lang: Target language code
        api_key: Gemini API key
    
    Returns:
        Translated sentence
    """
    config = Config(src_lang=src_lang, tgt_lang=tgt_lang)
    translator = SMUMTTranslator(config, api_key)
    
    # Use sample data for ICL mining
    sample_data = get_sample_data(src_lang, tgt_lang, 15)
    source_samples = [s for s, t in sample_data]
    
    # Run mining
    translator.mine_word_translations(source_samples)
    translator.create_synthetic_parallel(source_samples)
    
    # Translate
    return translator.translate_sentence(sentence)
