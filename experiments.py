"""
Experiment runner for SM-UMT improvements.
Runs comparative experiments between baseline and improved methods.

Implements rate limiting to respect API quotas.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sm_umt import Config, SMUMTTranslator
from sm_umt.dve import DynamicVocabularyExpander
from sm_umt.qais import QualityAwareICLSelector
from sm_umt.dam import DomainAdaptiveMiner, get_hate_speech_data
from sm_umt.evaluation import Evaluator, get_sample_data, load_flores200
from sm_umt.llm_client import LLMClient


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 15):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum API calls per minute (Gemini free tier: ~15)
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
    
    def wait(self):
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"  [Rate limit] Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_call_time = time.time()


class ExperimentRunner:
    """
    Runs comparative experiments for SM-UMT improvements.
    """
    
    def __init__(
        self,
        api_key: str,
        src_lang: str = 'fra',
        tgt_lang: str = 'eng',
        calls_per_minute: int = 15
    ):
        """
        Initialize experiment runner.
        
        Args:
            api_key: Gemini API key
            src_lang: Source language code
            tgt_lang: Target language code
            calls_per_minute: API rate limit
        """
        self.api_key = api_key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.rate_limiter = RateLimiter(calls_per_minute)
        
        self.config = Config(src_lang=src_lang, tgt_lang=tgt_lang)
        self.evaluator = Evaluator()
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
            },
            'experiments': {}
        }
    
    def run_baseline(
        self,
        test_data: List[Tuple[str, str]],
        num_samples: int = 10
    ) -> Dict:
        """
        Run baseline SM-UMT experiment.
        
        Args:
            test_data: List of (source, reference) pairs
            num_samples: Number of samples to evaluate
            
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT: Baseline SM-UMT")
        print(f"{'='*60}")
        
        samples = test_data[:num_samples]
        sources = [s for s, _ in samples]
        references = [r for _, r in samples]
        
        translator = SMUMTTranslator(self.config, api_key=self.api_key)
        
        translations = []
        start_time = time.time()
        
        for i, sentence in enumerate(sources):
            print(f"  [{i+1}/{len(sources)}] Translating: {sentence[:50]}...")
            self.rate_limiter.wait()
            
            try:
                result = translator.translate_sentence(sentence)
                translations.append(result)
                print(f"    -> {result[:50]}...")
            except Exception as e:
                print(f"    -> ERROR: {e}")
                translations.append("")
        
        elapsed = time.time() - start_time
        
        # Evaluate
        bleu_result = self.evaluator.compute_bleu(translations, references)
        
        result = {
            'method': 'baseline',
            'num_samples': len(samples),
            'bleu': bleu_result['bleu'],
            'precisions': bleu_result['precisions'],
            'time_seconds': elapsed,
            'translations': translations[:5],  # Save first 5 for inspection
            'references': references[:5],
        }
        
        print(f"\n  BLEU Score: {bleu_result['bleu']:.2f}")
        print(f"  Time: {elapsed:.1f}s")
        
        self.results['experiments']['baseline'] = result
        return result
    
    def run_with_dve(
        self,
        test_data: List[Tuple[str, str]],
        num_samples: int = 10
    ) -> Dict:
        """
        Run SM-UMT with Dynamic Vocabulary Expansion.
        
        Args:
            test_data: List of (source, reference) pairs
            num_samples: Number of samples
            
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT: SM-UMT + DVE (Dynamic Vocabulary Expansion)")
        print(f"{'='*60}")
        
        samples = test_data[:num_samples]
        sources = [s for s, _ in samples]
        references = [r for _, r in samples]
        
        translator = SMUMTTranslator(self.config, api_key=self.api_key)
        llm_client = LLMClient(api_key=self.api_key)
        dve = DynamicVocabularyExpander(llm_client, self.config)
        
        translations = []
        unknown_resolved = 0
        total_unknown = 0
        start_time = time.time()
        
        src_name = self.config.get_lang_name(self.src_lang)
        tgt_name = self.config.get_lang_name(self.tgt_lang)
        
        for i, sentence in enumerate(sources):
            print(f"  [{i+1}/{len(sources)}] Translating: {sentence[:50]}...")
            self.rate_limiter.wait()
            
            try:
                # Get current word dictionary
                word_dict = translator.word_translations if hasattr(translator, 'word_translations') else {}
                
                # Find and translate unknown words
                unknown = dve.find_unknown_words(sentence, word_dict)
                total_unknown += len(unknown)
                
                if unknown:
                    print(f"    Found {len(unknown)} unknown words: {unknown[:3]}...")
                    # Expand vocabulary
                    for word in unknown[:3]:  # Limit to save API calls
                        self.rate_limiter.wait()
                        translation = dve.translate_unknown_word(
                            word, sentence, src_name, tgt_name, word_dict
                        )
                        if translation:
                            word_dict[word] = translation
                            unknown_resolved += 1
                            print(f"      {word} -> {translation}")
                
                # Translate with expanded vocabulary
                translator.word_translations = word_dict
                result = translator.translate_sentence(sentence)
                translations.append(result)
                print(f"    -> {result[:50]}...")
                
            except Exception as e:
                print(f"    -> ERROR: {e}")
                translations.append("")
        
        elapsed = time.time() - start_time
        
        # Evaluate
        bleu_result = self.evaluator.compute_bleu(translations, references)
        
        dve_stats = dve.get_stats()
        result = {
            'method': 'baseline_dve',
            'num_samples': len(samples),
            'bleu': bleu_result['bleu'],
            'precisions': bleu_result['precisions'],
            'time_seconds': elapsed,
            'dve_stats': {
                'total_unknown': total_unknown,
                'resolved': unknown_resolved,
                'resolution_rate': unknown_resolved / max(total_unknown, 1),
            },
            'translations': translations[:5],
            'references': references[:5],
        }
        
        print(f"\n  BLEU Score: {bleu_result['bleu']:.2f}")
        print(f"  Unknown words found: {total_unknown}, resolved: {unknown_resolved}")
        print(f"  Time: {elapsed:.1f}s")
        
        self.results['experiments']['baseline_dve'] = result
        return result
    
    def run_with_qais(
        self,
        test_data: List[Tuple[str, str]],
        num_samples: int = 10
    ) -> Dict:
        """
        Run SM-UMT with Quality-Aware ICL Selection.
        
        Args:
            test_data: List of (source, reference) pairs
            num_samples: Number of samples
            
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT: SM-UMT + QAIS (Quality-Aware ICL Selection)")
        print(f"{'='*60}")
        
        samples = test_data[:num_samples]
        sources = [s for s, _ in samples]
        references = [r for _, r in samples]
        
        translator = SMUMTTranslator(self.config, api_key=self.api_key)
        llm_client = LLMClient(api_key=self.api_key)
        qais = QualityAwareICLSelector(llm_client, self.config)
        
        translations = []
        filtered_count = 0
        start_time = time.time()
        
        src_name = self.config.get_lang_name(self.src_lang)
        tgt_name = self.config.get_lang_name(self.tgt_lang)
        
        for i, sentence in enumerate(sources):
            print(f"  [{i+1}/{len(sources)}] Translating: {sentence[:50]}...")
            self.rate_limiter.wait()
            
            try:
                # Get ICL examples
                icl_examples = translator.mine_icl_examples(sentence)
                
                if icl_examples:
                    original_count = len(icl_examples)
                    # Filter by quality (no back-translation to save API calls)
                    icl_examples = qais.select_best_examples(
                        icl_examples, src_name, tgt_name,
                        k=self.config.k,
                        use_back_translation=False
                    )
                    filtered_count += original_count - len(icl_examples)
                
                # Translate with quality-filtered examples
                result = translator.translate_sentence(sentence, icl_examples=icl_examples)
                translations.append(result)
                print(f"    -> {result[:50]}...")
                
            except Exception as e:
                print(f"    -> ERROR: {e}")
                translations.append("")
        
        elapsed = time.time() - start_time
        
        # Evaluate
        bleu_result = self.evaluator.compute_bleu(translations, references)
        
        qais_stats = qais.get_stats()
        result = {
            'method': 'baseline_qais',
            'num_samples': len(samples),
            'bleu': bleu_result['bleu'],
            'precisions': bleu_result['precisions'],
            'time_seconds': elapsed,
            'qais_stats': {
                'total_evaluated': qais_stats['total_evaluated'],
                'passed_quality': qais_stats['passed_quality'],
                'filtered_out': filtered_count,
            },
            'translations': translations[:5],
            'references': references[:5],
        }
        
        print(f"\n  BLEU Score: {bleu_result['bleu']:.2f}")
        print(f"  Quality stats: {qais_stats}")
        print(f"  Time: {elapsed:.1f}s")
        
        self.results['experiments']['baseline_qais'] = result
        return result
    
    def run_domain_adaptive(
        self,
        num_samples: int = 10
    ) -> Dict:
        """
        Run SM-UMT with Domain-Adaptive Mining (hate speech domain).
        
        Args:
            num_samples: Number of samples
            
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT: SM-UMT + DAM (Domain-Adaptive Mining - Hate Speech)")
        print(f"{'='*60}")
        
        # Get domain-specific test data
        test_data = get_hate_speech_data(self.src_lang, self.tgt_lang, num_samples)
        
        if not test_data:
            print("  No domain data available for this language pair.")
            return {'error': 'No domain data available'}
        
        sources = [s for s, _ in test_data]
        references = [r for _, r in test_data]
        
        translator = SMUMTTranslator(self.config, api_key=self.api_key)
        llm_client = LLMClient(api_key=self.api_key)
        dam = DomainAdaptiveMiner(llm_client, self.config, domain='hate_speech')
        
        # Mine domain vocabulary
        print("  Mining domain vocabulary...")
        domain_vocab = dam.mine_domain_vocabulary(sources, self.src_lang, self.tgt_lang)
        print(f"  Domain vocabulary size: {len(domain_vocab)}")
        
        translations = []
        start_time = time.time()
        
        for i, sentence in enumerate(sources):
            print(f"  [{i+1}/{len(sources)}] Translating: {sentence[:50]}...")
            self.rate_limiter.wait()
            
            try:
                # Merge domain vocab with general vocab
                if hasattr(translator, 'word_translations'):
                    translator.word_translations = dam.merge_with_general_vocab(
                        translator.word_translations
                    )
                else:
                    translator.word_translations = domain_vocab
                
                result = translator.translate_sentence(sentence)
                translations.append(result)
                print(f"    -> {result[:50]}...")
                
            except Exception as e:
                print(f"    -> ERROR: {e}")
                translations.append("")
        
        elapsed = time.time() - start_time
        
        # Evaluate
        bleu_result = self.evaluator.compute_bleu(translations, references)
        
        dam_stats = dam.get_stats()
        result = {
            'method': 'baseline_dam',
            'domain': 'hate_speech',
            'num_samples': len(test_data),
            'bleu': bleu_result['bleu'],
            'precisions': bleu_result['precisions'],
            'time_seconds': elapsed,
            'dam_stats': {
                'domain_vocab_size': len(domain_vocab),
                'words_used': dam_stats['domain_words_used'],
                'words_mined': dam_stats['domain_words_mined'],
            },
            'translations': translations[:5],
            'references': references[:5],
        }
        
        print(f"\n  BLEU Score: {bleu_result['bleu']:.2f}")
        print(f"  Domain vocab: {len(domain_vocab)} words")
        print(f"  Time: {elapsed:.1f}s")
        
        self.results['experiments']['baseline_dam'] = result
        return result
    
    def run_all(self, num_samples: int = 10) -> Dict:
        """
        Run all experiments.
        
        Args:
            num_samples: Number of samples per experiment
            
        Returns:
            All results
        """
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print(f"Samples per experiment: {num_samples}")
        print("="*60)
        
        # Get test data
        test_data = get_sample_data(self.src_lang, self.tgt_lang, num_samples)
        
        # Run experiments
        self.run_baseline(test_data, num_samples)
        self.run_with_dve(test_data, num_samples)
        self.run_with_qais(test_data, num_samples)
        self.run_domain_adaptive(num_samples)  # Uses its own data
        
        # Summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print results summary."""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n{'Method':<30} {'BLEU':>10} {'Time (s)':>10}")
        print("-" * 52)
        
        for name, exp in self.results['experiments'].items():
            if 'error' not in exp:
                print(f"{name:<30} {exp['bleu']:>10.2f} {exp['time_seconds']:>10.1f}")
    
    def save_results(self, output_path: str = 'outputs/experiment_results.json'):
        """
        Save results to JSON file.
        
        Args:
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run SM-UMT experiments')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--src', type=str, default='fra', help='Source language (fra, arb)')
    parser.add_argument('--tgt', type=str, default='eng', help='Target language (eng)')
    parser.add_argument('--samples', type=int, default=5, help='Samples per experiment')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'baseline', 'dve', 'qais', 'dam'],
                        help='Which experiment to run')
    parser.add_argument('--rate-limit', type=int, default=10,
                        help='API calls per minute (Gemini free: ~15)')
    parser.add_argument('--output', type=str, default='outputs/experiment_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: Please provide --api-key or set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    # Create runner
    runner = ExperimentRunner(
        api_key=api_key,
        src_lang=args.src,
        tgt_lang=args.tgt,
        calls_per_minute=args.rate_limit
    )
    
    # Get test data
    test_data = get_sample_data(args.src, args.tgt, args.samples)
    
    # Run experiments
    if args.experiment == 'all':
        runner.run_all(args.samples)
    elif args.experiment == 'baseline':
        runner.run_baseline(test_data, args.samples)
    elif args.experiment == 'dve':
        runner.run_with_dve(test_data, args.samples)
    elif args.experiment == 'qais':
        runner.run_with_qais(test_data, args.samples)
    elif args.experiment == 'dam':
        runner.run_domain_adaptive(args.samples)
    
    # Save results
    runner.save_results(args.output)


if __name__ == '__main__':
    main()
