"""
SM-UMT: Self-Mining Unsupervised Machine Translation

Command-line interface for the SM-UMT translation system.
"""

import argparse
import os
import sys

from sm_umt.config import Config, LANGUAGE_PAIRS
from sm_umt.translator import SMUMTTranslator
from sm_umt.evaluation import get_sample_data, load_flores200, evaluate_translations
from sm_umt.utils import save_json, get_language_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SM-UMT: Self-Mining Unsupervised Machine Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a single sentence
  python main.py --input "Bonjour le monde" --src fra --tgt eng

  # Run translation on sample data
  python main.py --src fra --tgt eng --sample_size 10

  # Evaluate on FLORES-200
  python main.py --evaluate --src fra --tgt eng --sample_size 100

  # List available language pairs
  python main.py --list-langs
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Single sentence to translate"
    )
    
    parser.add_argument(
        "--src", "--source",
        type=str,
        default="fra",
        help="Source language code (fra, eng, arb)"
    )
    
    parser.add_argument(
        "--tgt", "--target",
        type=str,
        default="eng",
        help="Target language code (fra, eng, arb)"
    )
    
    parser.add_argument(
        "--sample_size", "-n",
        type=int,
        default=10,
        help="Number of samples for batch translation"
    )
    
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Evaluate translations with BLEU score"
    )
    
    parser.add_argument(
        "--use-flores",
        action="store_true",
        help="Use FLORES-200 dataset for evaluation"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="List available language pairs"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with minimal samples"
    )
    
    return parser.parse_args()


def list_languages():
    """Print available language pairs."""
    print("\nAvailable Language Pairs:")
    print("-" * 30)
    for src, tgt in LANGUAGE_PAIRS:
        src_name = get_language_name(src)
        tgt_name = get_language_name(tgt)
        print(f"  {src} -> {tgt}  ({src_name} -> {tgt_name})")
    print()


def translate_single(args):
    """Translate a single sentence."""
    print(f"\nTranslating: {args.input}")
    print(f"Direction: {get_language_name(args.src)} -> {get_language_name(args.tgt)}")
    print("-" * 50)
    
    config = Config(src_lang=args.src, tgt_lang=args.tgt)
    translator = SMUMTTranslator(config, api_key=args.api_key)
    
    # Get sample data for ICL mining
    sample_data = get_sample_data(args.src, args.tgt, 15)
    source_samples = [s for s, t in sample_data]
    
    # Run word mining
    if args.verbose:
        print("\nMining word translations...")
    translator.mine_word_translations(source_samples)
    translator.create_synthetic_parallel(source_samples)
    
    # Translate
    translation = translator.translate_sentence(args.input)
    
    print(f"\nSource: {args.input}")
    print(f"Translation: {translation}")
    
    return translation


def run_batch_translation(args):
    """Run batch translation with evaluation."""
    print(f"\nBatch Translation")
    print(f"Direction: {get_language_name(args.src)} -> {get_language_name(args.tgt)}")
    print(f"Sample size: {args.sample_size}")
    print("-" * 50)
    
    # Load data
    if args.use_flores:
        print("\nLoading FLORES-200 dataset...")
        data = load_flores200(args.src, args.tgt, max_samples=args.sample_size)
    else:
        print("\nUsing sample data...")
        data = get_sample_data(args.src, args.tgt, args.sample_size)
    
    source_sentences = [s for s, t in data]
    references = [t for s, t in data]
    
    # Initialize translator
    config = Config(
        src_lang=args.src,
        tgt_lang=args.tgt,
        output_dir=args.output
    )
    translator = SMUMTTranslator(config, api_key=args.api_key)
    
    # Run pipeline
    if args.evaluate:
        result = translator.run_pipeline(source_sentences, references)
    else:
        result = translator.run_pipeline(source_sentences)
    
    # Save results
    translator.save_results(result)
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Sentences translated: {len(source_sentences)}")
    print(f"Word translations mined: {result['num_word_translations']}")
    print(f"Synthetic pairs created: {result['num_synthetic_pairs']}")
    print(f"LLM API calls: {result['llm_stats']['request_count']}")
    
    if 'evaluation' in result:
        print(f"\nBLEU Score: {result['evaluation']['bleu']:.2f}")
    
    # Show some examples
    if args.verbose:
        print("\n" + "-" * 50)
        print("SAMPLE TRANSLATIONS")
        print("-" * 50)
        for i in range(min(3, len(source_sentences))):
            print(f"\nSource: {source_sentences[i]}")
            print(f"Translation: {result['translations'][i]}")
            if args.evaluate and references:
                print(f"Reference: {references[i]}")
    
    return result


def run_test(args):
    """Run quick test with minimal samples."""
    print("\n" + "=" * 50)
    print("RUNNING SM-UMT TEST")
    print("=" * 50)
    
    args.sample_size = 3
    args.evaluate = True
    args.verbose = True
    
    return run_batch_translation(args)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check for API key
    if not args.api_key and not os.environ.get("GEMINI_API_KEY"):
        print("\n⚠️  Warning: No Gemini API key found!")
        print("Set GEMINI_API_KEY environment variable or use --api-key argument.")
        print("\nTo get an API key, visit: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    if args.list_langs:
        list_languages()
        return
    
    if args.test:
        run_test(args)
        return
    
    if args.input:
        translate_single(args)
    else:
        run_batch_translation(args)


if __name__ == "__main__":
    main()
