"""
Utility functions for SM-UMT.
"""

import os
import json
import re
from typing import List, Tuple, Dict, Any


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_parallel_data(
    parallel_data: List[Tuple[str, str]],
    filepath: str
):
    """Save parallel data to TSV file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        for src, tgt in parallel_data:
            f.write(f"{src}\t{tgt}\n")


def load_parallel_data(filepath: str) -> List[Tuple[str, str]]:
    """Load parallel data from TSV file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_sentences(filepath: str) -> List[str]:
    """Load sentences from text file (one per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_sentences(sentences: List[str], filepath: str):
    """Save sentences to text file (one per line)."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent + '\n')


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_language_code(lang_name: str) -> str:
    """Convert language name to ISO 639-3 code."""
    lang_map = {
        'french': 'fra',
        'english': 'eng',
        'arabic': 'arb',
        'fr': 'fra',
        'en': 'eng',
        'ar': 'arb',
    }
    return lang_map.get(lang_name.lower(), lang_name)


def get_language_name(lang_code: str) -> str:
    """Convert ISO 639-3 code to language name."""
    code_map = {
        'fra': 'French',
        'eng': 'English',
        'arb': 'Arabic',
    }
    return code_map.get(lang_code, lang_code)


def format_translation_result(
    source: str,
    translation: str,
    src_lang: str,
    tgt_lang: str
) -> dict:
    """Format translation result as dictionary."""
    return {
        'source': source,
        'translation': translation,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang
    }


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches."""
    return [
        items[i:i+batch_size] 
        for i in range(0, len(items), batch_size)
    ]


class ResultLogger:
    """Logger for translation results."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = ensure_dir(output_dir)
        self.results = []
    
    def log(self, result: dict):
        """Log a single result."""
        self.results.append(result)
    
    def save(self, filename: str = "results.json"):
        """Save all results to file."""
        filepath = os.path.join(self.output_dir, filename)
        save_json(self.results, filepath)
        print(f"Results saved to {filepath}")
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            'total_translations': len(self.results),
            'output_file': os.path.join(self.output_dir, "results.json")
        }
