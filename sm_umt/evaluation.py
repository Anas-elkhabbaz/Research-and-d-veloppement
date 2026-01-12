"""
Evaluation module for SM-UMT.
Implements BLEU score evaluation and FLORES-200 benchmarking.
"""

from typing import List, Tuple, Dict, Optional
import sacrebleu
from sacrebleu.metrics import BLEU


class Evaluator:
    """
    Evaluator for translation quality.
    
    Computes BLEU scores and other metrics for translation evaluation.
    """
    
    def __init__(self, lowercase: bool = False):
        """
        Initialize Evaluator.
        
        Args:
            lowercase: Whether to lowercase before computing BLEU
        """
        self.lowercase = lowercase
        self.bleu = BLEU(lowercase=lowercase)
    
    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> dict:
        """
        Compute BLEU score between hypotheses and references.
        
        Args:
            hypotheses: List of translated sentences
            references: List of reference translations
        
        Returns:
            Dictionary with BLEU score and components
        """
        # sacrebleu expects references as a list of lists (one list per reference set)
        # For single reference: [references] where each ref matches the corresponding hypothesis
        result = self.bleu.corpus_score(hypotheses, [references])
        
        return {
            'bleu': result.score,
            'bp': result.bp,  # Brevity penalty
            'precisions': result.precisions,  # n-gram precisions
            'sys_len': result.sys_len,
            'ref_len': result.ref_len,
        }
    
    def compute_sentence_bleu(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """
        Compute sentence-level BLEU score.
        
        Args:
            hypothesis: Translated sentence
            reference: Reference translation
        
        Returns:
            Sentence BLEU score
        """
        result = self.bleu.sentence_score(hypothesis, [reference])
        return result.score
    
    def evaluate_batch(
        self,
        results: List[Dict],
        reference_key: str = 'reference',
        hypothesis_key: str = 'translation'
    ) -> dict:
        """
        Evaluate a batch of translation results.
        
        Args:
            results: List of result dicts with translations
            reference_key: Key for reference in result dict
            hypothesis_key: Key for hypothesis in result dict
        
        Returns:
            Evaluation metrics
        """
        hypotheses = [r[hypothesis_key] for r in results]
        references = [r[reference_key] for r in results]
        
        bleu_result = self.compute_bleu(hypotheses, references)
        
        # Compute sentence-level scores
        sentence_scores = [
            self.compute_sentence_bleu(h, r)
            for h, r in zip(hypotheses, references)
        ]
        
        return {
            'corpus_bleu': bleu_result['bleu'],
            'avg_sentence_bleu': sum(sentence_scores) / len(sentence_scores),
            'min_sentence_bleu': min(sentence_scores),
            'max_sentence_bleu': max(sentence_scores),
            'num_samples': len(results),
            'details': bleu_result
        }


def load_flores200(
    src_lang: str,
    tgt_lang: str,
    split: str = 'devtest',
    max_samples: int = None
) -> List[Tuple[str, str]]:
    """
    Load FLORES-200 dataset.
    
    Args:
        src_lang: Source language code (e.g., 'fra_Latn')
        tgt_lang: Target language code (e.g., 'eng_Latn')
        split: Dataset split ('dev' or 'devtest')
        max_samples: Maximum number of samples to load
    
    Returns:
        List of (source, target) sentence pairs
    """
    try:
        from datasets import load_dataset
        
        # Map simplified codes to FLORES-200 codes
        lang_code_map = {
            'fra': 'fra_Latn',
            'eng': 'eng_Latn',
            'arb': 'arb_Arab',
        }
        
        src_code = lang_code_map.get(src_lang, src_lang)
        tgt_code = lang_code_map.get(tgt_lang, tgt_lang)
        
        print(f"Loading FLORES-200: {src_code} -> {tgt_code}")
        
        # Load dataset
        dataset = load_dataset(
            'facebook/flores',
            'all',
            split=split
        )
        
        # Extract sentence pairs
        pairs = []
        for item in dataset:
            src_sent = item.get(f'sentence_{src_code}') or item.get('sentence', {}).get(src_code)
            tgt_sent = item.get(f'sentence_{tgt_code}') or item.get('sentence', {}).get(tgt_code)
            
            if src_sent and tgt_sent:
                pairs.append((src_sent, tgt_sent))
            
            if max_samples and len(pairs) >= max_samples:
                break
        
        print(f"Loaded {len(pairs)} sentence pairs")
        return pairs
        
    except ImportError:
        print("Warning: datasets library not installed. Using sample data.")
        return get_sample_data(src_lang, tgt_lang, max_samples or 10)
    except Exception as e:
        print(f"Error loading FLORES-200: {e}")
        print("Using sample data instead.")
        return get_sample_data(src_lang, tgt_lang, max_samples or 10)


def get_sample_data(
    src_lang: str,
    tgt_lang: str,
    num_samples: int = 10
) -> List[Tuple[str, str]]:
    """
    Get sample parallel data for testing.
    
    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        num_samples: Number of samples
    
    Returns:
        List of (source, target) sentence pairs
    """
    # Sample French-English parallel sentences
    fra_eng = [
        ("Bonjour, comment allez-vous?", "Hello, how are you?"),
        ("Je m'appelle Marie.", "My name is Marie."),
        ("Il fait beau aujourd'hui.", "The weather is nice today."),
        ("J'aime lire des livres.", "I like reading books."),
        ("Quelle heure est-il?", "What time is it?"),
        ("Le chat dort sur le canapé.", "The cat is sleeping on the couch."),
        ("Nous allons au restaurant ce soir.", "We are going to the restaurant tonight."),
        ("Elle travaille dans un hôpital.", "She works in a hospital."),
        ("Les enfants jouent dans le jardin.", "The children are playing in the garden."),
        ("J'ai besoin d'aide.", "I need help."),
        ("Le train arrive à midi.", "The train arrives at noon."),
        ("C'est une belle journée.", "It is a beautiful day."),
        ("Je voudrais un café, s'il vous plaît.", "I would like a coffee, please."),
        ("La bibliothèque est fermée.", "The library is closed."),
        ("Nous habitons à Paris.", "We live in Paris."),
    ]
    
    # Sample Arabic-English parallel sentences
    arb_eng = [
        ("مرحبا، كيف حالك؟", "Hello, how are you?"),
        ("اسمي أحمد.", "My name is Ahmed."),
        ("الطقس جميل اليوم.", "The weather is nice today."),
        ("أحب قراءة الكتب.", "I like reading books."),
        ("كم الساعة؟", "What time is it?"),
        ("القط ينام على الأريكة.", "The cat is sleeping on the couch."),
        ("سنذهب إلى المطعم هذا المساء.", "We are going to the restaurant tonight."),
        ("هي تعمل في مستشفى.", "She works in a hospital."),
        ("الأطفال يلعبون في الحديقة.", "The children are playing in the garden."),
        ("أحتاج مساعدة.", "I need help."),
    ]
    
    if src_lang == 'fra' and tgt_lang == 'eng':
        return fra_eng[:num_samples]
    elif src_lang == 'eng' and tgt_lang == 'fra':
        return [(e, f) for f, e in fra_eng[:num_samples]]
    elif src_lang == 'arb' and tgt_lang == 'eng':
        return arb_eng[:num_samples]
    elif src_lang == 'eng' and tgt_lang == 'arb':
        return [(e, a) for a, e in arb_eng[:num_samples]]
    else:
        # Default: return French-English
        return fra_eng[:num_samples]


def evaluate_translations(
    translations: List[str],
    references: List[str],
    lowercase: bool = False
) -> dict:
    """
    Convenience function for evaluating translations.
    
    Args:
        translations: List of translated sentences
        references: List of reference translations
        lowercase: Whether to lowercase before evaluation
    
    Returns:
        Evaluation metrics
    """
    evaluator = Evaluator(lowercase=lowercase)
    return evaluator.compute_bleu(translations, references)
