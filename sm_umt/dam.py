"""
Domain-Adaptive Mining (DAM) module for SM-UMT.
Implements domain-specific vocabulary mining for specialized translation.

Novel contribution: Domain-adaptive word mining for specialized domains like hate speech.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from .config import Config


# Domain-specific vocabulary for hate speech detection
HATE_SPEECH_VOCABULARY = {
    # French hate speech terms -> English
    'fra_eng': {
        'discriminer': 'discriminate',
        'discrimination': 'discrimination',
        'racisme': 'racism',
        'raciste': 'racist',
        'xénophobie': 'xenophobia',
        'xénophobe': 'xenophobic',
        'islamophobie': 'islamophobia',
        'antisémitisme': 'antisemitism',
        'homophobie': 'homophobia',
        'sexisme': 'sexism',
        'sexiste': 'sexist',
        'haine': 'hatred',
        'haïr': 'hate',
        'insulte': 'insult',
        'injure': 'slur',
        'menace': 'threat',
        'violence': 'violence',
        'agression': 'aggression',
        'stéréotype': 'stereotype',
        'préjugé': 'prejudice',
        'intolérance': 'intolerance',
        'oppression': 'oppression',
        'marginaliser': 'marginalize',
        'déshumaniser': 'dehumanize',
        'dégrader': 'degrade',
        'humilier': 'humiliate',
        'offenser': 'offend',
        'blesser': 'hurt',
        'mépris': 'contempt',
        'hostilité': 'hostility',
    },
    # Arabic hate speech terms -> English
    'arb_eng': {
        'تمييز': 'discrimination',
        'عنصرية': 'racism',
        'كراهية': 'hatred',
        'إهانة': 'insult',
        'تهديد': 'threat',
        'عنف': 'violence',
        'اضطهاد': 'persecution',
        'تحقير': 'degradation',
        'إساءة': 'abuse',
        'تطرف': 'extremism',
        'تعصب': 'bigotry',
    }
}

# Sample hate speech parallel sentences for evaluation
HATE_SPEECH_TEST_DATA = {
    'fra_eng': [
        ("Ce discours encourage la discrimination contre les minorités.",
         "This speech encourages discrimination against minorities."),
        ("Les commentaires racistes sont interdits sur notre plateforme.",
         "Racist comments are prohibited on our platform."),
        ("Nous devons combattre la xénophobie dans notre société.",
         "We must fight xenophobia in our society."),
        ("Cette déclaration contient des stéréotypes offensants.",
         "This statement contains offensive stereotypes."),
        ("Le discours de haine vise à déshumaniser les autres.",
         "Hate speech aims to dehumanize others."),
        ("Ces propos constituent une incitation à la violence.",
         "These words constitute an incitement to violence."),
        ("Le contenu haineux doit être signalé et supprimé.",
         "Hateful content should be reported and removed."),
        ("Les préjugés mènent souvent à la discrimination.",
         "Prejudice often leads to discrimination."),
        ("Cette insulte ciblait leur origine ethnique.",
         "This insult targeted their ethnic origin."),
        ("L'intolérance religieuse n'a pas sa place ici.",
         "Religious intolerance has no place here."),
    ],
    'arb_eng': [
        ("هذا الخطاب يشجع على التمييز ضد الأقليات.",
         "This speech encourages discrimination against minorities."),
        ("التعليقات العنصرية محظورة على منصتنا.",
         "Racist comments are prohibited on our platform."),
        ("يجب علينا محاربة الكراهية في مجتمعنا.",
         "We must fight hatred in our society."),
        ("هذا البيان يحتوي على إهانات.",
         "This statement contains insults."),
        ("خطاب الكراهية يهدف إلى تحقير الآخرين.",
         "Hate speech aims to degrade others."),
    ]
}


class DomainAdaptiveMiner:
    """
    Domain-Adaptive Mining for SM-UMT.
    
    Mines domain-specific vocabulary for specialized translation tasks.
    Currently supports: hate speech detection domain.
    """
    
    def __init__(
        self,
        llm_client,
        config: Config,
        domain: str = 'hate_speech'
    ):
        """
        Initialize DAM.
        
        Args:
            llm_client: LLM client for translation
            config: Configuration object
            domain: Domain name (currently 'hate_speech')
        """
        self.llm_client = llm_client
        self.config = config
        self.domain = domain
        self.domain_vocab = {}
        self.stats = {
            'domain_words_used': 0,
            'domain_words_mined': 0,
        }
    
    def load_domain_vocabulary(
        self,
        src_lang: str,
        tgt_lang: str
    ) -> Dict[str, str]:
        """
        Load pre-defined domain vocabulary.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Domain-specific word translations
        """
        key = f'{src_lang}_{tgt_lang}'
        
        if self.domain == 'hate_speech':
            vocab = HATE_SPEECH_VOCABULARY.get(key, {})
            self.domain_vocab = vocab.copy()
            return vocab
        
        return {}
    
    def extract_domain_words(
        self,
        sentences: List[str],
        src_lang: str
    ) -> Set[str]:
        """
        Extract domain-specific words from sentences.
        
        Args:
            sentences: List of source sentences
            src_lang: Source language code
            
        Returns:
            Set of domain-specific words found
        """
        domain_words = set()
        domain_vocab = HATE_SPEECH_VOCABULARY.get(f'{src_lang}_eng', {})
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                if word in domain_vocab:
                    domain_words.add(word)
        
        self.stats['domain_words_used'] = len(domain_words)
        return domain_words
    
    def mine_domain_vocabulary(
        self,
        sentences: List[str],
        src_lang: str,
        tgt_lang: str
    ) -> Dict[str, str]:
        """
        Mine domain vocabulary from sentences.
        
        Uses both pre-defined vocabulary and LLM for unknown domain terms.
        
        Args:
            sentences: List of source sentences
            src_lang: Source language code (e.g., 'fra')
            tgt_lang: Target language code (e.g., 'eng')
            
        Returns:
            Domain-specific word translations
        """
        src_name = self.config.get_lang_name(src_lang)
        tgt_name = self.config.get_lang_name(tgt_lang)
        
        # Start with pre-defined vocabulary
        vocab = self.load_domain_vocabulary(src_lang, tgt_lang)
        
        # Find domain words in sentences
        domain_words = self.extract_domain_words(sentences, src_lang)
        
        # For words not in pre-defined vocab, use LLM
        for word in domain_words:
            if word not in vocab:
                translation = self._translate_domain_word(word, src_name, tgt_name)
                if translation:
                    vocab[word] = translation
                    self.stats['domain_words_mined'] += 1
        
        self.domain_vocab = vocab
        return vocab
    
    def _translate_domain_word(
        self,
        word: str,
        src_lang: str,
        tgt_lang: str
    ) -> Optional[str]:
        """
        Translate a domain-specific word using LLM.
        
        Args:
            word: Source word
            src_lang: Source language name
            tgt_lang: Target language name
            
        Returns:
            Translated word
        """
        prompt = f"""Translate the following {src_lang} word related to hate speech detection to {tgt_lang}.
Only provide the single word translation.

Word: {word}
Translation:"""
        
        try:
            result = self.llm_client.generate(prompt)
            translation = result.strip().lower()
            return translation.split()[0] if translation.split() else None
        except Exception as e:
            print(f"DAM: Failed to translate '{word}': {e}")
            return None
    
    def merge_with_general_vocab(
        self,
        general_vocab: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Merge domain vocabulary with general vocabulary.
        
        Domain vocabulary takes precedence for overlapping words.
        
        Args:
            general_vocab: General word translations
            
        Returns:
            Merged vocabulary
        """
        merged = dict(general_vocab)
        merged.update(self.domain_vocab)  # Domain takes precedence
        return merged
    
    def get_domain_test_data(
        self,
        src_lang: str,
        tgt_lang: str,
        max_samples: int = None
    ) -> List[Tuple[str, str]]:
        """
        Get domain-specific test data.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            max_samples: Maximum samples to return
            
        Returns:
            List of (source, target) parallel pairs
        """
        key = f'{src_lang}_{tgt_lang}'
        data = HATE_SPEECH_TEST_DATA.get(key, [])
        
        if max_samples:
            return data[:max_samples]
        return data
    
    def get_stats(self) -> Dict[str, int]:
        """Get DAM statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'domain_words_used': 0,
            'domain_words_mined': 0,
        }


def get_hate_speech_data(
    src_lang: str = 'fra',
    tgt_lang: str = 'eng',
    max_samples: int = None
) -> List[Tuple[str, str]]:
    """
    Convenience function to get hate speech test data.
    
    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        max_samples: Maximum samples
        
    Returns:
        List of (source, target) parallel pairs
    """
    key = f'{src_lang}_{tgt_lang}'
    data = HATE_SPEECH_TEST_DATA.get(key, [])
    
    if max_samples:
        return data[:max_samples]
    return data
