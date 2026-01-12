# Self-Mining UMT (SM-UMT)
# Unsupervised Machine Translation using Self-Mined In-Context Examples

"""
Implementation of the SM-UMT system based on the NAACL 2025 paper:
"Effective Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs"
by El Mekki & Abdul-Mageed

Extended with novel improvements:
- DVE: Dynamic Vocabulary Expansion
- QAIS: Quality-Aware ICL Selection  
- DAM: Domain-Adaptive Mining
"""

__version__ = "2.0.0"
__author__ = "Based on El Mekki & Abdul-Mageed (NAACL 2025)"

from .config import Config
from .translator import SMUMTTranslator
from .dve import DynamicVocabularyExpander, translate_with_dve
from .qais import QualityAwareICLSelector, filter_icl_with_quality
from .dam import DomainAdaptiveMiner, get_hate_speech_data
