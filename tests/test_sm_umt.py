"""
Tests for SM-UMT modules.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_umt.config import Config
from sm_umt.bm25 import BM25, select_with_bm25
from sm_umt.prompts import (
    create_word_translation_prompt,
    create_sentence_translation_prompt
)
from sm_umt.utils import clean_text, get_language_code, get_language_name


class TestConfig:
    """Test configuration module."""
    
    def test_default_config(self):
        config = Config()
        assert config.kwp == 10
        assert config.k == 8
        assert config.tau == 0.90
        assert config.top_n == 20
    
    def test_custom_config(self):
        config = Config(src_lang="arb", tgt_lang="eng", k=5)
        assert config.src_lang == "arb"
        assert config.tgt_lang == "eng"
        assert config.k == 5
    
    def test_get_lang_name(self):
        config = Config()
        assert config.get_lang_name("fra") == "French"
        assert config.get_lang_name("eng") == "English"
        assert config.get_lang_name("arb") == "Arabic"
    
    def test_validate(self):
        config = Config()
        config.validate()  # Should not raise
        
        config.kwp = -1
        with pytest.raises(AssertionError):
            config.validate()


class TestBM25:
    """Test BM25 implementation."""
    
    def test_initialization(self):
        corpus = ["hello world", "foo bar", "hello foo"]
        bm25 = BM25(corpus=corpus)
        assert bm25.N == 3
        assert bm25.avgdl > 0
    
    def test_get_scores(self):
        corpus = ["hello world", "foo bar", "hello foo"]
        bm25 = BM25(corpus=corpus)
        scores = bm25.get_scores("hello")
        
        assert len(scores) == 3
        # "hello world" and "hello foo" should score higher than "foo bar"
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]
    
    def test_get_top_k(self):
        corpus = ["hello world", "foo bar", "hello foo", "test doc"]
        bm25 = BM25(corpus=corpus)
        top_k = bm25.get_top_k("hello world", k=2)
        
        assert len(top_k) == 2
        assert top_k[0][0] == 0  # "hello world" should be first
    
    def test_select_with_bm25(self):
        candidates = ["hello world", "foo bar", "hello there"]
        selected = select_with_bm25("hello", candidates, k=2)
        
        assert len(selected) == 2
        assert "hello world" in selected or "hello there" in selected


class TestPrompts:
    """Test prompt templates."""
    
    def test_word_translation_prompt(self):
        prompt = create_word_translation_prompt("bonjour", "French", "English")
        assert "bonjour" in prompt
        assert "French" in prompt
        assert "English" in prompt
    
    def test_word_translation_prompt_with_examples(self):
        examples = [("chat", "cat"), ("chien", "dog")]
        prompt = create_word_translation_prompt(
            "oiseau", "French", "English", examples
        )
        assert "chat" in prompt
        assert "cat" in prompt
        assert "oiseau" in prompt
    
    def test_sentence_translation_prompt(self):
        prompt = create_sentence_translation_prompt(
            "Bonjour le monde",
            "French",
            "English"
        )
        assert "Bonjour le monde" in prompt
        assert "French" in prompt
        assert "English" in prompt
    
    def test_sentence_translation_prompt_with_icl(self):
        icl_examples = [
            ("Comment allez-vous?", "How are you?"),
            ("Je vais bien.", "I am well.")
        ]
        prompt = create_sentence_translation_prompt(
            "Merci beaucoup",
            "French",
            "English",
            icl_examples
        )
        assert "Comment allez-vous?" in prompt
        assert "How are you?" in prompt
        assert "Merci beaucoup" in prompt


class TestUtils:
    """Test utility functions."""
    
    def test_clean_text(self):
        text = "  hello   world  "
        assert clean_text(text) == "hello world"
    
    def test_get_language_code(self):
        assert get_language_code("french") == "fra"
        assert get_language_code("english") == "eng"
        assert get_language_code("arabic") == "arb"
        assert get_language_code("fr") == "fra"
    
    def test_get_language_name(self):
        assert get_language_name("fra") == "French"
        assert get_language_name("eng") == "English"
        assert get_language_name("arb") == "Arabic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
