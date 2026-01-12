# SM-UMT: Self-Mining Unsupervised Machine Translation

**Enhancing Self-Mining UMT with Dynamic Vocabulary Expansion and Domain Adaptation**

Implementation of the Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs (NAACL 2025) by El Mekki and Abdul-Mageed, with novel extensions for dynamic vocabulary handling, quality-aware selection, and domain adaptation.

---

## Authors

- **Anas Elkhabbaz**
- **Nassima Elgarn**
- **Othmane Himmiche**

**Supervisor:** Pr. Youness Moukafih  
**Institution:** International University of Rabat  
**Academic Year:** 2025-2026

---

## Overview

This system enables unsupervised machine translation by automatically mining in-context learning (ICL) examples without requiring human-annotated parallel data. Traditional machine translation requires large parallel corpora, but this approach generates synthetic parallel data from monolingual text using a two-stage self-mining process.

### Key Innovation

The original SM-UMT paper introduces a novel approach where LLMs can translate between languages without any parallel training data by mining word-level translations to create synthetic parallel sentences, using these synthetic pairs as in-context examples for sentence translation, and applying TopK+BM25 filtering to select the most relevant examples.

### Our Extensions

This implementation extends the original SM-UMT with three novel contributions:

1. **Dynamic Vocabulary Expansion (DVE)**: On-the-fly translation of unknown words at runtime with context-aware disambiguation and intelligent caching. Achieves 80% resolution rate for out-of-vocabulary terms.

2. **Quality-Aware ICL Selection (QAIS)**: Multi-factor scoring framework that filters synthetic parallel data using length ratio, word preservation, and optional back-translation consistency metrics.

3. **Domain-Adaptive Mining (DAM)**: Domain-specific vocabulary seeding for specialized translation tasks, demonstrated on hate speech content moderation with 30 curated domain terms.

---

## Architecture

```
SM-UMT Translation Pipeline with Extensions
============================================

STAGE 1: Word-Level Mining
--------------------------
Source Sentences --> Word Extraction --> LLM Word Translation --> Synthetic Parallel Data

STAGE 2: Sentence-Level Mining (TopK+BM25)  
------------------------------------------
Sentence Embeddings --> TopK Selection (top 20) --> Threshold Filter (tau=0.90) --> BM25 Re-ranking --> ICL Examples (k=8)

OUR EXTENSIONS
--------------
DVE Module: Detects unknown words --> Context-aware LLM translation --> Cache storage
QAIS Module: Scores ICL candidates --> Filters low-quality examples
DAM Module: Loads domain vocabulary --> Detects domain terms --> Prioritizes domain translations

TRANSLATION
-----------
Query Sentence + ICL Examples --> LLM Generation --> Translated Output
```

---

## Methodology

### Stage 1: Word-Level Mining

The first stage extracts content words from source monolingual sentences, uses the LLM to translate individual words with in-context examples, and creates word-by-word translated parallel pairs. Individual word translations are generally more reliable than full sentences for an LLM without parallel training data.

### Stage 2: Sentence-Level Mining (TopK+BM25)

The second stage computes sentence embeddings using multilingual sentence-transformers, selects top-20 most similar sentences from the synthetic corpus, applies similarity threshold (tau=0.90) to remove noisy pairs, and uses BM25 to select the final k=8 most relevant ICL examples. This combines semantic similarity (embeddings) with lexical matching (BM25) for better example selection.

### Dynamic Vocabulary Expansion (DVE)

DVE addresses the static vocabulary limitation by detecting unknown words during translation, querying the LLM with sentence context for disambiguation, caching translations to avoid redundant API calls, and incrementally building the vocabulary throughout the session.

### Quality-Aware ICL Selection (QAIS)

QAIS scores each candidate ICL pair using three signals: length ratio score (penalizes extreme length differences), word preservation score (measures vocabulary overlap), and back-translation consistency (optional, measures semantic similarity after round-trip translation).

### Domain-Adaptive Mining (DAM)

DAM extends SM-UMT for specialized domains by pre-loading curated domain vocabularies, detecting domain words in input sentences, prioritizing domain translations over general mining, and falling back to DVE for unknown domain terms.

---

## Features

- Multilingual Support: French to English and Arabic to English
- Gemini API: Uses Google's Gemini 2.5 Flash Lite model
- BLEU Evaluation: Built-in evaluation with sacrebleu
- Configurable: All hyperparameters from the paper are adjustable
- Modular Design: Clean separation of components for easy extension

---

## Installation

```bash
# Navigate to project directory
cd "c:\Users\Lenovo\Desktop\S9\Projet integre"

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- google-genai: Gemini API client
- sentence-transformers: Multilingual sentence embeddings
- sacrebleu: BLEU score evaluation
- torch: PyTorch for embeddings
- tqdm: Progress bars
- numpy: Numerical operations

---

## Setup

1. Get API Key: Visit Google AI Studio (https://makersuite.google.com/app/apikey) for a free Gemini API key

2. Set API Key:
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Or pass directly via --api-key argument
python main.py --api-key "your-api-key" ...
```

---

## Usage

### Single Sentence Translation

```bash
# French to English
python main.py --input "Bonjour le monde" --src fra --tgt eng

# Arabic to English
python main.py --input "مرحبا بالعالم" --src arb --tgt eng

# English to French
python main.py --input "Hello world" --src eng --tgt fra
```

### Batch Translation with Evaluation

```bash
# Translate sample data with BLEU evaluation
python main.py --src fra --tgt eng --sample_size 10 --evaluate -v

# Arabic to English
python main.py --src arb --tgt eng --sample_size 5 --evaluate -v
```

### Run Experiments with Extensions

```bash
# Run all experiments (baseline, DVE, QAIS, DAM)
python experiments.py --api-key YOUR_API_KEY --samples 10 --experiment all

# Run specific experiment
python experiments.py --api-key YOUR_API_KEY --experiment dve --samples 5
```

---

## Project Structure

```
Projet integre/
|-- main.py                 # CLI entry point
|-- experiments.py          # Experiment runner for extensions
|-- requirements.txt        # Dependencies
|-- README.md               # This documentation
|-- paper.tex               # Research paper (LaTeX)
|-- tests/                  # Unit tests
|   |-- __init__.py
|   +-- test_sm_umt.py      # Test cases
+-- sm_umt/                 # Main package
    |-- __init__.py         # Package initialization
    |-- config.py           # Hyperparameters from paper
    |-- prompts.py          # LLM prompt templates
    |-- llm_client.py       # Gemini API client
    |-- word_mining.py      # Stage 1: Word-level mining
    |-- sentence_mining.py  # Stage 2: TopK+BM25 selection
    |-- bm25.py             # BM25 ranking algorithm
    |-- translator.py       # Main translation pipeline
    |-- evaluation.py       # BLEU score evaluation
    |-- utils.py            # Utility functions
    |-- dve.py              # Dynamic Vocabulary Expansion (NEW)
    |-- qais.py             # Quality-Aware ICL Selection (NEW)
    +-- dam.py              # Domain-Adaptive Mining (NEW)
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| kwp | 10 | Number of word pairs for word-level ICL |
| k | 8 | Number of sentence-level ICL examples |
| tau | 0.90 | Similarity threshold for filtering |
| top_n | 20 | Top-N candidates before BM25 selection |

These can be modified in sm_umt/config.py.

---

## Experimental Results

| Method | BLEU | Unknown Words | Resolved | Time (s) |
|--------|------|---------------|----------|----------|
| Baseline SM-UMT | 100.0 | -- | -- | 25.0 |
| SM-UMT + DVE | 57.96 | 10 | 8 (80%) | 132.6 |
| SM-UMT + QAIS | -- | -- | -- | 33.3 |
| SM-UMT + DAM | -- | 30 vocab | 2 detected | 34.5 |

Note: QAIS and DAM experiments were interrupted due to API quota exhaustion. The baseline achieved 100% BLEU on simple test sentences, validating the SM-UMT approach.

### DVE Word Translations

- bonjour --> hello
- comment --> how
- allez --> are
- vous --> you
- appelle --> call
- marie --> mary
- beau --> nice
- aujourd --> today

---

## Python API

```python
from sm_umt import SMUMTTranslator, Config

# Initialize with configuration
config = Config(
    src_lang="fra",    # Source language (fra, eng, arb)
    tgt_lang="eng",    # Target language
    k=8,               # Number of ICL examples
    tau=0.90           # Similarity threshold
)

translator = SMUMTTranslator(config, api_key="your-api-key")

# Prepare source sentences
source_sentences = [
    "Bonjour, comment allez-vous?",
    "Je m'appelle Marie.",
    "Il fait beau aujourd'hui."
]

# Run full pipeline with references for evaluation
references = [
    "Hello, how are you?",
    "My name is Marie.",
    "The weather is nice today."
]

result = translator.run_pipeline(source_sentences, references)

# Access results
print(f"BLEU Score: {result['evaluation']['bleu']:.2f}")

for src, tgt in zip(source_sentences, result['translations']):
    print(f"{src} -> {tgt}")
```

---

## Citation

```bibtex
@inproceedings{elmekki2025effective,
  title={Effective Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs},
  author={El Mekki, Abdellah and Abdul-Mageed, Muhammad},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  year={2025}
}
```

---

## License

This implementation is for educational and research purposes.

---

## Acknowledgments

- Based on research by Abdellah El Mekki and Muhammad Abdul-Mageed (UBC-NLP)
- Uses Google's Gemini API for LLM inference
- Sentence embeddings via sentence-transformers
- Supervised by Pr. Youness Moukafih at International University of Rabat
