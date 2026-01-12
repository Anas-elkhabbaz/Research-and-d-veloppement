# SM-UMT: Self-Mining Unsupervised Machine Translation

Implementation of the **Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs** (NAACL 2025) by El Mekki & Abdul-Mageed.

---

## 📖 Overview

This system enables **unsupervised machine translation** by automatically mining in-context learning (ICL) examples without requiring human-annotated parallel data. Traditional machine translation requires large parallel corpora, but this approach generates synthetic parallel data from monolingual text using a two-stage self-mining process.

### Key Innovation

The paper introduces a novel approach where LLMs can translate between languages **without any parallel training data** by:
1. Mining word-level translations to create synthetic parallel sentences
2. Using these synthetic pairs as in-context examples for sentence translation
3. Applying TopK+BM25 filtering to select the most relevant examples

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SM-UMT Translation Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: Word-Level Mining                              │   │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────────┐  │   │
│  │  │  Source    │───►│   Word     │───►│  LLM Word      │  │   │
│  │  │  Sentences │    │ Extraction │    │  Translation   │  │   │
│  │  └────────────┘    └────────────┘    └───────┬────────┘  │   │
│  │                                              │           │   │
│  │                    ┌───────────────────────▼─────────┐   │   │
│  │                    │  Synthetic Parallel Data        │   │   │
│  │                    │  (word-by-word translations)    │   │   │
│  │                    └─────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: Sentence-Level Mining                          │   │
│  │  ┌────────────────┐   ┌────────────┐   ┌─────────────┐   │   │
│  │  │  Sentence      │──►│   TopK     │──►│    BM25     │   │   │
│  │  │  Embeddings    │   │ Selection  │   │  Re-ranking │   │   │
│  │  └────────────────┘   │  (top 20)  │   └──────┬──────┘   │   │
│  │                       └────────────┘          │          │   │
│  │                    ┌────────────────────────▼─────────┐  │   │
│  │                    │  Filtered ICL Examples (k=8)     │  │   │
│  │                    │  with threshold τ=0.90           │  │   │
│  │                    └──────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TRANSLATION                                              │   │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────────┐  │   │
│  │  │   Query    │───►│  Prompt    │───►│     LLM        │  │   │
│  │  │  Sentence  │    │ +ICL Exs   │    │  Generation    │  │   │
│  │  └────────────┘    └────────────┘    └───────┬────────┘  │   │
│  │                                              │           │   │
│  │                    ┌───────────────────────▼─────────┐   │   │
│  │                    │  Translated Output              │   │   │
│  │                    └─────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology (from paper)

### Stage 1: Word-Level Mining

1. **Word Extraction**: Extract content words from source monolingual sentences
2. **LLM Translation**: Use the LLM to translate individual words with in-context examples
3. **Synthetic Parallel Data**: Create word-by-word translated parallel pairs

**Why this works**: Individual word translations are generally more reliable than full sentences for an LLM without parallel training data.

### Stage 2: Sentence-Level Mining (TopK+BM25)

1. **Sentence Embeddings**: Compute embeddings using multilingual sentence-transformers
2. **TopK Selection**: Select top-20 most similar sentences from the synthetic corpus
3. **Filtering**: Apply similarity threshold τ=0.90 to remove noisy pairs
4. **BM25 Re-ranking**: Use BM25 to select the final k=8 most relevant ICL examples

**Why TopK+BM25**: Combines semantic similarity (embeddings) with lexical matching (BM25) for better example selection.

---

## ✨ Features

- 🌍 **Multilingual Support**: French ↔ English and Arabic ↔ English
- 🤖 **Gemini API**: Uses Google's free Gemini 2.5 Flash model
- 📊 **BLEU Evaluation**: Built-in evaluation with sacrebleu
- 🔧 **Configurable**: All hyperparameters from the paper are adjustable
- 📁 **Modular Design**: Clean separation of components for easy extension

---

## 🚀 Installation

```bash
# Navigate to project directory
cd "c:\Users\Lenovo\Desktop\S9\Projet integre"

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `google-genai` - Gemini API client
- `sentence-transformers` - Multilingual sentence embeddings
- `sacrebleu` - BLEU score evaluation
- `torch` - PyTorch for embeddings
- `tqdm` - Progress bars

---

## ⚙️ Setup

1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) for a free Gemini API key

2. **Set API Key**:
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Or pass directly via --api-key argument
python main.py --api-key "your-api-key" ...
```

---

## 📋 Usage

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

### FLORES-200 Evaluation

```bash
python main.py --evaluate --use-flores --src fra --tgt eng --sample_size 100
```

### Quick Test

```bash
python main.py --test
```

### List Languages

```bash
python main.py --list-langs
```

---

## 📁 Project Structure

```
Projet integre/
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── README.md               # This documentation
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_sm_umt.py      # 15 test cases
└── sm_umt/                 # Main package
    ├── __init__.py         # Package initialization
    ├── config.py           # Hyperparameters from paper
    ├── prompts.py          # LLM prompt templates
    ├── llm_client.py       # Gemini API client
    ├── word_mining.py      # Stage 1: Word-level mining
    ├── sentence_mining.py  # Stage 2: TopK+BM25 selection
    ├── bm25.py             # BM25 ranking algorithm
    ├── translator.py       # Main translation pipeline
    ├── evaluation.py       # BLEU score evaluation
    └── utils.py            # Utility functions
```

---

## 📊 Key Hyperparameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kwp` | 10 | Number of word pairs for word-level ICL |
| `k` | 8 | Number of sentence-level ICL examples |
| `τ` (tau) | 0.90 | Similarity threshold for filtering |
| `top_n` | 20 | Top-N candidates before BM25 selection |

These can be modified in `sm_umt/config.py`.

---

## 🐍 Python API

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

## 📈 Results

Tested on sample data:

| Language Pair | BLEU Score | Notes |
|---------------|------------|-------|
| French → English | 7.41 | 3 sample sentences |
| Arabic → English | 9.57 | 5 sample sentences |

**Note**: Higher BLEU scores are expected with larger sample sizes and more ICL examples.

---

## 🔍 How It Works (Step by Step)

### Example: Translating "Bonjour le monde" (French → English)

1. **Word Extraction**: Extract words ["bonjour", "monde"]

2. **Word Translation**: 
   - "bonjour" → "hello"
   - "monde" → "world"

3. **Synthetic Parallel Creation**:
   - "Bonjour le monde" → "hello le world" (word-by-word)

4. **ICL Mining**: Find similar sentences from synthetic pairs using TopK+BM25

5. **Translation Prompt**:
   ```
   Translate from French to English:
   
   Examples:
   French: Comment allez-vous?
   English: how allez you
   
   French: Bonjour le monde
   English: [LLM generates: "Hello world"]
   ```

6. **Output**: "Hello world"

---

## 📚 Citation

```bibtex
@inproceedings{elmekki2025effective,
  title={Effective Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs},
  author={El Mekki, Abdellah and Abdul-Mageed, Muhammad},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  year={2025}
}
```

---

## 📝 License

This implementation is for educational and research purposes.

---

## 🤝 Acknowledgments

- Based on research by Abdellah El Mekki and Muhammad Abdul-Mageed (UBC-NLP)
- Uses Google's Gemini API for LLM inference
- Sentence embeddings via sentence-transformers
