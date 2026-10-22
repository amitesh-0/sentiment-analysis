# 📓 Notebooks Directory (`notebooks/`)

This directory is intended for interactive Jupyter Notebooks (`.ipynb`) used for exploratory data analysis (EDA), visual experimentation, and rapid prototyping.

---

## 📌 Suggested Notebook Workflow

1. **`01_imdb_eda.ipynb`**:
   * Inspect dataset shape, column types, and missing values.
   * Analyze class balance between positive and negative reviews.
   * Review length distributions (character count & word count histograms).
   * Word clouds and most frequent n-grams in positive vs. negative reviews.

2. **`02_text_preprocessing.ipynb`**:
   * Experiment with diverse text pre-processing techniques (HTML/URL stripping, contraction expansion, character elongation, emojis/emoticons).
   * Evaluate the "Stopword Dilemma" in sentiment analysis (Naive vs Sentiment-Aware).
   * Compare Stemming (Porter, Snowball) vs Lemmatization (WordNet).
   * Implement Negation Marking / Scoping (`NOT_` tag propagation).
   * Production-grade, modular `TextPreprocessor` Scikit-Learn transformer class.

3. **`03_model_prototyping.ipynb`**:
   * Quick experiments comparing TF-IDF + Logistic Regression vs. Naive Bayes.
   * Confusion matrix visualizations and sample misclassified reviews.

4. **`04_error_analysis.ipynb`**:
   * In-depth inspection of false positives and false negatives (e.g., sarcasm, negations, subtle movie reviews).

---

Natural Language Processing (Text Vectorization Pipeline)
│
├── Sparse Representations
│   │
│   ├── Preprocessing & Tokenization
│   │   ├── Lowercasing & Punctuation removal
│   │   ├── Stop-word removal
│   │   ├── Stemming (cutting words to root forms)
│   │   ├── Lemmatization (converting to dictionary forms)
│   │   └── Word-level Tokenization (splitting by spaces)
│   │
│   └── Count vectors
│       ├── One hot encoding
│       ├── Bag of words
│       └── TF-IDF
│
└── Dense Representations 
    │
    ├── Static / Context-Free Embeddings
    │   │
    │   ├── Preprocessing & Tokenization
    │   │   ├── Word-level Tokenization
    │   │   ├── Building a fixed global vocabulary
    │   │   └── Character n-gram splitting (Specifically used before FastText)
    │   │
    │   └── Neural networks
    │       ├── Word2Vec
    │       ├── GloVe
    │       └── FastText
    │
    └── Dynamic / Contextual Embeddings
        │
        ├── Preprocessing & Tokenization (Subword Algorithms)
        │   ├── Byte Pair Encoding (BPE) (Standard for GPT models)
        │   ├── WordPiece (Standard for BERT models)
        │   └── SentencePiece / Unigram
        │
        └── Transformers
            ├── BERT
            ├── GPT
            └── Sentence transformers

## 💡 Guidelines
* **Keep notebooks clean:** Move reusable data cleaning and modeling logic into `src/` as soon as it stabilizes.
* **Imports:** Import functions directly from `src/` (e.g. `from src.data.preprocessor import TextPreprocessor`).
