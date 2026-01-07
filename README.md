# 🐾 RAG Chatbot

This project is a local **Retrieval-Augmented Generation (RAG)** chatbot. It uses a **Hybrid Search** approach, combining semantic (vector) search and lexical (keyword) search, fused together using the **Reciprocal Rank Fusion (RRF)** algorithm for maximum accuracy.

## Overview of Simple RAG Flow

<img width="190" height="319" alt="image" src="https://github.com/user-attachments/assets/0879e8d1-6ae9-45cc-998f-e169eede6d23" />


## 🚀 Features

* **Dual-Engine Retrieval:** Combines the "meaning" of your query (Embeddings) with "exact match" keywords (BM25).
* **RRF Fusion:** Merges search results using a rank-based system, making it robust against score outliers.
* **Fully Local:** Powered by **Ollama**—your data never leaves your machine.
* **Smart Chunking:** Automatic paragraph and line-based splitting with context overlap.
* **Fast NLP:** Uses **spaCy** for intelligent lemmatization and noise removal.

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **Ollama:** [Download here](https://ollama.com/)
3. **Required Models:**
Run the following commands in your terminal to download the necessary models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text

```



---

## 📦 Installation

1. **Clone or create your project folder.**
2. **Install Python dependencies:**
```bash
pip install numpy ollama rank_bm25 spacy

```


3. **Download the spaCy language model:**
```bash
python -m spacy download en_core_web_sm

```



---

## 📂 Project Structure

* `demo.py`: The main Python script.
* `cat.txt`: Your knowledge base (place your text here).
* `cat_index.pkl`: The automatically generated index (created on first run).

---

## 📖 How to Use

1. **Prepare your data:** Place a text file named `cat.txt` in the same directory as the script.
2. **Run the script:**
```bash
python demo.py

```


3. **Interact:** Type your questions into the terminal. The system will show you the "RRF Score" for the retrieved context before generating the AI answer.

---

## 🧠 Technical Deep-Dive: Hybrid RRF Search

### Why Hybrid?

* **Semantic (Embeddings):** Good for conceptual questions (e.g., "How does a cat feel?" matches "feline emotions").
* **Lexical (BM25):** Good for specific terms (e.g., "Aspirin dosage" matches the exact word "Aspirin").

### Reciprocal Rank Fusion (RRF)

Instead of trying to force different types of scores into the same range, RRF uses the **rank** (position) of the results.

The formula used in your code is:


* ** (60):** A smoothing constant that prevents high-ranking items from being too dominant.
* **Weights:** Adjusted via `ALPHA_EMB` (default 0.7) and `BETA_BM25` (default 0.3).

---

## ⚙️ Configuration

You can fine-tune the behavior at the top of `demo.py`:

| Constant | Default | Description |
| --- | --- | --- |
| `MAX_WORDS` | 200 | Maximum size of each text chunk. |
| `OVERLAP_WORDS` | 40 | Words shared between chunks to keep context. |
| `ALPHA_EMB` | 0.7 | Importance of the Semantic search (0.0 to 1.0). |
| `BETA_BM25` | 0.3 | Importance of the Keyword search (0.0 to 1.0). |
| `TOP_K` | 5 | Number of context chunks sent to the LLM. |

---


These chunks are then embedded and indexed for similarity-based retrieval in the RAG pipeline.

# 🧠 Methodology: Advanced Hybrid Retrieval

This RAG system implements a Hybrid Search architecture combined with Reciprocal Rank Fusion (RRF) to ensure the LLM receives the most relevant context possible.

## 1. Hybrid Search Strategy

Traditional search systems usually choose between keyword or semantic search. This project uses both in parallel to capture different types of relevance:

- **Sparse Retrieval (BM25):** Focuses on lexical overlap. It treats the document as a "bag of words" and uses the BM25 (Best Matching 25) algorithm. This is critical for finding specific terms, technical IDs, or unique names (e.g., `Product-XYZ-123`).
- **Dense Retrieval (Embeddings):** Focuses on semantic intent. It uses the `nomic-embed-text` model to convert text into a 768-dimensional vector space. This allows the system to find "car" even if the user typed "automobile."

## 2. Reciprocal Rank Fusion (RRF)

Merging scores from BM25 (which can be any positive number) and Cosine Similarity (which is 0.0 to 1.0) is mathematically difficult. This system utilizes RRF, a state-of-the-art ranking algorithm that ignores raw scores in favor of rank position.

### The formula

```text
RRFscore(d) = sum_{r in R} w_r / (k + rank_r(d))
```

Or expressed in LaTeX form:

$$
\text{RRFscore}(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}_r(d)}
$$

Where:
- `R`: The set of retrieval methods (BM25 and Vector).
- `rank_r(d)`: The position of document `d` in list `r` (1-indexed).
- `k`: A constant (set to `60`) that prevents the top-ranked results from dominating the fusion too heavily.
- `w_r`: The weight applied to the specific retrieval method (defined by `ALPHA_EMB` and `BETA_BM25`).

## 3. NLP Normalization Pipeline

To improve the accuracy of the BM25 lexical engine, the system processes both the query and the document chunks through a spaCy pipeline:

- **Lowercasing:** Standardizes all text.
- **Lemmatization:** Reduces words to their base form (e.g., "running" → "run").
- **Noise Removal:** Filters out stopwords (the, is, at), punctuation, and extra whitespace.

This ensures that a search for "cats" will successfully match a document containing the word "cat."

## 4. Generation & Grounding

Once the top `K` chunks are fused, they are injected into the LLM system prompt. We implement strict grounding instructions:

- The model is instructed to answer **ONLY** using the provided context.
- If the information is missing, the model is forced to admit it doesn't know rather than hallucinating.

---

Constants referenced: `ALPHA_EMB` (embedding weight), `BETA_BM25` (BM25 weight), and `K` (number of fused chunks).
