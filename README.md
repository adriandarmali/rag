# 🐾 Hybrid RAG Chatbot (RRF Optimized)

This project is a local **Retrieval-Augmented Generation (RAG)** chatbot. It uses a **Hybrid Search** approach, combining semantic (vector) search and lexical (keyword) search, fused together using the **Reciprocal Rank Fusion (RRF)** algorithm for maximum accuracy.

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
