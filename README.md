# 🐾 Local Hybrid RAG Chatbot

> A low-cost, privacy-friendly AI research assistant for internal documents

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-black.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

This project is a fully local **Retrieval-Augmented Generation (RAG)** chatbot that helps users search, compare, and use information across large sets of documents more reliably.

Instead of guessing, it searches uploaded files, pulls the most relevant passages, and uses that evidence to answer questions — like a smarter internal research assistant.

<img width="196" alt="Chatbot screenshot" src="https://github.com/user-attachments/assets/a62472fb-635e-48ad-b494-d1041ef8d3bd" />
<img width="510" alt="Architecture diagram" src="https://github.com/user-attachments/assets/ae584933-ff39-49e3-b3fc-3fcb2992c695" />

---

## ✨ Key Benefits

| Benefit | Description |
|---|---|
| 💰 **Low cost** | Runs locally on a MacBook using the free version of Ollama — no paid API required |
| 🧠 **Lower hallucination risk** | Answers are grounded in retrieved evidence, not unsupported guesses |
| 🔒 **Privacy-first** | Documents stay on your local machine and are never sent to external platforms |

---

## 💼 Business Use Cases

Many teams already have the information they need — but it's buried across decks, reports, notes, PDFs, and internal files. The real problem is often not a lack of data, but the time required to find, compare, and act on it.

This project reduces that friction by making internal knowledge easier to search and use.

**Investment & Finance**
Search across company decks, diligence notes, internal memos, lender reports, and market research to support first-pass screening, memo writing, and portfolio review.

**Operations & Internal Teams**
Search SOPs, project notes, and internal documents without manually digging through folders.

**Legal, Compliance & Policy**
Pull relevant passages from long documents before drafting summaries or preparing reviews.

---

## 🔧 What Makes This System Different

This chatbot combines multiple retrieval methods to improve answer quality:

- **Dense retrieval** — embeddings-based semantic search
- **Sparse retrieval** — BM25 keyword search
- **Reciprocal Rank Fusion (RRF)** — merges ranked results from both methods
- **Parent-child chunking (small-to-big retrieval)** — small chunks improve matching accuracy; larger parent chunks provide generation context
- **Optional reranking** — reduces noisy evidence
- **Retrieval evaluation** — golden set with evidence anchors, reporting **Recall@k** and **MRR**

The goal is not just to produce answers, but to make them more grounded, more useful, and less likely to hallucinate.

---

## ✅ How It Works

1. Load documents from a local folder (`.txt`, optional `.pdf`, `.docx`)
2. Chunk documents into:
   - **Parent chunks** — larger context for generation
   - **Child chunks** — smaller units for more accurate retrieval
3. Build indexes:
   - Embeddings index for semantic retrieval
   - BM25 index for keyword retrieval

<img width="510" alt="Indexing diagram" src="https://github.com/user-attachments/assets/6bd81bcb-8ad6-4165-8a8d-5c3293734bd2" />

4. Retrieve candidate **child chunks** using hybrid search + RRF
5. Expand to **parent chunks** for broader context
6. *(Optional)* Rerank top parent chunks to reduce noise
7. Generate an answer using **only retrieved context**

---

## 🛠️ Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## 📦 Installation

**Minimum install:**

```bash
pip install numpy ollama rank_bm25
```

**Optional — PDF and DOCX ingestion:**

```bash
pip install pymupdf4llm python-docx
```

---

## 📂 Project Structure

```
.
├── demo2.py                    # Main chatbot: indexing, retrieval, generation (hybrid RRF + parent-child + optional rerank)
├── rag_index.pkl               # Saved index (created during indexing)
├── eval_retrieval.py           # Retrieval evaluation script (Recall@k, MRR)
├── groundtruth_with_anchor.csv # Golden set with evidence anchors
└── retrieval_results.csv       # Per-question debug output (hit rank + preview)
```

---

## ▶️ Quickstart

**1. Run the chatbot:**

```bash
/opt/anaconda3/bin/python /Users/adriandarmali/RAG/demo2.py
```

**2. Reindex documents** (at the `YOU:` prompt):

```
reindex
```

**3. Exit:**

```
exit
```

---

## 📊 Retrieval Evaluation

Run evaluation (Recall@k + MRR):

```bash
/opt/anaconda3/bin/python /Users/adriandarmali/RAG/eval_retrieval.py
```

### Methodology

This project evaluates the **retrieval layer** separately from the generation layer. In RAG systems, many hallucinations trace back to retrieving irrelevant — or missing the relevant — context. If retrieval is weak, generation quality cannot be trusted.

For each question in the test set:
1. The retriever returns a ranked list of contexts
2. The system checks whether any retrieved context contains a known piece of evidence
3. Ranking metrics are computed across all questions

### Golden Set Format

Each evaluation row contains:

| Field | Description |
|---|---|
| `id` | Unique question ID |
| `question` | User query |
| `answer` | Human-readable reference answer |
| `source` | Source file name (if applicable) |
| `evidence_anchor` | Short string expected to appear in the correct passage |

**Example:**

```csv
id,question,answer,source,evidence_anchor
003,On average what fraction of the day do cats spend sleeping?,Two thirds,cat.txt,2/3
028,Did Felicette survive the trip?,Yes,cat.txt,Felicette
```

### Why Evidence Anchors?

Many correct answers are difficult to match directly — "Yes/No" may not appear literally, formatting varies (`Two thirds` vs `2/3`), or the answer is implied. Evidence anchors make retrieval evaluation **stricter and more reliable**.

**Good anchors:** unique names, entities, numbers, dates, ratios, short distinctive phrases

**Bad anchors:** generic terms like `Yes`, `No`, `cat`

---

## 📈 Metrics

### Recall@k
> For what fraction of questions does the correct evidence appear in the top *k* retrieved chunks?

Recall@5 = 0.645 → 64.5% of questions had evidence in the top 5 results.

Improved by: better chunking, stronger indexing, better evidence anchors.

### MRR (Mean Reciprocal Rank)
> On average, how highly is the first correct hit ranked?

| Rank | Score |
|------|-------|
| 1 | 1.000 |
| 2 | 0.500 |
| 5 | 0.200 |
| No hit | 0.000 |

Improved by: better ranking, effective reranking, reduced retrieval noise.

---

## 🏆 Latest Results

| Metric | All Questions | Answerable Only |
|---|---|---|
| **Rows evaluated** | 200 | 131 |
| **Recall@5** | 0.645 | 0.985 |
| **Recall@10** | 0.655 | **1.000** |
| **MRR** | 0.593 | **0.906** |

> **Oracle coverage:** 131 / 200 questions answerable (65.5%). No retriever can exceed this ceiling without improving corpus coverage, indexing, or anchors.

### Interpretation

- Retrieval quality is **excellent on answerable questions** (Recall@10 = 1.000)
- The primary limiter for overall metrics is **corpus coverage** — some anchors or questions don't match the indexed corpus
- If answerable-only metrics are weak → the problem is retrieval or ranking
- If answerable-only metrics are strong but overall metrics are weak → the problem is coverage, anchors, or indexing

---

## ⚠️ Limitations

This project is intentionally built to be **completely free**. Running locally on a MacBook with Ollama keeps costs at zero and preserves document privacy, but creates real performance constraints compared to larger hosted models:

- **Speed** — local inference is slower
- **Reasoning quality** — smaller models have tighter limits
- **Context handling** — reduced context window vs. frontier models
- **Scalability** — single-machine setup

This version is a **practical, privacy-friendly proof of concept** — not a fully optimized production system. Substantial performance gains are available by upgrading infrastructure:

- **Colab Pro** or higher-compute environments for faster indexing and evaluation
- **Paid APIs (e.g., OpenAI)** for improved reasoning, fluency, and reliability

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
