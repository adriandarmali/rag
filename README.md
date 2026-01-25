# 🐾 Local Hybrid RAG Chatbot

**Hybrid RRF • Parent–Child (Small-to-Big) • Reranking • Retrieval Evaluation**

A fully local **Retrieval-Augmented Generation (RAG)** chatbot that combines:

* **Dense retrieval** using embeddings (semantic search)
* **Sparse retrieval** using BM25 (keyword search)
* **Reciprocal Rank Fusion (RRF)** to merge ranked results

It also implements:

* **Small-to-big retrieval (parent–child chunking)**: index small chunks for accurate matching, return larger parent context for generation
* **Optional two-stage retrieval (reranking)**: rerank top contexts to reduce noisy evidence
* **Retrieval evaluation** with a golden set using **evidence anchors**, reporting **Recall@k** and **MRR**

Runs locally with **Ollama**—your data stays on your machine.

<img width="196" height="315" alt="image" src="https://github.com/user-attachments/assets/a62472fb-635e-48ad-b494-d1041ef8d3bd" />


<img width="510" height="354" alt="image" src="https://github.com/user-attachments/assets/ae584933-ff39-49e3-b3fc-3fcb2992c695" />



---

## ✅ What it does

1. Load documents from a local folder (`.txt`, optional `.pdf`, `.docx`)
2. Chunk documents into:

   * **Parent chunks** (larger context for generation)
   * **Child chunks** (smaller chunks indexed for retrieval)
3. Build indexes:

   * Embeddings index (semantic retrieval)
   * BM25 index (keyword retrieval)
  
     <img width="510" height="354" alt="image" src="https://github.com/user-attachments/assets/6bd81bcb-8ad6-4165-8a8d-5c3293734bd2" />

4. Retrieve candidate **child chunks** using hybrid search + RRF
5. Expand to **parent chunks** (small-to-big)
6. **Optional:** rerank top parent chunks (two-stage retrieval)
7. Generate an answer using **only retrieved context**

---

## 🛠️ Requirements

* Python 3.10+
* Ollama installed and running

Pull models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## 📦 Installation

Minimum:

```bash
pip install numpy ollama rank_bm25
```

Optional (PDF/DOCX ingestion):

```bash
pip install pymupdf4llm python-docx
```

---

## 📂 Project structure

* `demo2.py` — main chatbot + indexing + retrieval (**hybrid RRF + parent–child + optional rerank**)
* `rag_index.pkl` — saved index (created during indexing)
* `eval_retrieval.py` — retrieval evaluation script (**Recall@k, MRR**)
* `groundtruth_with_anchor.csv` — golden set with evidence anchors
* `retrieval_results.csv` — per-question debug output (hit rank + preview)

---

## ▶️ Quickstart

1. Run the chatbot:

```bash
/opt/anaconda3/bin/python /Users/adriandarmali/RAG/demo2.py
```

2. Reindex (**inside the chatbot**, at the `YOU:` prompt):

```text
reindex
```

3. Exit:

```text
exit
```

---

## ✅ Retrieval evaluation (Recall@k + MRR)

Run evaluation:

```bash
/opt/anaconda3/bin/python /Users/adriandarmali/RAG/eval_retrieval.py
```

Sure — here’s a deeper, GitHub-ready explanation of the **retrieval evaluation methodology** you implemented, why it works, and how to interpret the outputs.

---

## Evaluation methodology

This project evaluates the **retrieval layer** of the RAG system separately from the generation layer. That is intentional: in RAG, most hallucinations and wrong answers come from either (a) retrieving irrelevant context or (b) missing the relevant context entirely. If retrieval is weak, generation quality cannot be trusted.

### What we are evaluating

For each question in a test set, we run:

1. **Retriever** returns a ranked list of contexts (parent chunks)
2. We check whether any of the retrieved contexts contains a known piece of evidence for the correct answer
3. We compute ranking metrics over all questions

This isolates *retrieval quality* (finding the right evidence) from *generation quality* (writing the final answer).

---

## Golden set format

Each evaluation row includes:

* `id`: unique question id
* `question`: user query
* `answer`: human-readable answer (for reference)
* `source`: which file the answer comes from (optional)
* `evidence_anchor`: a short evidence string expected to appear verbatim in the correct passage

Example:

```csv
id,question,answer,source,evidence_anchor
003,On average what fraction of the day do cats spend sleeping?,Two thirds,cat.txt,2/3
028,Did Felicette survive the trip?,Yes,cat.txt,Felicette
```

### Why we use evidence anchors (instead of answer strings)

Many correct answers are hard to match directly:

* **Yes/No answers** might not appear literally as “Yes”
* Paraphrasing and formatting differs: “Two thirds” vs “2/3”
* The answer may be implied, while an entity term is explicitly present

So we use `evidence_anchor` as a **verbatim evidence string** that is likely to appear inside the relevant chunk.

Best anchors are:

* unique entities (names, locations, product terms)
* numbers/dates/ratios exactly as written
* short key phrases that appear in the text

Avoid anchors that are too generic, like:

* “Do”, “Yes”, “No”, “cat” (too many false hits)

---

## Oracle coverage (answerable rate)

Before measuring retrieval quality, the script computes **oracle coverage**:

> Does the evidence anchor appear anywhere in the indexed corpus?

This answers: **How many questions are even possible to retrieve correctly?**

In your results:

* Rows answerable: **131/200**

So **69 questions are not answerable** given the current combination of indexed documents and evidence anchors.

This is important because no retriever can exceed oracle coverage. If your oracle coverage is 0.655, Recall@10 cannot realistically exceed ~0.655 unless you fix the dataset/index mismatch.

---

## Hit definition

A query is considered a “hit” if:

* The anchor appears as a substring in any retrieved chunk (after normalization)

Normalization includes:

* lowercasing
* removing punctuation
* collapsing whitespace
* keeping `/` so “2/3” still matches

This is deliberately simple and strict: we are checking whether the retrieved chunk contains evidence that can support the answer.

---

## Metrics used

### Recall@k

Recall@k answers:

> For what fraction of questions does the correct evidence appear in the top k retrieved chunks?

For example:

* Recall@5 = 0.645 means **64.5%** of questions had the evidence somewhere in the top 5 contexts.

Recall is a **coverage metric** — it is mainly improved by:

* better chunking
* bigger candidate pools
* better indexing/extraction
* better anchors

### MRR (Mean Reciprocal Rank)

MRR answers:

> On average, how high is the first correct hit ranked?

For each question:

* if first hit is rank 1 → score = 1.0
* rank 2 → 0.5
* rank 5 → 0.2
* no hit → 0

MRR is a **ranking quality metric** — it improves when:

* the retriever ranks the right chunk higher
* reranking works well
* fusion weights are tuned
* noise is reduced

---

## Why we report “all” vs “answerable-only”

Your script reports two sets of metrics:

### 1) Metrics on **all questions**

These include unanswerable rows, so they represent end-to-end reality of your dataset.

Example:

* Recall@10 (all) = 0.655

This is strongly limited by oracle coverage.

### 2) Metrics on **answerable-only questions**

These measure “pure retriever performance” when evidence exists in the corpus.

The results:

* Recall@10 (answerable) = **1.000**
* MRR (answerable) = **0.906**

Meaning: for questions where evidence exists, your retriever almost always finds it, and usually ranks it at #1.

This split is extremely useful for diagnosing what to improve:

* If answerable-only scores are weak → retrieval/ranking problem
* If answerable-only is strong but all is weak → coverage/anchors/indexing problem

---

## Summary of what the numbers mean

Given:

* Answerable = 131/200
* Recall@10 (all) = 0.655 ≈ oracle coverage
* Recall@10 (answerable) = 1.000
* MRR (answerable) = 0.906

Interpretation:

* The retriever is **good** on content that exists in the index
* The main bottleneck is **coverage**: 69 anchors/questions are not supported by the indexed corpus or anchors do not match the text



### Latest results

Rows read: **200**
Rows answerable: **131**
Recall@5 (all): **0.645**
Recall@10 (all): **0.655**
MRR (all): **0.593**
Recall@5 (answerable): **0.985**
Recall@10 (answerable): **1.000**
MRR (answerable): **0.906**

**Interpretation**

* “Answerable” means the anchor exists somewhere in the indexed corpus.
* Retrieval quality is excellent on answerable questions.
* The main limiter for “all” metrics is **coverage** (anchors/corpus mismatch for 69 questions).


