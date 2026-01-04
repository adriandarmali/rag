# Basic Retrieval-Augmented Generator (RAG) AI Model  
**Adrian Darmali**

---

## Overview
This project implements a basic **Retrieval-Augmented Generation (RAG)** pipeline using:

- A local text knowledge base (`cat.txt`)
- Ollama for **embeddings** and **chat generation**
- A lightweight, local “vector database” stored as a **pickle index**
- Cosine similarity retrieval for top-K context selection

---

## End-to-End Methodology

### 1) Load the Source File
- Read `cat.txt` from disk as raw text.

---

### 2) Preprocess & Chunk the Text
#### Format Detection
The pipeline detects the file structure:

- **Line-based (one fact per line)**  
  If the file is mostly non-empty lines with few blank lines.

- **Paragraph-based (blank-line separated)**  
  If the file contains blank lines separating blocks of text.

#### Block Construction (Atomic Units)
- Split the text into **blocks** based on detected format.
- Blocks are treated as “atomic” to avoid splitting logical units.

#### Chunk Construction (Size-Controlled)
- Build chunks by **greedily packing blocks** into ~`MAX_WORDS` (e.g., ~200 words).
- Add **overlap** between consecutive chunks (e.g., `OVERLAP_WORDS = 40`) to reduce boundary loss.
- If a single block exceeds the max size, split it into multiple chunks with overlap.

---

### 3) Embed & Index the Chunks (Vector Database)
- Generate an embedding vector for each chunk using an Ollama embedding model (e.g., `nomic-embed-text`).
- **L2-normalize embeddings** so cosine similarity is efficient.
- Persist `{chunks, embeddings, metadata}` locally using `pickle` as an on-disk vector index.

---

### 4) Query-Time Retrieval
- Embed the user question using the same embedding model.
- Compute **cosine similarity** between the query embedding and all chunk embeddings.
- Return the **top-K** most similar chunks as retrieved context.

---

### 5) Answer Generation (Grounded Response)
- Construct a system prompt that instructs the chatbot to:
  - Use **only** the retrieved chunks
  - Avoid making up new information
- Call an Ollama chat model (e.g., `llama3.2`) with:
  - **System message**: retrieved context + grounding rules
  - **User message**: the question
- Stream the model output directly to the terminal.

---

### 6) Operational Behavior
- Reuse the saved index on subsequent runs if the source file and settings haven’t changed  
  (checked via `mtime`, file size, model name, and chunk parameters).
- Rebuild embeddings and index only when the source data or configuration changes.

---

## Limitations & Improvements

### Limitation 1 — Multi-topic questions may under-retrieve
If a query spans multiple topics, similarity retrieval may not surface all relevant chunks because it ranks results based on a single embedding query.

**Potential improvements**
- Query rewriting: have the chatbot generate a better search query from user input
- Multi-query retrieval: generate multiple sub-queries and merge results

---

### Limitation 2 — Top-K by cosine similarity isn’t always best
Cosine similarity ranking may return chunks that are close semantically but not maximally relevant, especially when each chunk contains many facts.

**Potential improvements**
- Add a **reranking model** to reorder retrieved chunks by direct relevance to the query

---

### Limitation 3 — Not scalable for large datasets
The current setup loads and searches embeddings in memory.

**Potential improvements**
- Use a production-grade vector database such as:
  - Qdrant
  - Pinecone
  - pgvector (Postgres)

---

## Appendix: Chunking Methodology

This project converts raw text into retrieval-ready chunks using a **blocks → chunks** approach with **overlap**.

Goals:
- Preserve semantic meaning (don’t split facts)
- Keep chunk sizes consistent (better embeddings)
- Reduce boundary loss (context preserved across chunks)

---

### 1) Split Text into Blocks (Atomic Units)
The pipeline auto-detects file structure:

#### Line-based blocks (one fact per line)
- Used when the file contains very few blank lines.
- Each non-empty line becomes one block.
- Example: each line = one fact.

#### Paragraph-based blocks (blank-line separated)
- Used when the file uses blank lines to separate items.
- Each paragraph becomes one block.
- Multi-line facts remain together.

All blocks are cleaned:
- Trimmed
- Empty blocks removed
- Whitespace normalized

---

### 2) Pack Blocks into Fixed-Size Chunks (Greedy Packing)
Blocks are combined into chunks using a target word budget:

- `MAX_WORDS` (e.g., 200): maximum approximate chunk size in words

Algorithm:
1. Start a new chunk  
2. Append blocks sequentially until adding the next block would exceed `MAX_WORDS`  
3. Finalize the current chunk and start the next one  

Why greedy packing:
- Simple and fast
- Preserves original ordering
- Works well for short factual datasets

---

### 3) Add Overlap Between Chunks
To reduce boundary loss:

- `OVERLAP_WORDS` (e.g., 40): number of words copied from the end of the previous chunk into the next chunk

Why overlap helps:
- If key information lands near a boundary, overlap increases the chance that at least one retrieved chunk contains full context.

Trade-off:
- Overlap introduces duplication, so it should be moderate (~10–25% of `MAX_WORDS`).

---

### 4) Handle Oversized Blocks
If a single block exceeds `MAX_WORDS`, it is split into multiple chunks with overlap so that:
- no chunk becomes excessively large
- no content is dropped

---

### Output
The chunking stage produces a list of chunks that are:
- **Semantically coherent**
- **Size-controlled**
- **Boundary-robust**

These chunks are then embedded and indexed for similarity-based retrieval in the RAG pipeline.
