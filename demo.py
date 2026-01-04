from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import ollama
from rank_bm25 import BM25Okapi
import spacy

# ----------------------------
# Config
# ----------------------------
FILE_PATH = Path(__file__).with_name("cat.txt")
INDEX_PATH = Path(__file__).with_name("cat_index.pkl")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

MAX_WORDS = 200
OVERLAP_WORDS = 40
TOP_K = 5

# RRF Configuration
K_RRF = 60         # Smoothing constant (Standard value)
ALPHA_EMB = 0.7    # Weight for Semantic Search (Embeddings)
BETA_BM25 = 0.3    # Weight for Lexical Search (BM25)

# ----------------------------
# spaCy NLP setup
# ----------------------------
def load_spacy() -> spacy.language.Language:
    try:
        # Disable unused components for performance
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )

NLP = load_spacy()

def normalize_for_search(text: str) -> List[str]:
    """Lemmatizes and removes noise for lexical matching."""
    doc = NLP(text.lower())
    return [
        t.lemma_.strip() for t in doc 
        if not (t.is_stop or t.is_punct or t.is_space) and t.lemma_.strip()
    ]

# ----------------------------
# Text loading + Chunking
# ----------------------------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def split_into_blocks(text: str) -> List[str]:
    """Splits raw text into meaningful blocks (paragraphs or lines)."""
    lines = text.replace("\r\n", "\n").split("\n")
    blank_ratio = sum(1 for ln in lines if not ln.strip()) / max(len(lines), 1)
    
    if blank_ratio > 0.05:
        # Paragraph-based
        text = re.sub(r"\n\s*\n+", "\n\n", text.strip())
        return [re.sub(r"\s+", " ", b).strip() for b in text.split("\n\n") if b.strip()]
    else:
        # Line-based
        return [ln.strip() for ln in lines if ln.strip()]

def chunk_blocks(blocks: List[str], max_words: int, overlap_words: int) -> List[str]:
    """Combines blocks into chunks of specific word length with overlap."""
    chunks, cur_words = [], []
    cur_len = 0

    for block in blocks:
        bw = block.split()
        if len(bw) > max_words:
            if cur_words: chunks.append(" ".join(cur_words))
            start = 0
            while start < len(bw):
                end = min(start + max_words, len(bw))
                chunks.append(" ".join(bw[start:end]))
                start = max(0, end - overlap_words)
            cur_words, cur_len = [], 0
            continue

        if cur_words and (cur_len + len(bw) > max_words):
            chunks.append(" ".join(cur_words))
            cur_words = cur_words[-overlap_words:] if overlap_words > 0 else []
            cur_len = len(cur_words)

        cur_words.extend(bw)
        cur_len += len(bw)

    if cur_words: chunks.append(" ".join(cur_words))
    return chunks

# ----------------------------
# Embeddings & Indexing
# ----------------------------
def embed_one(text: str) -> np.ndarray:
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(resp["embedding"], dtype=np.float32)

def ensure_index(file_path: Path) -> Dict[str, Any]:
    stat = file_path.stat()
    meta_now = {"mtime": stat.st_mtime, "size": stat.st_size, "model": EMBED_MODEL}

    if INDEX_PATH.exists():
        with INDEX_PATH.open("rb") as f:
            idx = pickle.load(f)
            if idx.get("meta") == meta_now: return idx

    print(f"Indexing {file_path}... this might take a moment.")
    chunks = chunk_blocks(split_into_blocks(load_text(file_path)), MAX_WORDS, OVERLAP_WORDS)
    
    emb_list = [embed_one(ch) for ch in chunks]
    emb_matrix = np.vstack(emb_list)
    
    # Normalize for cosine similarity calculation
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.clip(norms, 1e-12, None)

    tokenized = [normalize_for_search(ch) for ch in chunks]
    bm25 = BM25Okapi(tokenized)

    idx = {"chunks": chunks, "emb": emb_matrix, "bm25": bm25, "meta": meta_now}
    with INDEX_PATH.open("wb") as f:
        pickle.dump(idx, f)
    return idx

# ----------------------------
# Hybrid Retrieval (RRF)
# ----------------------------
def retrieve(index: Dict[str, Any], query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """
    Combines Embedding and BM25 ranks using Reciprocal Rank Fusion.
    """
    # 1. Semantic Scores (Cosine Similarity)
    qv = embed_one(query)
    qv = qv / max(np.linalg.norm(qv), 1e-12)
    emb_sims = index["emb"] @ qv
    emb_ranks = np.argsort(-emb_sims)  # Indices of highest similarity first

    # 2. Lexical Scores (BM25)
    q_tokens = normalize_for_search(query)
    bm25_scores = np.array(index["bm25"].get_scores(q_tokens), dtype=np.float32)
    bm25_ranks = np.argsort(-bm25_scores) # Indices of highest lexical score first

    # 3. RRF Calculation
    # RRF Score = Sum ( Weight / (K_RRF + Rank) )
    rrf_map: Dict[int, float] = {}

    # Accumulate score from Embedding ranks
    for rank, idx in enumerate(emb_ranks):
        rrf_map[idx] = rrf_map.get(idx, 0) + ALPHA_EMB * (1.0 / (K_RRF + rank + 1))

    # Accumulate score from BM25 ranks
    for rank, idx in enumerate(bm25_ranks):
        rrf_map[idx] = rrf_map.get(idx, 0) + BETA_BM25 * (1.0 / (K_RRF + rank + 1))

    # 4. Final Sorting
    sorted_indices = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for i in range(min(top_k, len(sorted_indices))):
        idx, score = sorted_indices[i]
        results.append((index["chunks"][idx], score))
    return results

# ----------------------------
# Chat Logic
# ----------------------------
def stream_chat(system_prompt: str, user_prompt: str) -> None:
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
    )
    for part in stream:
        print(part['message']['content'], end="", flush=True)
    print()

def main() -> None:
    if not FILE_PATH.exists():
        print(f"File not found: {FILE_PATH}. Please provide a cat.txt file."); return

    index = ensure_index(FILE_PATH)
    print(f"Ready! Loaded {len(index['chunks'])} chunks.\n")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}: break
        if not q: continue

        hits = retrieve(index, q, top_k=TOP_K)

        print("\n--- Retrieved Knowledge (RRF Scores) ---")
        for chunk, score in hits:
            # Displaying preview of the chunk
            print(f"[{score:.5f}] {chunk[:100].replace('\n', ' ')}...")

        context_text = "\n".join([f"- {c}" for c, _ in hits])
        sys_prompt = (
            "You are a helpful AI assistant. Use the following context to answer the question.\n"
            "If the answer is not in the context, say you don't know based on the provided data.\n"
            "Be concise and professional.\n\n"
            f"Context:\n{context_text}"
        )

        print("\n--- Assistant Response ---")
        stream_chat(sys_prompt, q)

if __name__ == "__main__":
    main()








