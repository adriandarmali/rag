# demo.py
# RAG chatbot with:
# - chunking (blocks -> chunks with overlap)
# - embeddings retrieval (Ollama)
# - NLP normalization (spaCy)
# - hybrid retrieval (Embeddings + BM25)
# - robust streaming output
#
# Run: python demo.py

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
# Config (EDIT THESE)
# ----------------------------
FILE_PATH = Path(__file__).with_name("cat.txt")  # put cat.txt next to demo.py or set absolute path
INDEX_PATH = Path(__file__).with_name("cat_index.pkl")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

MAX_WORDS = 200
OVERLAP_WORDS = 40
TOP_K = 5

# Hybrid retrieval weighting (tune if needed)
ALPHA_EMB = 0.7   # weight for embedding similarity
BETA_BM25 = 0.3   # weight for BM25 lexical score


# ----------------------------
# spaCy NLP setup
# ----------------------------
def load_spacy() -> spacy.language.Language:
    # Disable components we don't need for speed
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except OSError as e:
        raise RuntimeError(
            "spaCy model not found.\n"
            "Run:\n  python -m spacy download en_core_web_sm"
        ) from e
    return nlp

NLP = load_spacy()


def normalize_for_search(text: str) -> List[str]:
    """
    NLP normalization for lexical search:
    - lowercase
    - lemmatize
    - remove stopwords/punct/space
    Returns token list suitable for BM25.
    """
    doc = NLP(text.lower())
    toks = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        lemma = t.lemma_.strip()
        if lemma:
            toks.append(lemma)
    return toks


# ----------------------------
# Text loading + chunking
# ----------------------------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_line_blocks(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return [ln for ln in lines if ln]


def split_into_paragraph_blocks(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text.strip())
    blocks = [re.sub(r"\s+", " ", b).strip() for b in text.split("\n\n")]
    return [b for b in blocks if b]


def chunk_blocks(blocks: List[str], max_words: int, overlap_words: int) -> List[str]:
    chunks: List[str] = []
    cur_words: List[str] = []
    cur_len = 0

    for block in blocks:
        bw = block.split()
        b_len = len(bw)

        # Oversized block -> split by words with overlap
        if b_len > max_words:
            if cur_words:
                chunks.append(" ".join(cur_words))
                cur_words, cur_len = [], 0

            start = 0
            while start < b_len:
                end = min(start + max_words, b_len)
                chunks.append(" ".join(bw[start:end]))
                start = max(0, end - overlap_words)
            continue

        # Flush if adding exceeds budget
        if cur_words and (cur_len + b_len > max_words):
            chunks.append(" ".join(cur_words))
            if overlap_words > 0:
                cur_words = cur_words[-overlap_words:]
                cur_len = len(cur_words)
            else:
                cur_words, cur_len = [], 0

        cur_words.extend(bw)
        cur_len += b_len

    if cur_words:
        chunks.append(" ".join(cur_words))

    return chunks


def build_chunks(file_path: Path) -> List[str]:
    text = load_text(file_path)

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blank = sum(1 for ln in lines if not ln.strip())
    use_lines = blank < max(3, int(0.02 * len(lines)))

    blocks = split_into_line_blocks(text) if use_lines else split_into_paragraph_blocks(text)
    chunks = chunk_blocks(blocks, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS)

    if not chunks:
        raise ValueError("No chunks produced. Check that the input file is not empty.")

    print(f"Detected format: {'line-based' if use_lines else 'paragraph-based'}")
    print(f"Blocks: {len(blocks)} | Chunks: {len(chunks)}")
    return chunks


# ----------------------------
# Embeddings + indexing
# ----------------------------
def embed_one(text: str) -> np.ndarray:
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    except ollama._types.ResponseError as e:
        if "not found" in str(e).lower():
            raise RuntimeError(
                f'Embedding model "{EMBED_MODEL}" not found.\n'
                f"Run: ollama pull {EMBED_MODEL}\n"
                f"Or set EMBED_MODEL to a model you have (see: ollama list)."
            ) from e
        raise
    return np.array(resp["embedding"], dtype=np.float32)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def ensure_index(file_path: Path) -> Dict[str, Any]:
    stat = file_path.stat()
    meta_now = {
        "file_path": str(file_path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "embed_model": EMBED_MODEL,
        "max_words": MAX_WORDS,
        "overlap_words": OVERLAP_WORDS,
    }

    if INDEX_PATH.exists():
        try:
            with INDEX_PATH.open("rb") as f:
                idx = pickle.load(f)
            if idx.get("meta") == meta_now:
                return idx
        except Exception:
            pass  # rebuild if corrupted

    chunks = build_chunks(file_path)

    # Embeddings
    emb_list: List[np.ndarray] = []
    for i, ch in enumerate(chunks, start=1):
        emb_list.append(embed_one(ch))
        if i % 50 == 0 or i == len(chunks):
            print(f"Embedded {i}/{len(chunks)} chunks...")

    emb = normalize_rows(np.vstack(emb_list))

    # NLP tokens for BM25 (lexical index)
    tokenized = [normalize_for_search(ch) for ch in chunks]
    bm25 = BM25Okapi(tokenized)

    idx = {
        "chunks": chunks,
        "emb": emb,
        "bm25": bm25,
        "bm25_tokens": tokenized,  # kept for debugging/inspection
        "meta": meta_now,
    }

    with INDEX_PATH.open("wb") as f:
        pickle.dump(idx, f)

    return idx


# ----------------------------
# Hybrid retrieval (Embeddings + BM25)
# ----------------------------
def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin, xmax = float(x.min()), float(x.max())
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def retrieve(index: Dict[str, Any], query: str, top_k: int = TOP_K) -> List[Tuple[str, float, float, float]]:
    """
    Returns list of (chunk, final_score, emb_score, bm25_score).
    """
    # Embedding similarity
    qv = embed_one(query)
    qv = qv / max(np.linalg.norm(qv), 1e-12)
    emb_sims = index["emb"] @ qv  # cosine sims

    # BM25 lexical score
    q_tokens = normalize_for_search(query)
    bm25_scores = np.array(index["bm25"].get_scores(q_tokens), dtype=np.float32)

    # Normalize + combine
    emb_n = minmax(emb_sims.astype(np.float32))
    bm25_n = minmax(bm25_scores)

    final = ALPHA_EMB * emb_n + BETA_BM25 * bm25_n

    k = min(top_k, final.shape[0])
    top_idx = np.argpartition(-final, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-final[top_idx])]

    results = []
    for i in top_idx:
        results.append((index["chunks"][i], float(final[i]), float(emb_sims[i]), float(bm25_scores[i])))
    return results


# ----------------------------
# Chat
# ----------------------------
def build_system_prompt(retrieved: List[Tuple[str, float, float, float]]) -> str:
    context = "\n".join([f"- {chunk}" for chunk, *_ in retrieved])
    return (
        "You are a helpful chatbot.\n"
        "Use ONLY the context below to answer.\n"
        "If the answer is not present, say \"I don't know based on the provided context.\".\n\n"
        f"Context:\n{context}\n"
    )


def stream_chat(model: str, system_prompt: str, user_prompt: str) -> None:
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    printed_any = False
    for part in stream:
        content = ""
        if isinstance(part, dict):
            content = ((part.get("message") or {}).get("content") or part.get("response") or "")
        else:
            content = (
                getattr(getattr(part, "message", None), "content", "")
                or getattr(part, "response", "")
                or ""
            )

        if content:
            printed_any = True
            print(content, end="", flush=True)

    if not printed_any:
        print("[No tokens returned from model stream]")
    print()


def main() -> None:
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FILE_PATH}")

    print(f"Loading/building index for: {FILE_PATH}")
    index = ensure_index(FILE_PATH)
    print(f"Ready. Total chunks: {len(index['chunks'])}\n")

    while True:
        q = input("Ask me a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        hits = retrieve(index, q, top_k=TOP_K)

        print("\nRetrieved knowledge (hybrid scoring):")
        for chunk, final_s, emb_s, bm25_s in hits:
            preview = chunk[:160].replace("\n", " ")
            print(f" - final={final_s:.3f} | emb={emb_s:.3f} | bm25={bm25_s:.3f} | {preview}{'...' if len(chunk)>160 else ''}")

        system_prompt = build_system_prompt(hits)

        print("\nChatbot response:")
        stream_chat(LLM_MODEL, system_prompt, q)
        print()


if __name__ == "__main__":
    main()








