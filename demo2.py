import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

# Optional loaders
try:
    import pymupdf4llm
except Exception:
    pymupdf4llm = None

try:
    from docx import Document
except Exception:
    Document = None


# ---------------------------------------------------------
# 1) CONFIG
# ---------------------------------------------------------
class Config:
    DATA_DIR = Path("/Users/adriandarmali/RAG")
    INDEX_FILE = Path("/Users/adriandarmali/RAG/rag_index.pkl")

    EMBED_MODEL = "nomic-embed-text"
    LLM_MODEL = "llama3.2"

    # Parent–child chunking (small-to-big)
    PARENT_MAX_CHARS = 3500
    CHILD_MAX_CHARS = 900
    CHILD_OVERLAP_CHARS = 150

    # Hybrid retrieval (RRF)
    K_RRF = 60
    ALPHA_VEC = 0.7
    BETA_BM25 = 0.3

    # Retrieval sizing
    CANDIDATE_K = 50               # top child candidates after fusion/aggregation
    PRE_RERANK_PARENTS = 10        # how many parents to rerank
    FINAL_K = 5                    # how many parents to return

    # Two-stage reranking
    RERANK_ENABLED = True
    RERANK_CACHE_SIZE = 512

    # Index schema version
    INDEX_VERSION = 3


# ---------------------------------------------------------
# 2) TOKENIZER (simple, no spaCy needed)
# ---------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ---------------------------------------------------------
# 3) SAFE NUMERICS
# ---------------------------------------------------------
def safe_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector safely.
    If norm is 0 or not finite, return all-zeros vector (prevents NaN/Inf).
    """
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if (not np.isfinite(n)) or n == 0.0:
        return np.zeros_like(v, dtype=np.float32)
    return v / n


def sanitize_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Ensure embedding matrix has no NaN/Inf and is float32.
    """
    m = np.asarray(mat, dtype=np.float32)
    return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------
# 4) DOCUMENT PROCESSING
# ---------------------------------------------------------
class DocumentProcessor:
    @staticmethod
    def load_file(path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            if pymupdf4llm is None:
                print(f"⚠️ pymupdf4llm not installed, skipping PDF: {path.name}")
                return ""
            return pymupdf4llm.to_markdown(str(path))
        if ext == ".docx":
            if Document is None:
                print(f"⚠️ python-docx not installed, skipping DOCX: {path.name}")
                return ""
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        return ""

    @staticmethod
    def split_into_parents(text: str) -> List[str]:
        """
        Build parent chunks using paragraph boundaries first, then length cap.
        """
        text = (text or "").strip()
        if not text:
            return []

        paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        parents: List[str] = []
        buf = ""

        def flush():
            nonlocal buf
            if buf.strip():
                parents.append(buf.strip())
            buf = ""

        for p in paras:
            # hard-split huge paragraphs
            if len(p) > Config.PARENT_MAX_CHARS:
                flush()
                start = 0
                while start < len(p):
                    end = min(len(p), start + Config.PARENT_MAX_CHARS)
                    parents.append(p[start:end].strip())
                    start = end
                continue

            if len(buf) + len(p) + 2 <= Config.PARENT_MAX_CHARS:
                buf = f"{buf}\n\n{p}".strip() if buf else p
            else:
                flush()
                buf = p

        flush()
        return [x for x in parents if len(x) > 50]

    @staticmethod
    def split_parent_into_children(parent_text: str) -> List[str]:
        """
        Sliding window chunking on characters with overlap.
        """
        t = (parent_text or "").strip()
        if not t:
            return []

        kids: List[str] = []
        start = 0
        while start < len(t):
            end = min(len(t), start + Config.CHILD_MAX_CHARS)
            chunk = t[start:end].strip()
            if len(chunk) >= 40:
                kids.append(chunk)
            if end >= len(t):
                break
            start = max(0, end - Config.CHILD_OVERLAP_CHARS)

        return kids


# ---------------------------------------------------------
# 5) RAG ENGINE (Hybrid + Parent–Child + Optional Rerank)
# ---------------------------------------------------------
class RAGEngine:
    def __init__(self):
        self.parents: List[str] = []
        self.children: List[str] = []
        self.child_to_parent: List[int] = []

        self.embeddings: Optional[np.ndarray] = None  # child embeddings
        self.bm25: Optional[BM25Okapi] = None

        self._rerank_cache: Dict[Tuple[str, str], float] = {}

    # ---------- Embedding ----------
    def _embed(self, text: str) -> np.ndarray:
        resp = ollama.embeddings(model=Config.EMBED_MODEL, prompt=text)
        v = np.array(resp["embedding"], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_normalize(v)

    # ---------- Index ----------
    def build_index(self):
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        files = [f for f in Config.DATA_DIR.glob("*") if f.suffix.lower() in [".pdf", ".docx", ".txt"]]
        if not files:
            print("⚠️ No supported files found (.pdf, .docx, .txt).")
            return

        all_parents: List[str] = []
        for file in files:
            print(f"📄 Processing: {file.name}")
            text = DocumentProcessor.load_file(file)
            if not text:
                continue
            all_parents.extend(DocumentProcessor.split_into_parents(text))

        if not all_parents:
            print("🛑 No text extracted. (If PDFs are scans, convert them to searchable text first.)")
            return

        # Build parent–child mapping
        parents: List[str] = []
        children: List[str] = []
        child_to_parent: List[int] = []

        for p in all_parents:
            pid = len(parents)
            parents.append(p)
            kids = DocumentProcessor.split_parent_into_children(p)
            for k in kids:
                children.append(k)
                child_to_parent.append(pid)

        self.parents = parents
        self.children = children
        self.child_to_parent = child_to_parent

        print(f"🏗️ Parents: {len(self.parents)} | Children: {len(self.children)}")
        if not self.children:
            print("🛑 No child chunks produced.")
            return

        # Embed children
        print(f"🏗️ Embedding children (Ollama: {Config.EMBED_MODEL})...")
        vecs: List[np.ndarray] = []
        kept_children: List[str] = []
        kept_map: List[int] = []

        for i, ch in enumerate(self.children):
            try:
                v = self._embed(ch)
                # If embedding is all zeros, skip it (prevents garbage vectors)
                if np.allclose(v, 0.0):
                    continue
                vecs.append(v)
                kept_children.append(ch)
                kept_map.append(self.child_to_parent[i])
            except Exception as e:
                print(f"⚠️ Embedding failed on chunk {i}, skipping. Error: {e}")

        self.children = kept_children
        self.child_to_parent = kept_map

        if not vecs:
            print("🛑 No embeddings generated (all failed or zero).")
            return

        self.embeddings = sanitize_matrix(np.vstack(vecs))

        # BM25 on children
        print("🧮 Building BM25 on children...")
        tokenized = [tokenize(c) for c in self.children]
        self.bm25 = BM25Okapi(tokenized)

        payload = {
            "version": Config.INDEX_VERSION,
            "parents": self.parents,
            "children": self.children,
            "child_to_parent": self.child_to_parent,
            "emb": self.embeddings,
            "bm25": self.bm25,
        }
        with open(Config.INDEX_FILE, "wb") as f:
            pickle.dump(payload, f)

        print(f"✅ Index saved to {Config.INDEX_FILE}")

    def load_index(self) -> bool:
        if not Config.INDEX_FILE.exists():
            return False

        with open(Config.INDEX_FILE, "rb") as f:
            data = pickle.load(f)

        # Schema check
        if data.get("version") != Config.INDEX_VERSION:
            print("⚠️ Index schema mismatch. Delete rag_index.pkl and reindex.")
            return False

        self.parents = data["parents"]
        self.children = data["children"]
        self.child_to_parent = data["child_to_parent"]
        self.embeddings = sanitize_matrix(data["emb"])
        self.bm25 = data["bm25"]

        # Extra safety
        if self.embeddings is None or len(self.children) == 0 or self.bm25 is None:
            print("⚠️ Index appears incomplete. Delete rag_index.pkl and reindex.")
            return False

        return True

    # ---------- Retrieval core ----------
    def _rrf_child_scores(self, query: str) -> Dict[int, float]:
        if self.embeddings is None or self.bm25 is None:
            return {}

        q_emb = self._embed(query)
        if np.allclose(q_emb, 0.0):
            # query embedding invalid -> fall back to BM25 only
            v_ranks = np.array([], dtype=int)
        else:
            # dot product safely
            emb = self.embeddings
            v_scores = emb @ q_emb  # both float32, sanitized
            # handle any stray non-finite scores
            v_scores = np.nan_to_num(v_scores, nan=0.0, posinf=0.0, neginf=0.0)
            v_ranks = np.argsort(-v_scores)

        l_scores = self.bm25.get_scores(tokenize(query))
        l_scores = np.nan_to_num(np.asarray(l_scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        l_ranks = np.argsort(-l_scores)

        rrf: Dict[int, float] = {}
        # limit list sizes to keep work bounded
        vec_limit = max(Config.CANDIDATE_K * 3, 200)
        lex_limit = max(Config.CANDIDATE_K * 3, 200)

        for rank, idx in enumerate(v_ranks[:vec_limit]):
            rrf[idx] = rrf.get(idx, 0.0) + Config.ALPHA_VEC * (1.0 / (Config.K_RRF + rank + 1))

        for rank, idx in enumerate(l_ranks[:lex_limit]):
            rrf[idx] = rrf.get(idx, 0.0) + Config.BETA_BM25 * (1.0 / (Config.K_RRF + rank + 1))

        return rrf

    def _aggregate_to_parents(self, child_scores: Dict[int, float], top_child_n: int) -> List[int]:
        """
        Take top-N children by score and aggregate to parents via max score per parent.
        Returns parent_ids sorted descending by aggregated score.
        """
        ranked_children = sorted(child_scores.items(), key=lambda x: x[1], reverse=True)[:top_child_n]
        parent_best: Dict[int, float] = {}
        for cidx, score in ranked_children:
            pid = self.child_to_parent[cidx]
            if pid not in parent_best or score > parent_best[pid]:
                parent_best[pid] = score

        ranked_parents = sorted(parent_best.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked_parents]

    # ---------- Reranker ----------
    def _rerank_parents_llm(self, query: str, parent_ids: List[int]) -> List[int]:
        """
        Rerank parents using local LLM scorer (slow but fully local).
        Returns parent_ids sorted by rerank score desc.
        """
        def score_one(q: str, context: str) -> float:
            key = (q, context[:400])
            if key in self._rerank_cache:
                return self._rerank_cache[key]

            prompt = (
                "You are a relevance scorer for retrieval.\n"
                "Score how relevant the CONTEXT is for answering the QUESTION.\n"
                "Return ONLY a number from 0 to 100.\n\n"
                f"QUESTION:\n{q}\n\nCONTEXT:\n{context}\n"
            )
            try:
                resp = ollama.chat(
                    model=Config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                txt = resp["message"]["content"].strip()
                m = re.search(r"(\d+(\.\d+)?)", txt)
                val = float(m.group(1)) if m else 0.0
            except Exception:
                val = 0.0

            # bound cache size
            if len(self._rerank_cache) >= Config.RERANK_CACHE_SIZE:
                self._rerank_cache.pop(next(iter(self._rerank_cache)))
            self._rerank_cache[key] = val
            return val

        scored = [(pid, score_one(query, self.parents[pid])) for pid in parent_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored]

    # ---------- Public retrieve ----------
    def retrieve(self, query: str) -> List[str]:
        """
        1) Hybrid RRF over child chunks
        2) Aggregate to parents (small-to-big)
        3) Optional rerank parents (two-stage)
        4) Return top FINAL_K parent chunks
        """
        child_scores = self._rrf_child_scores(query)
        if not child_scores:
            return []

        parent_ids = self._aggregate_to_parents(child_scores, top_child_n=Config.CANDIDATE_K)
        if not parent_ids:
            return []

        # Stage 2: rerank top parents
        pre = parent_ids[: Config.PRE_RERANK_PARENTS]
        if Config.RERANK_ENABLED and pre:
            final_ids = self._rerank_parents_llm(query, pre)[: Config.FINAL_K]
        else:
            final_ids = pre[: Config.FINAL_K]

        return [self.parents[pid] for pid in final_ids]


# ---------------------------------------------------------
# 6) CHAT INTERFACE
# ---------------------------------------------------------
def main():
    engine = RAGEngine()

    if engine.load_index():
        print(f"📂 Loaded index from {Config.INDEX_FILE}")
        print("💡 Added new files? Type 'reindex' to rebuild.")
    else:
        engine.build_index()

    if not engine.parents:
        print("🛑 System offline: No documents loaded.")
        return

    print("\n--- 🐾 Hybrid RAG Chatbot (Parent–Child + Optional Rerank) ---")
    print("Commands: 'reindex' to rebuild, 'exit' to quit.\n")

    while True:
        query = input("YOU: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        if query.lower() == "reindex":
            if Config.INDEX_FILE.exists():
                os.remove(Config.INDEX_FILE)
            engine.build_index()
            continue

        contexts = engine.retrieve(query)
        context_text = "\n\n---\n\n".join(contexts)

        prompt = (
            "Answer the question using ONLY the context below.\n"
            "If the context does not contain the answer, say you don't know.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER:"
        )

        print("\nASSISTANT: ", end="", flush=True)
        try:
            stream = ollama.chat(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                print(chunk["message"]["content"], end="", flush=True)
        except Exception as e:
            print(f"\n❌ LLM Error: {e}")
        print("\n")


if __name__ == "__main__":
    main()

