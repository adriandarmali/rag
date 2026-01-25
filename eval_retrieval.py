import csv
import re
from typing import List, Optional, Tuple

from demo2 import RAGEngine, Config


# ----------------------------
# Normalization
# ----------------------------
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s/]", " ", s)  # keep /
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Anchor-based hit detection
# ----------------------------
def find_hit_rank_anchor(retrieved_contexts: List[str], anchor: str) -> Optional[int]:
    a = normalize(anchor)
    if not a:
        return None
    for i, ctx in enumerate(retrieved_contexts, start=1):
        if a in normalize(ctx):
            return i
    return None


# ----------------------------
# Metrics
# ----------------------------
def recall_at_k(hit_ranks: List[Optional[int]], k: int) -> float:
    if not hit_ranks:
        return 0.0
    hits = sum(1 for r in hit_ranks if r is not None and r <= k)
    return hits / len(hit_ranks)


def mrr(hit_ranks: List[Optional[int]]) -> float:
    if not hit_ranks:
        return 0.0
    return sum((1.0 / r) if r is not None else 0.0 for r in hit_ranks) / len(hit_ranks)


# ----------------------------
# Robust CSV reader
# ----------------------------
def open_csv_robust(path: str):
    f = open(path, newline="", encoding="utf-8-sig")

    sample = f.read(4096).replace("，", ",").replace("；", ";")
    f.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    except Exception:
        dialect = csv.excel
        dialect.delimiter = ","

    reader = csv.DictReader(f, dialect=dialect)
    fieldnames = reader.fieldnames or []

    # Handle single header cell like: "id,question,answer,source,evidence_anchor"
    if len(fieldnames) == 1 and "id" in fieldnames[0].lower() and "," in fieldnames[0]:
        raw_header = fieldnames[0].strip().strip('"')
        header = [h.strip() for h in raw_header.split(",")]

        f.seek(0)
        lines = f.read().splitlines()
        while lines and not lines[0].strip():
            lines = lines[1:]
        data_lines = lines[1:]

        def gen_rows():
            for line in data_lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                parts = [p.strip() for p in line.split(",")]
                parts = (parts + [""] * len(header))[: len(header)]
                yield dict(zip(header, parts))

        return f, gen_rows(), header

    return f, reader, fieldnames


# ----------------------------
# Oracle coverage (answerable rate)
# ----------------------------
def oracle_coverage(engine: RAGEngine, csv_path: str) -> Tuple[int, int]:
    parents = getattr(engine, "parents", []) or []
    corpus = normalize("\n\n".join(parents))

    total = 0
    present = 0

    fh, reader, _ = open_csv_robust(csv_path)
    try:
        for row in reader:
            row = {str(k).strip().lstrip("\ufeff").lower(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}

            anchor = (row.get("evidence_anchor") or row.get("answer") or "").strip()
            if not anchor:
                continue
            total += 1
            if normalize(anchor) in corpus:
                present += 1
    finally:
        fh.close()

    return present, total


# ----------------------------
# Main
# ----------------------------
def main():
    # IMPORTANT: evaluation should not call LLM reranker (too slow)
    if hasattr(Config, "RERANK_ENABLED"):
        Config.RERANK_ENABLED = False

    # Evaluate using top 10 contexts
    Config.FINAL_K = 10
    Config.CANDIDATE_K = max(getattr(Config, "CANDIDATE_K", 20), 50)

    engine = RAGEngine()
    if not engine.load_index():
        print("❌ Could not load index. Delete rag_index.pkl and reindex using demo2.py.")
        return

    csv_path = "/Users/adriandarmali/RAG/groundtruth_with_anchor.csv"

    present, total = oracle_coverage(engine, csv_path)
    print(f"Oracle coverage (anchor exists anywhere): {present}/{total}")

    hit_all: List[Optional[int]] = []
    hit_answerable: List[Optional[int]] = []

    rows_read = 0
    rows_used = 0
    rows_skipped = 0
    rows_answerable = 0
    rows_errors = 0

    rows_out = []

    parents = getattr(engine, "parents", []) or []
    corpus = normalize("\n\n".join(parents))

    fh, reader, fields = open_csv_robust(csv_path)
    try:
        print("CSV columns:", fields)

        for row in reader:
            rows_read += 1
            row = {str(k).strip().lstrip("\ufeff").lower(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}

            qid = (row.get("id") or "").strip()
            q = (row.get("question") or "").strip()
            anchor = (row.get("evidence_anchor") or row.get("answer") or "").strip()

            if not q or not anchor:
                rows_skipped += 1
                continue

            rows_used += 1
            answerable = normalize(anchor) in corpus
            if answerable:
                rows_answerable += 1

            try:
                retrieved = engine.retrieve(q)
            except Exception as e:
                rows_errors += 1
                retrieved = []
                rank = None
                rows_out.append({
                    "id": qid,
                    "question": q,
                    "anchor_used": anchor,
                    "hit_rank": "",
                    "top1_preview": "",
                    "error": str(e),
                })
                hit_all.append(None)
                if answerable:
                    hit_answerable.append(None)
                continue

            rank = find_hit_rank_anchor(retrieved, anchor)

            hit_all.append(rank)
            if answerable:
                hit_answerable.append(rank)

            top1_preview = " ".join(retrieved[0][:180].splitlines()) if retrieved else ""
            rows_out.append({
                "id": qid,
                "question": q,
                "anchor_used": anchor,
                "hit_rank": rank if rank is not None else "",
                "top1_preview": top1_preview,
                "error": "",
            })

    finally:
        fh.close()

    print(f"Rows read: {rows_read}")
    print(f"Rows used: {rows_used}")
    print(f"Rows skipped (empty): {rows_skipped}")
    print(f"Rows answerable: {rows_answerable}")
    print(f"Rows retrieval errors: {rows_errors}")

    r5_all = recall_at_k(hit_all, 5)
    r10_all = recall_at_k(hit_all, 10)
    mrr_all = mrr(hit_all)

    r5_ans = recall_at_k(hit_answerable, 5)
    r10_ans = recall_at_k(hit_answerable, 10)
    mrr_ans = mrr(hit_answerable)

    print(f"Recall@5 (all):         {r5_all:.3f}")
    print(f"Recall@10 (all):        {r10_all:.3f}")
    print(f"MRR (all):              {mrr_all:.3f}")
    print(f"Recall@5 (answerable):  {r5_ans:.3f}")
    print(f"Recall@10 (answerable): {r10_ans:.3f}")
    print(f"MRR (answerable):       {mrr_ans:.3f}")

    out_path = "/Users/adriandarmali/RAG/retrieval_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        fieldnames = ["id", "question", "anchor_used", "hit_rank", "top1_preview", "error"]
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
