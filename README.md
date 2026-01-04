# rag
Basic RAG model using Ollama chatbot with NLP embedded.

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Basic Retrieval Augmented Generator AI Model — README</title>
  <style>
    :root {
      --max-width: 900px;
      --accent: #0366d6;
      --muted: #586069;
      --bg: #ffffff;
      --code-bg: #f6f8fa;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", "Courier New", monospace;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      color: #24292f;
      background: #f7f8fa;
      margin: 0;
      padding: 32px;
      display: flex;
      justify-content: center;
    }

    .container {
      max-width: var(--max-width);
      background: var(--bg);
      padding: 36px;
      border-radius: 8px;
      box-shadow: 0 6px 18px rgba(35, 45, 60, 0.08);
    }

    header h1 {
      margin: 0 0 6px 0;
      font-size: 28px;
      line-height: 1.2;
    }

    header p.author {
      margin: 0 0 18px 0;
      color: var(--muted);
    }

    h2 {
      margin-top: 26px;
      color: #111827;
      border-bottom: 1px solid #e6edf3;
      padding-bottom: 8px;
    }

    p, li {
      line-height: 1.6;
      margin: 8px 0;
    }

    ul, ol {
      margin: 8px 0 8px 20px;
    }

    pre {
      background: var(--code-bg);
      padding: 14px;
      border-radius: 6px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.45;
      border: 1px solid #e1e4e8;
    }

    code {
      font-family: var(--mono);
      background: rgba(27,31,35,0.05);
      padding: 0.15em 0.25em;
      border-radius: 4px;
      font-size: 0.95em;
    }

    .note {
      background: #fff8c6;
      border: 1px solid #ffec99;
      padding: 12px;
      border-radius: 6px;
      color: #5a4a00;
      margin: 12px 0;
    }

    footer {
      margin-top: 28px;
      color: var(--muted);
      font-size: 13px;
    }

    .example {
      background: #f1f8ff;
      border-left: 4px solid var(--accent);
      padding: 12px;
      border-radius: 4px;
      margin: 8px 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Basic Retrieval Augmented Generator AI Model</h1>
      <p class="author">Adrian Darmali</p>
    </header>

    <main>
      <section>
        <h2>Overview</h2>
        <p>This project implements a simple Retrieval-Augmented Generation (RAG) pipeline. It converts source text into retrieval-ready chunks, builds embeddings for those chunks, and at query time retrieves the most relevant chunks to ground a generative model response.</p>
      </section>

      <section>
        <h2>Pipeline</h2>

        <h3>Load the source file</h3>
        <p>Read <code>cat.txt</code> from disk as raw text.</p>

        <h3>Preprocess + chunk the text</h3>
        <p>Detect format:</p>
        <ul>
          <li>If the file is mostly non-empty lines with few blank lines → treat it as <strong>one-fact-per-line</strong>.</li>
          <li>Otherwise → treat it as <strong>paragraph / blank-line separated content</strong>.</li>
        </ul>

        <p>Split into blocks (atomic units) using the chosen format. Build chunks by greedily packing blocks up to a target size (e.g., ~200 words). Add overlap between consecutive chunks (e.g., 40 words) to preserve context across boundaries. If any single block exceeds the max size, split it into multiple chunks with overlap.</p>

        <h3>Embed and index the chunks (vector database)</h3>
        <ul>
          <li>Generate an embedding vector for each chunk using an Ollama embedding model (e.g., <code>nomic-embed-text</code>).</li>
          <li>L2-normalize embeddings so cosine similarity is efficient.</li>
          <li>Persist <code>{chunks, embeddings, metadata}</code> locally (pickle) as an on-disk vector index.</li>
        </ul>

        <h3>Query-time retrieval</h3>
        <p>Take the user question and generate its embedding. Compute cosine similarity between the query embedding and all chunk embeddings. Select the top-K most similar chunks as the retrieved context.</p>

        <h3>Answer generation (grounded response)</h3>
        <p>Construct a system prompt instructing the chatbot to use only the retrieved chunks. Call an Ollama chat model (e.g., <code>llama3.2</code>) with:</p>
        <ul>
          <li><strong>system message</strong> = retrieved context + grounding rule</li>
          <li><strong>user message</strong> = the question</li>
        </ul>
        <p>Stream the model output to the terminal.</p>

        <h3>Operational behavior</h3>
        <ul>
          <li>Reuse the saved index on subsequent runs if the source file and settings haven’t changed (mtime/size/model/params match).</li>
          <li>Rebuild embeddings and index only when the source data or configuration changes.</li>
        </ul>
      </section>

      <section>
        <h2>Limitations</h2>
        <ul>
          <li>If the question covers multiple topics at the same time, the system may not provide a good answer because the retriever selects chunks based only on similarity to the query without deeper query decomposition.</li>
          <li>Possible solutions:
            <ul>
              <li>Have the chatbot write its own focused queries based on the user's input, then retrieve using those queries.</li>
              <li>Use multiple queries to retrieve more relevant information.</li>
            </ul>
          </li>
          <li>The top-N results are returned based on cosine similarity — this may not always give the best results. A reranking model can help re-rank retrieved chunks by relevance.</li>
          <li>The database is currently stored in memory which may not scale for very large datasets. Consider more efficient vector DBs (Qdrant, Pinecone, pgvector).</li>
        </ul>
      </section>

      <section>
        <h2>Appendix — Chunking Methodology</h2>
        <p>This project converts raw text into retrieval-ready chunks using a blocks → chunks approach with overlap. The goal is to preserve semantic meaning (don’t split facts), keep chunk sizes consistent (better embeddings), and reduce boundary issues (context isn’t lost between chunks).</p>

        <h3>1) Split text into blocks (atomic units)</h3>
        <p>Before chunking, the text is split into blocks—units that should stay intact whenever possible. The pipeline auto-detects the file structure:</p>
        <ul>
          <li><strong>Line-based blocks (one fact per line)</strong><br>
            Used when the file contains very few blank lines. Each non-empty line becomes one block. Example: each line = one fact.
          </li>
          <li><strong>Paragraph-based blocks (blank-line separated)</strong><br>
            Used when the file uses blank lines to separate items. Each paragraph becomes one block. Multi-line facts remain together.</li>
        </ul>
        <p>All blocks are cleaned (trimmed, empty blocks removed, whitespace normalized) to improve embedding stability.</p>

        <h3>2) Pack blocks into fixed-size chunks (greedy packing)</h3>
        <p>Blocks are combined into chunks using a target word budget:</p>
        <ul>
          <li><code>MAX_WORDS</code> (e.g., 200): maximum approximate chunk size in words</li>
        </ul>
        <p>Algorithm (greedy packing):</p>
        <ol>
          <li>Start a new chunk.</li>
          <li>Append blocks sequentially until adding the next block would exceed <code>MAX_WORDS</code>.</li>
          <li>Finalize the current chunk and start the next one.</li>
        </ol>
        <p>This is greedy (simple, fast) and preserves the original ordering of the source content.</p>

        <h3>3) Add overlap to preserve context across chunk boundaries</h3>
        <p>To reduce boundary loss, consecutive chunks share trailing context:</p>
        <ul>
          <li><code>OVERLAP_WORDS</code> (e.g., 40): number of words copied from the end of the previous chunk into the start of the next chunk.</li>
        </ul>
        <p>Why this helps: If important information lands near a chunk boundary, overlap increases the chance that at least one retrieved chunk contains the complete context needed to answer a question.</p>
        <p>Trade-off: Overlap introduces duplication across chunks — keep it moderate (~10–25% of <code>MAX_WORDS</code>).</p>

        <h3>4) Handle oversized blocks</h3>
        <p>If a single block exceeds <code>MAX_WORDS</code> (e.g., one very long paragraph), it is split into multiple chunks using the same overlap rule. This ensures:</p>
        <ul>
          <li>No chunk becomes excessively large.</li>
          <li>No content is dropped.</li>
        </ul>

        <h3>Output</h3>
        <p>The chunking stage produces a list of chunk strings that are:</p>
        <ul>
          <li>Semantically coherent (facts/paragraphs kept intact when possible)</li>
          <li>Size-controlled (roughly uniform word counts)</li>
          <li>Boundary-robust (overlap preserves continuity)</li>
        </ul>
        <p>These chunks are then embedded and indexed for similarity-based retrieval in the RAG pipeline.</p>
      </section>

      <section>
        <h2>Example: Packing & Overlap (conceptual)</h2>
        <div class="example">
          <p>Given a sequence of paragraph-blocks, pack them into ~200-word chunks, and include the last ~40 words as overlap into the next chunk. If one paragraph is 500 words, split it into 3 chunks with overlaps between them.</p>
        </div>
      </section>

      <footer>
        <p>Generated: 2026-01-04</p>
        <p>Author: Adrian Darmali</p>
      </footer>
    </main>
  </div>
</body>
</html>
