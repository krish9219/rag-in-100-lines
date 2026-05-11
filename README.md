# rag-in-100-lines

> A complete Retrieval-Augmented Generation engine in one Python file. No vector database. No framework. No magic.

Most RAG tutorials reach for LangChain, LlamaIndex, Pinecone, and three layers of abstraction before retrieving a single chunk. The actual algorithm is small. This repo is the algorithm, end-to-end, in 100 readable lines.

```
   docs ──► chunk ──► embed ──► numpy index ──► cosine retrieve ──► LLM ──► answer + citations
```

## Quick start

```bash
git clone https://github.com/krish9219/rag-in-100-lines
cd rag-in-100-lines
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python rag.py "What is RAG and why does it matter?"
```

The first run embeds the sample docs in `docs/` and caches an index to `.rag_index.pkl`. Subsequent runs are instant.

## Example

```bash
$ python rag.py "How should I chunk documents in production?"
```

```json
{
  "answer": "In production, naive whitespace splitting destroys structure. Better options are semantic chunking (split on paragraph boundaries, then merge until a token budget is reached), markdown-aware chunking (never split inside a code block or table row), and a sliding window with 50–100 token overlap [1]. ...",
  "sources": [
    {"source": "docs/production.md", "preview": "Naive whitespace splitting destroys structure. Better options..."},
    {"source": "docs/rag-explained.md", "preview": "Chunk — split source documents into bite-size passages..."}
  ]
}
```

## Interactive chat

```bash
python chat.py
```

```
RAG chat — type a question (Ctrl-C to exit)

> what's the difference between dense and sparse retrieval?
Dense retrieval (embeddings) and sparse retrieval (BM25) make different mistakes...
```

## What's actually in here

| File | Lines | Purpose |
|---|---|---|
| `rag.py` | ~100 | The whole engine: chunk, embed, index, retrieve, generate |
| `chat.py` | ~25 | Interactive REPL |
| `test_rag.py` | ~45 | Offline tests — no API calls |
| `docs/*.md` | — | Sample knowledge base for the demo |

## How it works (read the source, but here's the summary)

1. **Chunk** — split each doc on whitespace, sliding window of 500 words with 50-word overlap.
2. **Embed** — call OpenAI `text-embedding-3-small` once per chunk. Normalize vectors so cosine = dot product.
3. **Index** — stack vectors into a `(N, 1536)` numpy array. Pickle alongside the chunks.
4. **Retrieve** — embed the query, dot-product against the matrix, take top-k.
5. **Generate** — concatenate hits as numbered context, instruct `gpt-4o-mini` to cite by number.

That's it. Every fancy RAG framework is variations on these five steps.

## When to use this vs a real framework

Use this when:
- You want to understand how RAG works
- You have <100k chunks
- You're prototyping and don't want to fight LangChain
- You're teaching RAG to someone

Use a real framework when:
- You need ANN search, sharding, hybrid retrieval, or reranking out of the box
- You're integrating 10+ data sources
- You need observability hooks for production

## Reading order

If you want to learn RAG from this repo:

1. Read [`docs/rag-explained.md`](docs/rag-explained.md) — the concept.
2. Read [`rag.py`](rag.py) top to bottom. Each function is one stage.
3. Run the test suite: `python test_rag.py`
4. Try changing `CHUNK_SIZE`, `TOP_K`, or the system prompt. Notice what breaks.

## Tests

```bash
python test_rag.py
```

The tests verify chunking and retrieval ranking without spending API credits.

## License

MIT — see [LICENSE](LICENSE).
