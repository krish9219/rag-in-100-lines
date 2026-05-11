# Taking RAG to production

A 100-line RAG works on a laptop. A production RAG has to survive concurrent users, growing corpora, and adversarial queries. Here are the changes that matter, roughly in order of impact.

## Vector storage

Numpy arrays in a pickle file scale to maybe 100k chunks before search latency hurts. Beyond that, use a vector database with an ANN index: Qdrant, Weaviate, Pinecone, or pgvector with HNSW. The accuracy/latency tradeoff is tunable per index.

## Chunking strategy

Naive whitespace splitting destroys structure. Better options:

- **Semantic chunking**: split on paragraph boundaries, then merge until a token budget is reached.
- **Markdown-aware chunking**: never split inside a code block or a table row.
- **Sliding window with overlap**: 50–100 token overlap rescues answers that straddle chunk boundaries.

## Query rewriting

User queries are often underspecified ("how do I do this?") or conversational ("and what about that one?"). A cheap LLM call to rewrite the query before retrieval — incorporating prior turns and expanding shorthand — typically lifts recall by 20%+ at small cost.

## Evaluation

Without an eval set, every change is a guess. Build a small set of 50–200 question/answer pairs with known correct sources. Track three numbers on every change: retrieval hit rate, answer faithfulness (did the model stick to the context?), and answer correctness. Tools: Ragas, Promptfoo, custom scripts.

## Caching

Embedding the same chunk twice is wasted money. Cache by content hash. For generation, cache by (query, retrieved-chunks) tuple — repeat questions are extremely common in real workloads.

## Security

Treat retrieved text as untrusted input. Prompt injection through retrieved documents is a real attack — an attacker who controls a document in your corpus can hijack the model. Mitigations: sanitize markdown, strip executable instructions from chunks, and never feed retrieved text into tool-calling without a guard.
