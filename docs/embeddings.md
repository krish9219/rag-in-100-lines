# Embedding models

An embedding model is a neural network that maps text into a fixed-dimensional vector of floats. The defining property is that semantically similar inputs map to nearby vectors.

## Choosing a model

OpenAI's `text-embedding-3-small` is the pragmatic default for most projects: 1536 dimensions, low latency, low cost. `text-embedding-3-large` doubles the cost for a few points of retrieval accuracy. Open-source options like `bge-base-en-v1.5` and `nomic-embed-text` are competitive and run locally.

## Similarity math

For unit-normalized vectors, cosine similarity equals the dot product. This is why production retrieval code normalizes once at index time and then uses dot products at query time — it's faster and gives identical rankings.

## Hybrid retrieval

Dense retrieval (embeddings) and sparse retrieval (BM25) make different mistakes. Dense retrieval misses exact keyword matches like product names and error codes. Sparse retrieval misses paraphrases. Combining the two — reciprocal rank fusion is the simplest method — usually outperforms either alone.

## Reranking

After initial retrieval, a cross-encoder reranker can reorder the top-k by jointly attending to the query and each candidate. This adds latency but typically raises hit rate by 10–30 percentage points on hard queries. Cohere Rerank and BGE Reranker are popular choices.
