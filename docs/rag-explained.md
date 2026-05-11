# What is RAG?

Retrieval-Augmented Generation (RAG) is a technique for grounding large language model responses in external knowledge. Instead of relying solely on a model's parametric memory, the system retrieves the most relevant snippets from a knowledge base at query time and feeds them to the model as context.

## Why RAG matters

Large language models hallucinate when asked about facts outside their training data, and even within their training data they can confidently produce false information. RAG mitigates this in three concrete ways:

- **Freshness**: knowledge bases update faster than models are retrained.
- **Citations**: retrieved chunks let the model point to a source, which makes answers auditable.
- **Cost**: small models with good retrieval often beat large models with no retrieval.

## The four stages

1. **Chunk** — split source documents into bite-size passages. Too large and you waste context; too small and you lose meaning. 200–800 tokens is a common sweet spot.
2. **Embed** — convert each chunk into a vector using an embedding model. Vectors with similar meanings end up close in vector space.
3. **Retrieve** — embed the user's query the same way and find the nearest chunks by cosine similarity (or dot product on unit vectors).
4. **Generate** — pass the retrieved chunks to a chat model along with the original question. Instruct the model to stay within the provided context.

## Common pitfalls

- **Bad chunking destroys recall.** Splitting in the middle of a sentence loses the answer.
- **Embeddings drift between models.** If you re-embed the query with a different model than the corpus, similarity is meaningless.
- **Top-k is not free.** Each chunk eats context window. K=4 with 500-token chunks already spends 2000 tokens before the model says a word.
- **No retrieval beats bad retrieval.** If the retrieved chunks are irrelevant, the model still tries to answer using them and produces worse output than asking the model directly.
