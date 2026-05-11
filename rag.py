"""
RAG in 100 lines — a minimal, dependency-light Retrieval-Augmented Generation engine.

Pipeline:
    docs -> chunk -> embed -> store (numpy) -> retrieve (cosine) -> generate

Run:
    export OPENAI_API_KEY=sk-...
    python rag.py "What does this repo do?"
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

_CLIENT: OpenAI | None = None


def client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
INDEX_PATH = Path(".rag_index.pkl")


@dataclass
class Chunk:
    text: str
    source: str


def chunk_text(text: str, source: str) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        body = " ".join(words[i : i + CHUNK_SIZE]).strip()
        if body:
            chunks.append(Chunk(text=body, source=source))
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def build_index(docs_glob: str = "docs/**/*.md") -> tuple[list[Chunk], np.ndarray]:
    chunks: list[Chunk] = []
    for path in glob.glob(docs_glob, recursive=True):
        chunks.extend(chunk_text(Path(path).read_text(encoding="utf-8"), source=path))
    if not chunks:
        raise SystemExit(f"No docs matched: {docs_glob}")
    print(f"[index] embedding {len(chunks)} chunks…")
    vectors = embed([c.text for c in chunks])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    INDEX_PATH.write_bytes(pickle.dumps({"chunks": chunks, "vectors": vectors}))
    print(f"[index] saved {INDEX_PATH}")
    return chunks, vectors


def load_index() -> tuple[list[Chunk], np.ndarray]:
    if not INDEX_PATH.exists():
        return build_index()
    data = pickle.loads(INDEX_PATH.read_bytes())
    return data["chunks"], data["vectors"]


def retrieve(query: str, chunks: list[Chunk], vectors: np.ndarray, k: int = TOP_K) -> list[Chunk]:
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-12
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [chunks[i] for i in top]


def generate(query: str, context_chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{i+1}] ({c.source})\n{c.text}" for i, c in enumerate(context_chunks))
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using ONLY the provided context. "
                "Cite sources as [1], [2], etc. If the context is insufficient, say so plainly."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    resp = client().chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content or ""


def ask(query: str) -> dict:
    chunks, vectors = load_index()
    hits = retrieve(query, chunks, vectors)
    answer = generate(query, hits)
    return {"answer": answer, "sources": [{"source": c.source, "preview": c.text[:120]} for c in hits]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question"')
        sys.exit(1)
    print(json.dumps(ask(" ".join(sys.argv[1:])), indent=2))
