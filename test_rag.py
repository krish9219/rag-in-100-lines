"""Offline tests for rag.py — no OpenAI calls.

These verify the pure-Python pieces: chunking, similarity math, and the retrieval
ranking. The embed() and generate() functions hit OpenAI and are excluded.
"""

from __future__ import annotations

import numpy as np

import rag


def test_chunking_respects_size_and_overlap() -> None:
    text = " ".join(str(i) for i in range(1000))
    chunks = rag.chunk_text(text, source="t.md")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= rag.CHUNK_SIZE
        assert c.source == "t.md"
    first_words = chunks[0].text.split()[-rag.CHUNK_OVERLAP:]
    second_start = chunks[1].text.split()[: rag.CHUNK_OVERLAP]
    assert first_words == second_start, "chunks should overlap by CHUNK_OVERLAP words"


def test_retrieve_ranks_by_cosine() -> None:
    chunks = [
        rag.Chunk(text="cats are mammals", source="a"),
        rag.Chunk(text="python is a programming language", source="b"),
        rag.Chunk(text="dogs are mammals too", source="c"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    def fake_embed(texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    original = rag.embed
    rag.embed = fake_embed  # type: ignore[assignment]
    try:
        hits = rag.retrieve("anything", chunks, vectors, k=2)
        assert [h.source for h in hits] == ["a", "c"]
    finally:
        rag.embed = original  # type: ignore[assignment]


if __name__ == "__main__":
    test_chunking_respects_size_and_overlap()
    test_retrieve_ranks_by_cosine()
    print("ok")
