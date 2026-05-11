# Contributing

Thank you for your interest. This repo has a strict constraint: **`rag.py` must stay under 100 lines.** Everything else (docs, examples, tests) is open for expansion.

## Quick start

```bash
git clone https://github.com/krish9219/rag-in-100-lines
cd rag-in-100-lines
pip install -r requirements.txt
python test_rag.py
```

## What kinds of PRs are likely to be accepted

- Fixes to bugs in `rag.py` that don't grow the line count.
- New offline tests in `test_rag.py`.
- Improvements to `docs/*.md` — clearer explanations, missed pitfalls, better diagrams.
- New examples that live in their own file (e.g., `examples/local_models.py`).
- CI / repo-hygiene improvements.

## What kinds of PRs are unlikely to be accepted

- Anything that grows `rag.py` past 100 lines.
- Adding LangChain / LlamaIndex / Haystack as a dependency.
- Adding a vector database as a dependency.
- Generic "make it more flexible" refactors.

If your idea conflicts with the line-count constraint, open an issue first to discuss whether an `examples/` file is the right home for it.

## PR checklist

- [ ] Tests still pass (`python test_rag.py`).
- [ ] `rag.py` is still ≤ 100 lines (`wc -l rag.py`).
- [ ] README is updated if behavior changed.
- [ ] No new dependencies in `requirements.txt` unless absolutely needed.

## Style

- Black-formatted Python.
- Type hints on every public function.
- Docstrings only when the function name doesn't already explain it.
