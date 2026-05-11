"""Interactive chat against the RAG index. Reads questions on stdin until EOF."""

from __future__ import annotations

import sys

from rag import ask


def main() -> None:
    print("RAG chat — type a question (Ctrl-C to exit)")
    try:
        while True:
            sys.stdout.write("\n> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            q = line.strip()
            if not q:
                continue
            result = ask(q)
            print("\n" + result["answer"])
            print("\nSources:")
            for i, src in enumerate(result["sources"], 1):
                print(f"  [{i}] {src['source']}: {src['preview']}...")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
