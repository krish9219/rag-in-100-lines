# Security policy

## Supported versions

Only the `main` branch is supported. There are no version releases yet.

## Reporting a vulnerability

Please **do not** open a public issue for security concerns. Instead, email the maintainer directly via the email on the GitHub profile, or use GitHub's private vulnerability reporting feature on this repo.

You can expect:

- Acknowledgement within 7 days.
- A first assessment within 14 days.
- A fix (or a "won't fix" explanation) within 30 days for confirmed vulnerabilities.

## Threat model

This repo is a learning artifact, not a production system. The threat surface is small:

- **Prompt injection through retrieved documents.** If you point this at untrusted markdown, the LLM may follow instructions hidden in the docs. Treat all retrieved text as untrusted input before letting the model do anything irreversible (tool use, code execution, etc.).
- **Pickle deserialization.** The cached index is loaded with `pickle.loads()`. Don't share `.rag_index.pkl` files across machines you don't trust.
- **API key exposure.** `OPENAI_API_KEY` is read from the environment. The `.gitignore` excludes `.env`.

If you discover any other category of vulnerability, please report it.
