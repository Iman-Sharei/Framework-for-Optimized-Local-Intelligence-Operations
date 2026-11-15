# folio

Standalone tool: local hybrid RAG for AI engineering / MLOps / AI infra / LangChain docs-nav.
Shared index: `~/.folio/` (override with `FOLIO_HOME`).

Before AI-eng / MLOps / RAG / agent / AI-infra work, run:
`folio query "<question>" --json`

For LangChain / LangGraph / LangSmith coding:
`folio query "<task> docs" --source langchain-docs-nav -k 12 --json`
then open the returned `docs.langchain.com` URLs before writing code.

Do **not** fetch curriculum GitHub repos into chat — query the local index only.
Exception: official docs.langchain.com URLs from langchain-docs-nav may be fetched.

Skill: `.claude/skills/folio/SKILL.md` / `.agents/skills/folio/SKILL.md`.
If index missing: `folio bootstrap` or `folio build`. Refresh: `folio update`.
