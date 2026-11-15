"""Helpers for LangChain docs-nav retrieval → official page URLs to open."""

from __future__ import annotations

import re
from typing import Any, Iterable

DOC_URL_RE = re.compile(r"https://docs\.langchain\.com/[^\s\)\]\"'<>]+")


def extract_doc_urls(text: str) -> list[str]:
    """Extract unique docs.langchain.com URLs from retrieved chunk text."""
    seen: set[str] = set()
    out: list[str] = []
    for match in DOC_URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def recommend_doc_urls_from_hits(
    hits: Iterable[Any],
    *,
    limit: int = 12,
) -> list[str]:
    """Ordered unique doc URLs from retrieve hits (best-first)."""
    seen: set[str] = set()
    out: list[str] = []
    for hit in hits:
        text = getattr(hit, "text", None)
        if text is None and isinstance(hit, dict):
            text = hit.get("text") or ""
        for url in extract_doc_urls(str(text or "")):
            if url in seen:
                continue
            seen.add(url)
            out.append(url)
            if len(out) >= limit:
                return out
    return out


def langchain_workflow_required(rule_or_skill_text: str) -> bool:
    """True if project wiring encodes the nav → open-docs workflow."""
    t = rule_or_skill_text.lower()
    return (
        ("langchain-docs-nav" in t or "docs-suggest" in t)
        and "docs.langchain.com" in t
        and ("webfetch" in t or "open" in t)
    )


def privacy_no_source_names(rule_or_skill_text: str) -> bool:
    """True if wiring tells assistants not to name folio sources to the user."""
    t = rule_or_skill_text.lower()
    return (
        "never" in t
        and ("source" in t)
        and ("source_id" in t or "folio source" in t or "corpus" in t)
    )
