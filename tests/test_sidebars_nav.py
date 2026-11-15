from __future__ import annotations

import json
from pathlib import Path

from folio.add_doc import _sidebars_json_to_markdown
from folio.docs_nav import extract_doc_urls, recommend_doc_urls_from_hits


def _mini_sidebars(tmp_path: Path) -> Path:
    links = [
        {
            "type": "link",
            "title": title,
            "url": f"https://docs.langchain.com/oss/python/langgraph/{slug}",
        }
        for title, slug in [
            ("Persistence", "persistence"),
            ("Streaming", "streaming"),
            ("Checkpoints", "persistence-checkpoints"),
            ("Memory", "memory"),
            ("Tools", "tools"),
            ("Human in the loop", "human-in-the-loop"),
            ("Subgraphs", "subgraph"),
            ("Time travel", "time-travel"),
            ("Workflows", "workflows"),
            ("Agents", "agents"),
            ("MCP", "mcp"),
            ("Deploy", "deploy"),
        ]
    ]
    data = {
        "generated_at": "test",
        "source": "unit-test",
        "page_count": 1,
        "pages": [
            {
                "url": "https://docs.langchain.com/oss/python/langgraph/index",
                "language": "Python",
                "section_trail": [
                    "AGENT DEVELOPMENT LIFECYCLE",
                    "Build",
                    "Python",
                    "LangGraph",
                ],
                "sidebar": [
                    {
                        "type": "group",
                        "title": "Capabilities",
                        "children": links,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "sidebars.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_sidebars_json_to_markdown_includes_official_urls(tmp_path: Path):
    md = _sidebars_json_to_markdown(_mini_sidebars(tmp_path))
    assert "LangChain ecosystem docs navigation map" in md
    assert "https://docs.langchain.com/oss/python/langgraph/persistence" in md
    assert "https://docs.langchain.com/oss/python/langgraph/streaming" in md
    assert "How assistants must use this index" in md
    urls = extract_doc_urls(md)
    assert len(urls) >= 10


def test_recommend_doc_urls_from_hits_dedupes_and_orders():
    class Hit:
        def __init__(self, text: str):
            self.text = text

    hits = [
        Hit("- Persistence → https://docs.langchain.com/oss/python/langgraph/persistence\n"),
        Hit(
            "- Streaming → https://docs.langchain.com/oss/python/langgraph/streaming\n"
            "- Persistence again → https://docs.langchain.com/oss/python/langgraph/persistence\n"
        ),
    ]
    urls = recommend_doc_urls_from_hits(hits, limit=5)
    assert urls == [
        "https://docs.langchain.com/oss/python/langgraph/persistence",
        "https://docs.langchain.com/oss/python/langgraph/streaming",
    ]

# path-view refresh
