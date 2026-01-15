import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

import chat


def _make_small_index(tmp_path: Path):
    dim = 4
    emb = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    import faiss

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    chunks = [
        {
            "i": 0,
            "chunk_uid": "doc1:0000",
            "doc_id": "doc1",
            "chunk_id": 0,
            "url": "https://www.appdome.com/how-to/test-xposed",
            "title": "Protect from Xposed",
            "section": "Overview",
            "last_updated": "2026-01-01",
            "breadcrumbs": ["How to", "Android"],
            "text": "Detect and protect against the Xposed framework.",
        },
        {
            "i": 1,
            "chunk_uid": "doc2:0000",
            "doc_id": "doc2",
            "chunk_id": 0,
            "url": "https://www.appdome.com/how-to/test-jailbreak",
            "title": "Detect Jailbreak",
            "section": "Overview",
            "last_updated": "2026-01-02",
            "breadcrumbs": ["How to", "iOS"],
            "text": "Detect jailbreak and apply runtime protections.",
        },
    ]

    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row) + "\n")

    (index_dir / "manifest.json").write_text(json.dumps({"model": "dummy"}, indent=2), encoding="utf-8")
    return index_dir, chunks


def test_retrieve_orders_by_score(tmp_path):
    index_dir, chunks = _make_small_index(tmp_path)

    index = chat.load_faiss_index(index_dir / "index.faiss")
    rows = chat.load_chunks(index_dir / "chunks.jsonl")

    q = np.array([[1, 0, 0, 0]], dtype=np.float32)
    retrieved = chat.retrieve(index, rows, q, top_k=2, min_score=0.0)
    assert retrieved
    assert retrieved[0].row.url == chunks[0]["url"]


def test_build_messages_includes_urls_and_returns_url_list(tmp_path):
    index_dir, _ = _make_small_index(tmp_path)
    rows = chat.load_chunks(index_dir / "chunks.jsonl")

    retrieved = [
        chat.RetrievedChunk(score=0.9, row=rows[0]),
        chat.RetrievedChunk(score=0.7, row=rows[1]),
    ]

    messages, urls = chat.build_messages(
        question="How do I detect Xposed?",
        retrieved=retrieved,
        history=[],
        history_turns=0,
        max_source_chars=200,
        max_total_chars=2000,
    )

    assert urls and urls[0].startswith("https://")
    user_msg = messages[-1]["content"]
    assert "Appdome KB Sources" in user_msg
    assert rows[0].url in user_msg
    assert rows[1].url in user_msg


def test_enforce_sources_appends_when_missing():
    answer = "Here is the answer without citations."
    out = chat.enforce_sources(answer, ["https://www.appdome.com/how-to/test-xposed"])
    assert "Sources:" in out
    assert "https://www.appdome.com/how-to/test-xposed" in out
