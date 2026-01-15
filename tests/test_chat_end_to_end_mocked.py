import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

import chat


class DummyEmbedModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        out = []
        for t in texts:
            t = (t or "").lower()
            if "xposed" in t:
                out.append([1, 0, 0, 0])
            elif "jailbreak" in t:
                out.append([0, 1, 0, 0])
            else:
                out.append([0, 0, 1, 0])
        arr = np.asarray(out, dtype=np.float32)
        if normalize_embeddings:
            n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / n
        return arr


def _make_index_artifacts(tmp_path: Path):
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
    return index_dir


def _rag_kwargs(index_dir):
    kw = dict(
        index_dir=index_dir,
        embed_model_name_or_path="dummy",
        device=None,
        force_offline=True,
        llm_model="llama3",
        ollama_url="http://localhost:11434",
        top_k=2,
        min_score=0.0,
        max_source_chars=200,
        max_total_context_chars=2000,
        history_turns=0,
        temperature=0.0,
        seed=42,
        debug=False,
    )
    if "trust_env" in chat.ThreatExpertRAG.__init__.__code__.co_varnames:
        kw["trust_env"] = False
    return kw


def test_rag_answer_appends_sources_when_llm_omits(monkeypatch, tmp_path):
    index_dir = _make_index_artifacts(tmp_path)

    monkeypatch.setattr(chat, "load_embedding_model", lambda *a, **k: DummyEmbedModel())

    captured = {}

    def fake_generate_answer(*args, **kwargs):
        # Session-aware patched chat.py uses keyword args; store prompt for validation.
        msgs = kwargs.get("messages") or (args[3] if len(args) > 3 else None)
        if msgs:
            captured["prompt"] = msgs[-1]["content"]
        return "Use Appdome protections to detect Xposed at runtime."  # no URL on purpose

    monkeypatch.setattr(chat, "generate_answer", fake_generate_answer)

    rag = chat.ThreatExpertRAG(**_rag_kwargs(index_dir))
    out = rag.answer("How do I detect Xposed?")

    assert "Sources:" in out
    assert "https://www.appdome.com/how-to/test-xposed" in out
    assert "https://www.appdome.com/how-to/test-xposed" in captured.get("prompt", "")


def test_rag_ollama_model_not_found_message(monkeypatch, tmp_path):
    index_dir = _make_index_artifacts(tmp_path)
    monkeypatch.setattr(chat, "load_embedding_model", lambda *a, **k: DummyEmbedModel())

    # If chat.py exposes OllamaAPIError, raise it; otherwise raise RuntimeError.
    err_cls = getattr(chat, "OllamaAPIError", RuntimeError)

    def fake_generate_answer(*args, **kwargs):
        raise err_cls("model 'llama3' not found, try pulling it first")

    monkeypatch.setattr(chat, "generate_answer", fake_generate_answer)

    rag = chat.ThreatExpertRAG(**_rag_kwargs(index_dir))
    out = rag.answer("How do I detect Xposed?")

    assert "pull llama3" in out.lower()
