import math
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

import chat


# -------------------------
# Metric helpers (offline, deterministic)
# -------------------------

def hit_at_k(ranked_ids, relevant_ids, k: int) -> float:
    top = ranked_ids[:k]
    return 1.0 if any(x in relevant_ids for x in top) else 0.0


def precision_at_k(ranked_ids, relevant_ids, k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    rel = sum(1 for x in top if x in relevant_ids)
    return rel / float(k)


def recall_at_k(ranked_ids, relevant_ids, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = ranked_ids[:k]
    rel = sum(1 for x in top if x in relevant_ids)
    return rel / float(len(relevant_ids))


def mrr(ranked_ids, relevant_ids) -> float:
    for i, x in enumerate(ranked_ids, start=1):
        if x in relevant_ids:
            return 1.0 / float(i)
    return 0.0


def average_precision(ranked_ids, relevant_ids, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = ranked_ids[:k]
    num_rel = 0
    acc = 0.0
    for i, x in enumerate(top, start=1):
        if x in relevant_ids:
            num_rel += 1
            acc += num_rel / float(i)
    return acc / float(len(relevant_ids))


def dcg_at_k(ranked_ids, rel_map, k: int) -> float:
    dcg = 0.0
    for i, x in enumerate(ranked_ids[:k], start=1):
        rel = float(rel_map.get(x, 0.0))
        dcg += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return dcg


def ndcg_at_k(ranked_ids, rel_map, k: int) -> float:
    dcg = dcg_at_k(ranked_ids, rel_map, k)
    ideal = sorted([float(v) for v in rel_map.values()], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k], start=1):
        idcg += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return 0.0 if idcg == 0.0 else dcg / idcg


_STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "a", "an", "for", "with", "on", "at", "by",
    "is", "are", "was", "were", "be", "as", "that", "this", "it", "from", "into",
}


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def token_f1(pred: str, gold: str) -> float:
    p = _tokens(pred)
    g = _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_set = set(p)
    g_set = set(g)
    tp = len(p_set & g_set)
    prec = tp / float(len(p_set)) if p_set else 0.0
    rec = tp / float(len(g_set)) if g_set else 0.0
    return 0.0 if (prec + rec) == 0.0 else (2.0 * prec * rec) / (prec + rec)


def sentence_faithfulness(answer: str, context: str, min_token_overlap: float = 0.6) -> float:
    '''
    Offline proxy for Faithfulness / Groundedness:
    - Split answer into sentences (claims).
    - A claim is 'supported' if >= min_token_overlap of its content tokens appear in context.
    '''
    ctx = (context or "").lower()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (answer or "").strip()) if s.strip()]
    # Do not treat citation lines as claims.
    sents = [s for s in sents if not s.lower().startswith("sources:") and "http" not in s.lower()]
    if not sents:
        return 0.0
    supported = 0
    for s in sents:
        toks = [t for t in _tokens(s) if t not in _STOPWORDS and len(t) > 3]
        if not toks:
            supported += 1
            continue
        overlap = sum(1 for t in toks if t in ctx)
        if overlap / float(len(toks)) >= min_token_overlap:
            supported += 1
    return supported / float(len(sents))


# -------------------------
# Deterministic fixtures
# -------------------------

class DummyEmbedModel:
    '''Deterministic, keyword-based embeddings so retrieval is stable for unit tests.'''
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        out = []
        for t in texts:
            t = (t or "").lower()
            if "xposed" in t:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "hook" in t or "instrument" in t:
                out.append([0.8, 0.2, 0.0, 0.0])
            elif "jailbreak" in t:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0, 0.0])
        arr = np.asarray(out, dtype=np.float32)
        if normalize_embeddings:
            n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / n
        return arr


def make_index_artifacts(tmp_path: Path):
    import faiss
    import json

    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],   # chunk0: xposed (most relevant)
            [0.8, 0.2, 0.0, 0.0],   # chunk1: hooking/instrumentation (partially relevant)
            [0.0, 1.0, 0.0, 0.0],   # chunk2: jailbreak
        ],
        dtype=np.float32,
    )
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    index = faiss.IndexFlatIP(4)
    index.add(vecs)

    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    rows = [
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
            "chunk_uid": "doc1:0001",
            "doc_id": "doc1",
            "chunk_id": 1,
            "url": "https://www.appdome.com/how-to/test-hooking",
            "title": "Prevent Hooking / Instrumentation",
            "section": "Overview",
            "last_updated": "2026-01-01",
            "breadcrumbs": ["How to", "Android"],
            "text": "Detect runtime hooking and instrumentation attempts.",
        },
        {
            "i": 2,
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
        for r in rows:
            f.write(json.dumps(r) + "\n")

    (index_dir / "manifest.json").write_text(json.dumps({"model": "dummy"}, indent=2), encoding="utf-8")
    return index_dir, rows


# -------------------------
# Retrieval metric tests (IR style + RAG context proxies)
# -------------------------

@pytest.mark.parametrize(
    "query,relevant_ids,graded_rel,k,expected",
    [
        (
            "How do I detect Xposed?",
            {"doc1:0000", "doc1:0001"},
            {"doc1:0000": 2, "doc1:0001": 1, "doc2:0000": 0},
            3,
            {"hit@1": 1.0, "recall@3": 1.0, "precision@2_min": 0.5, "ndcg@3_min": 0.9},
        ),
        (
            "How do I detect jailbreak?",
            {"doc2:0000"},
            {"doc2:0000": 2, "doc1:0000": 0, "doc1:0001": 0},
            3,
            {"hit@1": 1.0, "recall@3": 1.0, "precision@1_min": 1.0, "ndcg@3_min": 0.9},
        ),
    ],
)
def test_retrieval_ranking_metrics(monkeypatch, tmp_path, query, relevant_ids, graded_rel, k, expected):
    index_dir, _rows = make_index_artifacts(tmp_path)

    monkeypatch.setattr(chat, "load_embedding_model", lambda *a, **k: DummyEmbedModel())

    rag_kwargs = dict(
        index_dir=index_dir,
        embed_model_name_or_path="dummy",
        device=None,
        force_offline=True,
        llm_model="llama3",
        ollama_url="http://localhost:11434",
        top_k=k,
        min_score=0.0,
        max_source_chars=200,
        max_total_context_chars=2000,
        history_turns=0,
        temperature=0.0,
        seed=42,
        debug=False,
    )
    if "trust_env" in chat.ThreatExpertRAG.__init__.__code__.co_varnames:
        rag_kwargs["trust_env"] = False

    rag = chat.ThreatExpertRAG(**rag_kwargs)

    qvec = chat.embed_query(rag.embed_model, query)
    retrieved = chat.retrieve(rag.index, rag.chunks, qvec, top_k=k, min_score=0.0)
    ranked = [r.row.chunk_uid for r in retrieved]

    assert hit_at_k(ranked, relevant_ids, 1) == expected["hit@1"]
    assert recall_at_k(ranked, relevant_ids, k) == expected["recall@3"]
    assert mrr(ranked, relevant_ids) >= 1.0
    assert ndcg_at_k(ranked, graded_rel, k) >= expected["ndcg@3_min"]

    if "precision@1_min" in expected:
        assert precision_at_k(ranked, relevant_ids, 1) >= expected["precision@1_min"]
    if "precision@2_min" in expected:
        assert precision_at_k(ranked, relevant_ids, 2) >= expected["precision@2_min"]

    ctx_precision = precision_at_k(ranked, relevant_ids, k)
    ctx_recall = recall_at_k(ranked, relevant_ids, k)
    assert ctx_recall >= 0.8
    assert ctx_precision >= 0.3


# -------------------------
# Grounding / faithfulness + answer correctness tests (offline, mocked generation)
# -------------------------

def test_grounding_and_answer_correctness_metrics(monkeypatch, tmp_path):
    index_dir, _rows = make_index_artifacts(tmp_path)

    monkeypatch.setattr(chat, "load_embedding_model", lambda *a, **k: DummyEmbedModel())

    def fake_generate_answer(*args, **kwargs):
        return (
            "To address Xposed, enable protection to detect and protect against the Xposed framework at runtime. "
            "This helps detect runtime hooking and instrumentation attempts while the app is running.\n\n"
            "Sources: https://www.appdome.com/how-to/test-xposed"
        )

    monkeypatch.setattr(chat, "generate_answer", fake_generate_answer)

    rag_kwargs = dict(
        index_dir=index_dir,
        embed_model_name_or_path="dummy",
        device=None,
        force_offline=True,
        llm_model="llama3",
        ollama_url="http://localhost:11434",
        top_k=3,
        min_score=0.0,
        max_source_chars=200,
        max_total_context_chars=2000,
        history_turns=0,
        temperature=0.0,
        seed=42,
        debug=False,
    )
    if "trust_env" in chat.ThreatExpertRAG.__init__.__code__.co_varnames:
        rag_kwargs["trust_env"] = False

    rag = chat.ThreatExpertRAG(**rag_kwargs)

    question = "How do I detect Xposed?"
    answer = rag.answer(question)

    qvec = chat.embed_query(rag.embed_model, question)
    retrieved = chat.retrieve(rag.index, rag.chunks, qvec, top_k=3, min_score=0.0)
    context = "\n\n".join([r.row.text for r in retrieved])

    faith = sentence_faithfulness(answer, context, min_token_overlap=0.5)
    assert faith >= 0.6

    assert "xposed" in answer.lower()

    gold = "Detect and protect against the Xposed framework."
    assert token_f1(answer, gold) >= 0.4

    assert re.search(r"https?://", answer)


def test_noise_sensitivity_detects_hallucination(monkeypatch, tmp_path):
    index_dir, _rows = make_index_artifacts(tmp_path)

    monkeypatch.setattr(chat, "load_embedding_model", lambda *a, **k: DummyEmbedModel())

    def hallucinated_generate(*args, **kwargs):
        return (
            "Appdome provides anti-vishing signals and blocks screen sharing malware for ATO prevention. "
            "It also detects Xposed.\n\n"
            "Sources: https://www.appdome.com/how-to/test-xposed"
        )

    monkeypatch.setattr(chat, "generate_answer", hallucinated_generate)

    rag_kwargs = dict(
        index_dir=index_dir,
        embed_model_name_or_path="dummy",
        device=None,
        force_offline=True,
        llm_model="llama3",
        ollama_url="http://localhost:11434",
        top_k=1,
        min_score=0.0,
        max_source_chars=200,
        max_total_context_chars=2000,
        history_turns=0,
        temperature=0.0,
        seed=42,
        debug=False,
    )
    if "trust_env" in chat.ThreatExpertRAG.__init__.__code__.co_varnames:
        rag_kwargs["trust_env"] = False

    rag = chat.ThreatExpertRAG(**rag_kwargs)

    question = "How do I detect Xposed?"
    answer = rag.answer(question)

    qvec = chat.embed_query(rag.embed_model, question)
    retrieved = chat.retrieve(rag.index, rag.chunks, qvec, top_k=1, min_score=0.0)
    context = "\n\n".join([r.row.text for r in retrieved])

    faith = sentence_faithfulness(answer, context, min_token_overlap=0.6)
    assert faith <= 0.7
