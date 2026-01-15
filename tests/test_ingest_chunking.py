import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

import ingest


def test_heading_aware_sections_basic():
    blocks = [
        {"type": "heading", "level": 2, "text": "Protect"},
        {"type": "paragraph", "text": "Intro text."},
        {"type": "heading", "level": 3, "text": "Steps"},
        {"type": "list", "items": ["Step A", "Step B"]},
        {"type": "code", "text": "print('hi')"},
    ]
    sections = ingest.heading_aware_sections(blocks)
    assert len(sections) >= 2

    # First section should include intro
    path0, txt0 = sections[0]
    assert "Protect" in path0
    assert "Intro text." in txt0

    # Second section should include list + code
    path1, txt1 = sections[1]
    assert "Protect" in path1 and "Steps" in path1
    assert "- Step A" in txt1
    assert "print('hi')" in txt1


def test_chunk_text_respects_size_and_produces_multiple_chunks():
    text = "\n\n".join(["A" * 180, "B" * 180, "C" * 180])
    chunks = ingest.chunk_text(text, max_chars=220, overlap=50)
    assert len(chunks) >= 2
    # Hard cap is max_chars*1.25 by design
    assert all(len(c) <= int(220 * 1.25) for c in chunks)
