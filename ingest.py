#!/usr/bin/env python3
"""
ingest.py
---------
Build a local vector database (FAISS) from Appdome How-To KB JSON files produced by crawler.py.

Design goals:
- Fully local: no hosted/vector DB services.
- Robust parsing: tolerate missing/partial fields.
- Semantics-first chunking: heading-aware + paragraph-aware splitting.
- Simple CLI without argparse.

Outputs (default under data/index/):
- index.faiss            : FAISS index (cosine similarity via normalized embeddings + IndexFlatIP)
- chunks.jsonl           : one JSON object per indexed chunk (metadata + text)
- docs.jsonl             : one JSON object per source document
- manifest.json          : build metadata for reproducibility

Dependencies:
  pip install sentence-transformers faiss-cpu numpy

Offline model note:
- sentence-transformers will try to download models if not present.
- For truly offline use, pre-download the model to disk and pass --model /path/to/model
  or ensure your HF cache already contains it and export TRANSFORMERS_OFFLINE=1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss import failed. Install with: pip install faiss-cpu (or faiss-gpu)"
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers import failed. Install with: pip install sentence-transformers"
    ) from e


# ----------------------------
# Configuration defaults
# ----------------------------

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/index")
DEFAULT_MODEL = os.environ.get("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_DEVICE = os.environ.get("ST_DEVICE", None)  # e.g. "cpu" / "cuda"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP = 200
DEFAULT_BATCH_SIZE = 64


# ----------------------------
# Logging
# ----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------
# Tiny CLI parser (no argparse)
# ----------------------------

def _to_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_kv_args(argv: List[str]) -> Dict[str, Any]:
    """
    Parse args in the form:
      --key value
      --flag
      --flag=true

    Returns a dict: {"key": "value", "flag": True}
    """
    out: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if "=" in key:
            k, v = key.split("=", 1)
            out[k.strip()] = v.strip()
            i += 1
            continue
        # flag with no value
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            out[key] = True
            i += 1
            continue
        out[key] = argv[i + 1]
        i += 2
    return out


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Document:
    doc_id: str
    url: str
    title: str
    last_updated: Optional[str]
    breadcrumbs: List[str]
    source_file: str
    text: str


@dataclass
class Chunk:
    chunk_uid: str
    doc_id: str
    chunk_id: int
    url: str
    title: str
    section: str
    last_updated: Optional[str]
    breadcrumbs: List[str]
    text: str


# ----------------------------
# Helpers
# ----------------------------

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Failed to read JSON: %s", path)
        return None


def list_raw_json_files(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        return []
    files = sorted([p for p in raw_dir.glob("*.json") if p.is_file()])
    # ignore crawler artifacts / meta
    ignore_names = {"index.json", "_crawl_state.json"}
    return [p for p in files if p.name not in ignore_names and not p.name.startswith("_")]


def blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """
    Convert structured blocks into a single canonical text string.
    This is used as a fallback if content.text is missing.
    """
    parts: List[str] = []
    for b in blocks:
        t = (b.get("type") or "").strip().lower()
        if t == "heading":
            h = (b.get("text") or "").strip()
            if h:
                parts.append(h)
        elif t == "paragraph":
            p = (b.get("text") or "").strip()
            if p:
                parts.append(p)
        elif t == "list":
            items = b.get("items") or []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, str):
                        s = it.strip()
                        if s:
                            parts.append(f"- {s}")
                    elif isinstance(it, dict):
                        s = (it.get("text") or "").strip()
                        if s:
                            parts.append(f"- {s}")
        elif t == "code":
            c = (b.get("text") or "").rstrip()
            if c:
                parts.append(c)
        elif t == "table":
            # best-effort table rendering
            rows = b.get("rows") or []
            if isinstance(rows, list) and rows:
                for r in rows:
                    if isinstance(r, list):
                        parts.append(" | ".join(str(x).strip() for x in r))
        elif t == "blockquote":
            q = (b.get("text") or "").strip()
            if q:
                parts.append(q)
        else:
            # unknown block type: try any "text" field
            x = (b.get("text") or "").strip()
            if x:
                parts.append(x)
    return "\n\n".join(parts).strip()


def normalize_ws(s: str) -> str:
    # Collapse crazy whitespace but keep paragraph breaks
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_paragraphs(text: str) -> List[str]:
    # Split on blank lines; keep meaningful blocks
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter (no nltk downloads)
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def sliding_window_join(units: List[str], max_chars: int, overlap: int) -> List[str]:
    """
    Build chunks by joining units (paragraphs or sentences) with a sliding window.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        chunk = "\n\n".join(cur).strip()
        if chunk:
            chunks.append(chunk)
        # create overlap by keeping tail
        if overlap > 0 and chunk:
            tail = chunk[-overlap:]
            cur = [tail]
            cur_len = len(tail)
        else:
            cur = []
            cur_len = 0

    for u in units:
        if not u:
            continue
        add_len = len(u) + (2 if cur else 0)
        if cur_len + add_len > max_chars and cur:
            flush()
        cur.append(u)
        cur_len += add_len

    flush()
    return [c.strip() for c in chunks if c.strip()]


def heading_aware_sections(blocks: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Turn blocks into (section_path, section_text) pairs.
    section_path is derived from heading stack.
    """
    stack: List[Tuple[int, str]] = []
    sections: List[Tuple[str, List[str]]] = []
    cur_text: List[str] = []
    cur_path = ""

    def current_path() -> str:
        if not stack:
            return ""
        return " > ".join([h for _, h in stack if h])

    def flush() -> None:
        nonlocal cur_text, cur_path
        if cur_text:
            sections.append((cur_path, cur_text))
        cur_text = []

    for b in blocks:
        t = (b.get("type") or "").strip().lower()

        if t == "heading":
            txt = (b.get("text") or "").strip()
            if not txt:
                continue
            level = int(b.get("level") or 2)
            # flush current buffer under previous path
            flush()
            # update stack
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, txt))
            cur_path = current_path()
            continue

        # non-heading content -> textualize and append to current section
        if t == "paragraph":
            txt = (b.get("text") or "").strip()
            if txt:
                cur_text.append(txt)
        elif t == "list":
            items = b.get("items") or []
            if isinstance(items, list):
                lines = []
                for it in items:
                    if isinstance(it, str) and it.strip():
                        lines.append(f"- {it.strip()}")
                    elif isinstance(it, dict):
                        x = (it.get("text") or "").strip()
                        if x:
                            lines.append(f"- {x}")
                if lines:
                    cur_text.append("\n".join(lines))
        elif t == "code":
            txt = (b.get("text") or "").rstrip()
            if txt:
                cur_text.append(txt)
        elif t == "table":
            rows = b.get("rows") or []
            if isinstance(rows, list) and rows:
                rendered: List[str] = []
                for r in rows:
                    if isinstance(r, list):
                        rendered.append(" | ".join(str(x).strip() for x in r))
                if rendered:
                    cur_text.append("\n".join(rendered))
        elif t == "blockquote":
            txt = (b.get("text") or "").strip()
            if txt:
                cur_text.append(txt)
        else:
            txt = (b.get("text") or "").strip()
            if txt:
                cur_text.append(txt)

    flush()

    # convert to strings
    out: List[Tuple[str, str]] = []
    for path, parts in sections:
        s = normalize_ws("\n\n".join(parts))
        if s:
            out.append((path, s))
    return out


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Chunk text using paragraph-first splitting; falls back to sentence splitting for large paragraphs.
    """
    text = normalize_ws(text)
    if not text:
        return []
    paras = split_paragraphs(text)
    if not paras:
        return []

    # First pass: paragraph windowing
    chunks = sliding_window_join(paras, max_chars=max_chars, overlap=overlap)

    # Second pass: if any chunk is still huge (rare), split by sentences
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars * 1.25:
            final.append(c)
            continue
        sents = split_sentences(c)
        if sents:
            final.extend(sliding_window_join(sents, max_chars=max_chars, overlap=overlap))
        else:
            final.append(c[:max_chars])
    return [normalize_ws(x) for x in final if normalize_ws(x)]


def load_documents(raw_dir: Path, limit: Optional[int] = None) -> List[Document]:
    docs: List[Document] = []
    files = list_raw_json_files(raw_dir)
    if limit is not None:
        files = files[:limit]

    for p in files:
        j = safe_read_json(p)
        if not j:
            continue

        url = str(j.get("url") or "").strip()
        title = str(j.get("title") or "").strip() or "Untitled"
        last_updated = j.get("last_updated")
        breadcrumbs = j.get("breadcrumbs") or []
        if not isinstance(breadcrumbs, list):
            breadcrumbs = []

        content = j.get("content") or {}
        text = ""
        if isinstance(content, dict):
            text = str(content.get("text") or "").strip()
            if not text:
                blocks = content.get("blocks") or []
                if isinstance(blocks, list):
                    text = blocks_to_text(blocks)

        text = normalize_ws(text)
        if not url or not text:
            continue

        doc_id = sha1_hex(url)[:16]
        docs.append(
            Document(
                doc_id=doc_id,
                url=url,
                title=title,
                last_updated=str(last_updated) if last_updated else None,
                breadcrumbs=[str(x) for x in breadcrumbs if str(x).strip()],
                source_file=str(p.as_posix()),
                text=text,
            )
        )

    # dedupe by url/doc_id
    uniq: Dict[str, Document] = {}
    for d in docs:
        uniq[d.doc_id] = d
    return list(uniq.values())


def build_chunks(doc: Document, blocks: Optional[List[Dict[str, Any]]], max_chars: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []

    # If we have blocks, do heading-aware sections. Otherwise chunk whole doc.
    sections: List[Tuple[str, str]] = []
    if blocks:
        sections = heading_aware_sections(blocks)

    if not sections:
        sections = [("", doc.text)]

    chunk_id = 0
    for section_path, section_text in sections:
        for piece in chunk_text(section_text, max_chars=max_chars, overlap=overlap):
            if len(piece) < 60:
                continue
            uid = f"{doc.doc_id}:{chunk_id:04d}"
            chunks.append(
                Chunk(
                    chunk_uid=uid,
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    url=doc.url,
                    title=doc.title,
                    section=section_path,
                    last_updated=doc.last_updated,
                    breadcrumbs=doc.breadcrumbs,
                    text=piece,
                )
            )
            chunk_id += 1
    return chunks


def load_blocks_for_doc(raw_dir: Path, doc: Document) -> Optional[List[Dict[str, Any]]]:
    # doc.source_file points to the JSON path created by crawler
    p = Path(doc.source_file)
    if not p.exists():
        # maybe ingesting from a moved directory: attempt relative lookup by filename
        candidate = raw_dir / p.name
        if candidate.exists():
            p = candidate
        else:
            return None
    j = safe_read_json(p)
    if not j:
        return None
    content = j.get("content") or {}
    if not isinstance(content, dict):
        return None
    blocks = content.get("blocks")
    if isinstance(blocks, list):
        # ensure dicts
        return [b for b in blocks if isinstance(b, dict)]
    return None


# ----------------------------
# Embedding + FAISS
# ----------------------------

def load_embedding_model(model_name_or_path: str, device: Optional[str], force_offline: bool) -> SentenceTransformer:
    if force_offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        if device:
            return SentenceTransformer(model_name_or_path, device=device)
        return SentenceTransformer(model_name_or_path)
    except Exception as ex:
        msg = (
            f"Failed to load SentenceTransformer model '{model_name_or_path}'.\n"
            f"Offline mode is {'ON' if force_offline else 'OFF'}.\n\n"
            "Fix options:\n"
            "  1) Pre-download the model to your local disk and run:\n"
            "       python ingest.py --model /absolute/path/to/model\n"
            "  2) If you have an HF cache with the model, ensure it is available and keep TRANSFORMERS_OFFLINE=1.\n"
            "  3) To allow downloads (NOT recommended for the assignment), run with:\n"
            "       python ingest.py --no-offline\n"
        )
        raise RuntimeError(msg) from ex


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    # normalize_embeddings=True gives unit vectors -> cosine similarity == dot product
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)  # cosine via normalized embeddings
    index.add(embeddings)
    return index


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    setup_logging()
    args = parse_kv_args(sys.argv[1:])

    raw_dir = Path(str(args.get("raw-dir", DEFAULT_RAW_DIR)))
    out_dir = Path(str(args.get("out-dir", DEFAULT_OUT_DIR)))
    model_name = str(args.get("model", DEFAULT_MODEL))
    device = args.get("device", DEFAULT_DEVICE)
    max_chars = int(args.get("max-chars", DEFAULT_MAX_CHARS))
    overlap = int(args.get("overlap", DEFAULT_OVERLAP))
    batch_size = int(args.get("batch-size", DEFAULT_BATCH_SIZE))
    rebuild = _to_bool(str(args.get("rebuild", "false"))) if "rebuild" in args else False
    limit = int(args["limit"]) if "limit" in args else None
    force_offline = not _to_bool(str(args.get("no-offline", "false")))  # default True

    if not raw_dir.exists():
        logging.error("Raw dir does not exist: %s", raw_dir)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.faiss"
    chunks_path = out_dir / "chunks.jsonl"
    docs_path = out_dir / "docs.jsonl"
    manifest_path = out_dir / "manifest.json"

    if (index_path.exists() or chunks_path.exists() or docs_path.exists()) and not rebuild:
        logging.info("Existing index artifacts found in %s. Use --rebuild true to overwrite.", out_dir)
        logging.info("Nothing to do.")
        return 0

    # Clean outputs if rebuilding
    for p in [index_path, chunks_path, docs_path, manifest_path]:
        if p.exists():
            p.unlink()

    logging.info("Loading documents from: %s", raw_dir)
    docs = load_documents(raw_dir, limit=limit)
    if not docs:
        logging.error("No valid documents found in %s. Ensure crawler produced JSON with content.text.", raw_dir)
        return 3

    logging.info("Loaded %d documents. Building chunks…", len(docs))

    all_chunks: List[Chunk] = []
    for d in docs:
        blocks = load_blocks_for_doc(raw_dir, d)
        all_chunks.extend(build_chunks(d, blocks, max_chars=max_chars, overlap=overlap))

    if not all_chunks:
        logging.error("No chunks produced. Check chunking parameters or input text quality.")
        return 4

    logging.info("Produced %d chunks. Loading embedding model…", len(all_chunks))
    model = load_embedding_model(model_name, device=device if device else None, force_offline=force_offline)

    texts = [c.text for c in all_chunks]
    logging.info("Embedding %d chunks (batch_size=%d)…", len(texts), batch_size)
    t0 = time.time()
    embeddings = embed_texts(model, texts, batch_size=batch_size)
    dt = time.time() - t0
    logging.info("Embeddings computed in %.1fs (dim=%d).", dt, embeddings.shape[1])

    logging.info("Building FAISS index…")
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(index_path))

    # Persist metadata
    logging.info("Writing chunk metadata: %s", chunks_path)
    with chunks_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(all_chunks):
            row = asdict(c)
            row["i"] = i
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Writing doc metadata: %s", docs_path)
    with docs_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    manifest = {
        "created_at": utc_now_iso(),
        "raw_dir": str(raw_dir.as_posix()),
        "out_dir": str(out_dir.as_posix()),
        "model": model_name,
        "device": device or "auto",
        "force_offline": force_offline,
        "max_chars": max_chars,
        "overlap": overlap,
        "batch_size": batch_size,
        "doc_count": len(docs),
        "chunk_count": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_index": str(index_path.name),
        "chunks_file": str(chunks_path.name),
        "docs_file": str(docs_path.name),
        "similarity": "cosine (IndexFlatIP + normalized embeddings)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info("Done. Index saved to: %s", index_path)
    logging.info("Next step: implement retrieve→prompt→generate in chat.py using %s + %s", index_path.name, chunks_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
