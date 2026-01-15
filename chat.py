#!/usr/bin/env python3
"""
chat.py
-------
Terminal-based "Appdome Threat Expert" RAG agent.

Pipeline:
  retrieve (FAISS) -> prompt (grounded) -> generate (Ollama llama3) -> answer w/ sources

Key requirements satisfied:
- Fully local: FAISS + SentenceTransformers for embeddings; local Ollama for LLM generation.
- Grounded answers: model is instructed to only use retrieved KB context.
- At least one source URL per answer (enforced).
- No argparse (simple --key value / --flag parsing).
- Solid edge-case handling: missing index, Ollama down, empty retrieval, proxy env pitfalls.

Expected files from ingest.py (default data/index/):
  - index.faiss
  - chunks.jsonl
  - manifest.json (optional but recommended)

Dependencies:
  pip install sentence-transformers faiss-cpu numpy requests

Usage:
  # Interactive chat
  python chat.py

  # Single question mode
  python chat.py --question "How do I integrate Threat Remediation Center in my iOS app?"

Common options:
  python chat.py --index-dir data/index --llm llama3 --top-k 6 --debug true
  python chat.py --ollama http://localhost:11434

Proxy note (common pitfall):
- Python requests respects HTTP_PROXY/HTTPS_PROXY by default.
- In corporate environments, localhost may accidentally go through the proxy.
- This script disables proxy env usage by default for Ollama (trust_env=False).
  To re-enable env proxy behavior, pass: --trust-env true

Offline note:
  For embeddings, this script defaults to offline mode (TRANSFORMERS_OFFLINE=1).
  Ensure your embedding model is already cached OR pass --embed-model /path/to/model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("faiss import failed. Install with: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("sentence-transformers import failed. Install with: pip install sentence-transformers") from e


# ----------------------------
# Defaults
# ----------------------------

DEFAULT_INDEX_DIR = Path("data/index")
DEFAULT_LLM_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_TOP_K = 6
DEFAULT_MIN_SCORE = 0.15
DEFAULT_MAX_SOURCE_CHARS = 900
DEFAULT_MAX_TOTAL_CONTEXT_CHARS = 5500
DEFAULT_HISTORY_TURNS = 4
DEFAULT_TEMPERATURE = 0.2
DEFAULT_SEED = 42

DEFAULT_EMBED_MODEL_FALLBACK = os.environ.get("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_DEVICE = os.environ.get("ST_DEVICE", None)  # cpu/cuda


# ----------------------------
# Logging
# ----------------------------

def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
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
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            out[key] = True
            i += 1
            continue
        out[key] = argv[i + 1]
        i += 2
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ChunkRow:
    i: int
    chunk_uid: str
    doc_id: str
    chunk_id: int
    url: str
    title: str
    section: str
    last_updated: Optional[str]
    breadcrumbs: List[str]
    text: str


@dataclass
class RetrievedChunk:
    score: float
    row: ChunkRow


# ----------------------------
# IO helpers
# ----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception:
                continue
    return items


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def truncate_middle(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    head = max_len // 2
    tail = max_len - head - 15
    return s[:head].rstrip() + "\n… [snip] …\n" + s[-tail:].lstrip()


# ----------------------------
# Embeddings + retrieval
# ----------------------------

def load_embedding_model(model_name_or_path: str, device: Optional[str], force_offline: bool) -> SentenceTransformer:
    if force_offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if device:
        return SentenceTransformer(model_name_or_path, device=device)
    return SentenceTransformer(model_name_or_path)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vec = np.asarray(vec, dtype=np.float32)
    return vec


def load_faiss_index(index_path: Path) -> "faiss.Index":
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    return faiss.read_index(str(index_path))


def load_chunks(chunks_path: Path) -> List[ChunkRow]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl: {chunks_path}")
    raw = read_jsonl(chunks_path)
    rows: List[ChunkRow] = []
    for obj in raw:
        try:
            rows.append(
                ChunkRow(
                    i=int(obj.get("i", len(rows))),
                    chunk_uid=str(obj.get("chunk_uid", "")),
                    doc_id=str(obj.get("doc_id", "")),
                    chunk_id=int(obj.get("chunk_id", 0)),
                    url=str(obj.get("url", "")),
                    title=str(obj.get("title", "")),
                    section=str(obj.get("section", "")),
                    last_updated=obj.get("last_updated"),
                    breadcrumbs=list(obj.get("breadcrumbs") or []),
                    text=str(obj.get("text", "")),
                )
            )
        except Exception:
            continue
    rows.sort(key=lambda r: r.i)
    return rows


def retrieve(
    index: "faiss.Index",
    chunk_rows: List[ChunkRow],
    query_vec: np.ndarray,
    top_k: int,
    min_score: float,
) -> List[RetrievedChunk]:
    if query_vec.ndim != 2 or query_vec.shape[0] != 1:
        raise ValueError(f"Expected query vec shape (1, d), got {query_vec.shape}")
    D, I = index.search(query_vec, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    out: List[RetrievedChunk] = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(chunk_rows):
            continue
        if float(score) < float(min_score):
            continue
        out.append(RetrievedChunk(score=float(score), row=chunk_rows[idx]))

    seen = set()
    deduped: List[RetrievedChunk] = []
    for r in out:
        if r.row.chunk_uid in seen:
            continue
        seen.add(r.row.chunk_uid)
        deduped.append(r)
    return deduped


# ----------------------------
# Ollama client (robust errors + proxy-safe)
# ----------------------------

class OllamaAPIError(RuntimeError):
    """Raised for non-2xx responses from Ollama API endpoints."""


def make_session(trust_env: bool) -> requests.Session:
    s = requests.Session()
    # If trust_env=False, requests ignores HTTP(S)_PROXY env vars.
    s.trust_env = trust_env
    return s


def _extract_error_detail(resp: requests.Response) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict) and "error" in j:
            return str(j["error"])
        return json.dumps(j)[:5000]
    except Exception:
        return (resp.text or "").strip()[:5000]


def _post_json(session: requests.Session, url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    resp = session.post(url, json=payload, timeout=timeout_s)
    if resp.status_code == 404:
        raise FileNotFoundError(f"{url} not found (404)")
    if not resp.ok:
        detail = _extract_error_detail(resp)
        raise OllamaAPIError(f"Ollama API error {resp.status_code} at {url}: {detail}")
    try:
        return resp.json()
    except Exception:
        raise OllamaAPIError(f"Ollama returned non-JSON response at {url}: {resp.text[:500]}")


def ollama_chat(
    session: requests.Session,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    seed: int,
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature, "seed": seed}}
    data = _post_json(session, url, payload, timeout_s=timeout_s)
    msg = data.get("message") or {}
    return str(msg.get("content") or "").strip()


def ollama_generate(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "seed": seed}}
    data = _post_json(session, url, payload, timeout_s=timeout_s)
    return str(data.get("response") or "").strip()


def generate_answer(
    session: requests.Session,
    ollama_url: str,
    llm_model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    seed: int,
) -> str:
    # Preferred: /api/chat. Fallback: /api/generate.
    try:
        return ollama_chat(session, ollama_url, llm_model, messages, temperature=temperature, seed=seed)
    except FileNotFoundError:
        flat = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in messages])
        return ollama_generate(session, ollama_url, llm_model, flat, temperature=temperature, seed=seed)


def classify_ollama_error(e: Exception, ollama_url: str, model: str, trust_env: bool) -> str:
    base = (
        "Ollama is reachable via browser but the API call failed from Python.\n"
        f"- Ollama URL: {ollama_url}\n"
        f"- Model: {model}\n"
        f"- requests.trust_env: {trust_env}\n"
    )

    s = str(e)
    if isinstance(e, OllamaAPIError) and ("model" in s.lower()) and ("not found" in s.lower() or "pull" in s.lower()):
        return (
            base
            + "\nLikely cause: the model isn't installed in Ollama.\n"
            "Remediation (inside WSL / wherever Ollama runs):\n"
            f"  ollama pull {model}\n"
            "\nRaw error: " + s
        )

    if isinstance(e, (requests.exceptions.ProxyError, requests.exceptions.SSLError)) or ("proxy" in s.lower()):
        return (
            base
            + "\nLikely cause: proxy settings are intercepting localhost traffic.\n"
            "Remediation options:\n"
            "  1) Keep proxies disabled (default): do NOT pass --trust-env true\n"
            "  2) Or set NO_PROXY=localhost,127.0.0.1\n"
            "\nRaw error: " + s
        )

    if isinstance(e, requests.exceptions.ConnectionError):
        return (
            base
            + "\nLikely cause: connection refused/reset to the API endpoint.\n"
            "Remediation:\n"
            "  1) Confirm API endpoints:\n"
            "     - GET  http://localhost:11434/api/tags\n"
            "     - POST http://localhost:11434/api/generate\n"
            "  2) If Ollama runs in WSL2, verify port binding and Windows reachability.\n"
            "\nRaw error: " + s
        )

    return (
        base
        + "\nRemediation:\n"
        "  1) Start Ollama: `ollama serve`\n"
        f"  2) Pull model: `ollama pull {model}`\n"
        "  3) Test API: GET http://localhost:11434/api/tags\n"
        "\nRaw error: " + s
    )


# ----------------------------
# Prompting
# ----------------------------

SYSTEM_PROMPT = """You are Appdome Threat-Expert, a security documentation assistant.
You MUST answer using ONLY the provided Appdome How-To KB sources.
If the sources don't contain the answer, say you don't know from the KB and ask a targeted follow-up question.
Constraints:
- Be accurate and grounded.
- Cite at least one source URL in every answer.
- Do not invent product features, steps, signals, or configuration options.
- Prefer crisp, actionable steps and security-relevant details.
- Provide full answers, do not miss important information.
Formatting:
- Use short paragraphs or numbered steps.
- Add a final line: "Sources: <url1> <url2> ..." (at least one URL).
"""


def build_context_snippets(
    retrieved: List[RetrievedChunk],
    max_source_chars: int,
    max_total_chars: int,
) -> Tuple[str, List[str]]:
    parts: List[str] = []
    urls: List[str] = []
    total = 0

    for j, r in enumerate(retrieved, start=1):
        url = r.row.url.strip()
        if not url:
            continue
        title = (r.row.title or "").strip()
        section = (r.row.section or "").strip()
        heading = f"{title} — {section}" if section else title
        excerpt = truncate_middle(normalize_ws(r.row.text), max_source_chars)

        block = (
            f"[{j}] {heading}\n"
            f"URL: {url}\n"
            f"RelevanceScore: {r.score:.3f}\n"
            f"Excerpt:\n{excerpt}\n"
        )

        if total + len(block) > max_total_chars and parts:
            break

        parts.append(block)
        total += len(block)
        urls.append(url)

    return "\n\n".join(parts).strip(), urls


def build_messages(
    question: str,
    retrieved: List[RetrievedChunk],
    history: List[Tuple[str, str]],
    history_turns: int,
    max_source_chars: int,
    max_total_chars: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    context, urls = build_context_snippets(retrieved, max_source_chars, max_total_chars)

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history and history_turns > 0:
        recent = history[-history_turns:]
        for u, a in recent:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})

    user_prompt = f"""Question:
{question}

Appdome KB Sources (use these only):
{context if context else "[No relevant sources retrieved]"}

Instructions:
- Answer using only the sources above.
- If sources are insufficient, explicitly say so and ask a focused clarifying question.
- Include at least one source URL.
"""
    msgs.append({"role": "user", "content": user_prompt})
    return msgs, urls


def enforce_sources(answer: str, urls: List[str]) -> str:
    if re.search(r"https?://", answer):
        return answer.strip()
    if urls:
        return (answer.rstrip() + "\n\nSources: " + " ".join(urls[:3])).strip()
    return (answer.rstrip() + "\n\nSources: [no KB sources retrieved]").strip()


# ----------------------------
# RAG Engine
# ----------------------------

class ThreatExpertRAG:
    def __init__(
        self,
        index_dir: Path,
        embed_model_name_or_path: str,
        device: Optional[str],
        force_offline: bool,
        llm_model: str,
        ollama_url: str,
        top_k: int,
        min_score: float,
        max_source_chars: int,
        max_total_context_chars: int,
        history_turns: int,
        temperature: float,
        seed: int,
        trust_env: bool = False,
        debug: bool = False,
    ) -> None:
        self.index_dir = index_dir
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.top_k = top_k
        self.min_score = min_score
        self.max_source_chars = max_source_chars
        self.max_total_context_chars = max_total_context_chars
        self.history_turns = history_turns
        self.temperature = temperature
        self.seed = seed
        self.trust_env = trust_env
        self.debug = debug

        self.session = make_session(trust_env=self.trust_env)

        self.index = load_faiss_index(index_dir / "index.faiss")
        self.chunks = load_chunks(index_dir / "chunks.jsonl")
        self.manifest = load_manifest(index_dir / "manifest.json")

        # sanity: FAISS size should align with chunk list
        try:
            ntotal = int(self.index.ntotal)
            if ntotal != len(self.chunks):
                logging.warning("FAISS ntotal=%d but chunks=%d. Retrieval may misalign.", ntotal, len(self.chunks))
        except Exception:
            pass

        self.embed_model = load_embedding_model(embed_model_name_or_path, device=device, force_offline=force_offline)
        self.history: List[Tuple[str, str]] = []

    def answer(self, question: str) -> str:
        q = normalize_ws(question)
        if not q:
            return "Please ask a question about Appdome threats, detections, or protections (e.g., jailbreak/root detection, ATO prevention, anti-tampering)."

        qvec = embed_query(self.embed_model, q)
        retrieved = retrieve(self.index, self.chunks, qvec, top_k=self.top_k, min_score=self.min_score)

        if self.debug:
            logging.debug("Retrieved %d chunks for query: %s", len(retrieved), q)
            for r in retrieved[: min(5, len(retrieved))]:
                logging.debug("  score=%.3f url=%s section=%s", r.score, r.row.url, r.row.section)

        messages, urls = build_messages(
            question=q,
            retrieved=retrieved,
            history=self.history,
            history_turns=self.history_turns,
            max_source_chars=self.max_source_chars,
            max_total_chars=self.max_total_context_chars,
        )

        if not retrieved:
            urls = []

        try:
            raw = generate_answer(
                session=self.session,
                ollama_url=self.ollama_url,
                llm_model=self.llm_model,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
            )
        except (requests.exceptions.RequestException, OllamaAPIError) as e:
            return classify_ollama_error(e, self.ollama_url, self.llm_model, self.trust_env).strip()

        final = enforce_sources(raw, urls if urls else [r.row.url for r in retrieved[:3] if r.row.url])
        self.history.append((q, final))
        return final


# ----------------------------
# UI / main
# ----------------------------

BANNER = r"""
🔐  Appdome Threat-Expert
Type your threat/security question.
Commands: /exit  /quit  /clear  /sources
"""


def print_sources_last(answer: str) -> None:
    urls = re.findall(r"https?://\S+", answer)
    if not urls:
        print("No sources detected in the last answer.")
        return
    print("Sources:")
    for u in urls:
        print(f"- {u.rstrip(').,;')}")


def main() -> int:
    args = parse_kv_args(sys.argv[1:])

    debug = _to_bool(str(args.get("debug", "false"))) if "debug" in args else False
    setup_logging(debug=debug)

    index_dir = Path(str(args.get("index-dir", DEFAULT_INDEX_DIR)))
    llm_model = str(args.get("llm", DEFAULT_LLM_MODEL))
    ollama_url = str(args.get("ollama", DEFAULT_OLLAMA_URL))

    top_k = int(args.get("top-k", DEFAULT_TOP_K))
    min_score = float(args.get("min-score", DEFAULT_MIN_SCORE))
    max_source_chars = int(args.get("max-source-chars", DEFAULT_MAX_SOURCE_CHARS))
    max_total_ctx = int(args.get("max-context-chars", DEFAULT_MAX_TOTAL_CONTEXT_CHARS))
    history_turns = int(args.get("history", DEFAULT_HISTORY_TURNS))
    temperature = float(args.get("temperature", DEFAULT_TEMPERATURE))
    seed = int(args.get("seed", DEFAULT_SEED))

    force_offline = not _to_bool(str(args.get("no-offline", "false")))  # default True
    device = args.get("device", DEFAULT_DEVICE)

    trust_env = _to_bool(str(args.get("trust-env", "false"))) if "trust-env" in args else False

    question = str(args.get("question", "")).strip()

    embed_model = str(args.get("embed-model", "")).strip()
    manifest = load_manifest(index_dir / "manifest.json")
    if not embed_model:
        embed_model = str(manifest.get("model") or DEFAULT_EMBED_MODEL_FALLBACK)

    try:
        rag = ThreatExpertRAG(
            index_dir=index_dir,
            embed_model_name_or_path=embed_model,
            device=str(device) if device else None,
            force_offline=force_offline,
            llm_model=llm_model,
            ollama_url=ollama_url,
            top_k=top_k,
            min_score=min_score,
            max_source_chars=max_source_chars,
            max_total_context_chars=max_total_ctx,
            history_turns=history_turns,
            temperature=temperature,
            seed=seed,
            trust_env=trust_env,
            debug=debug,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        logging.error("Did you run ingest.py to build the index? Expected in: %s", index_dir)
        return 2
    except Exception as e:
        logging.error("Failed to initialize RAG engine: %s", repr(e))
        return 3

    if question:
        print(rag.answer(question))
        return 0

    print(BANNER.strip())
    last_answer = ""

    while True:
        try:
            q = input("✦ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n/exit")
            break

        if not q:
            continue
        cmd = q.strip().lower()
        if cmd in {"/exit", "/quit"}:
            break
        if cmd == "/clear":
            rag.history.clear()
            last_answer = ""
            print("Context cleared.")
            continue
        if cmd == "/sources":
            print_sources_last(last_answer)
            continue

        print("✦ Agent: ", end="", flush=True)
        t0 = time.time()
        ans = rag.answer(q)
        dt = time.time() - t0
        last_answer = ans
        print(ans)
        if debug:
            print(f"\n(debug) answered in {dt:.1f}s at {utc_now_iso()}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
