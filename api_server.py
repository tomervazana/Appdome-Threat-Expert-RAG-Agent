#!/usr/bin/env python3
"""
api_server.py (FastAPI)
-----------------------
A minimal HTTP API wrapper around the Appdome Threat-Expert RAG agent.

Why this exists:
- Lets you plug *any* web chat template (HTML/JS/React/etc.) into your agent
- Keeps the core RAG logic in chat.py
- Runs locally, fully offline (except calls to local Ollama)

Run:
  pip install fastapi uvicorn
  uvicorn api_server:app --host 127.0.0.1 --port 8000

Then open:
  http://127.0.0.1:8000/

Environment variables (optional):
  INDEX_DIR        (default: data/index)
  EMBED_MODEL      (default: uses chat.py defaults / manifest)
  OLLAMA_URL       (default: http://localhost:11434)
  OLLAMA_MODEL     (default: llama3)
  FORCE_OFFLINE    (default: 1)
  TOP_K            (default: 6)
  MIN_SCORE        (default: 0.15)
  TRUST_ENV        (default: 0)  # proxy env vars (usually keep 0)
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    import chat
except Exception as e:
    raise RuntimeError(
        "Failed to import chat.py. Place api_server.py in the same folder as chat.py."
    ) from e

URL_RE = re.compile(r"https?://\S+")


def _to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _extract_sources(text: str):
    urls = URL_RE.findall(text or "")
    cleaned = [u.rstrip(").,;") for u in urls]
    out = []
    seen = set()
    for u in cleaned:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _default_embed_model(index_dir: Path) -> str:
    # Prefer ingest manifest if exists, otherwise chat.py fallback.
    try:
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            import json
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(m, dict) and m.get("model"):
                return str(m["model"])
    except Exception:
        pass
    return str(getattr(chat, "DEFAULT_EMBED_MODEL_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2"))


# ---- FastAPI app ----
app = FastAPI(title="Appdome Threat‑Expert API", version="1.0")

RAG = None  # initialized on startup


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.on_event("startup")
def _startup() -> None:
    global RAG

    index_dir = Path(os.environ.get("INDEX_DIR", str(getattr(chat, "DEFAULT_INDEX_DIR", "data/index"))))
    embed_model = os.environ.get("EMBED_MODEL", "").strip() or _default_embed_model(index_dir)

    kwargs = dict(
        index_dir=index_dir,
        embed_model_name_or_path=embed_model,
        device=os.environ.get("DEVICE", None),
        force_offline=_to_bool(os.environ.get("FORCE_OFFLINE", "1"), True),
        llm_model=os.environ.get("OLLAMA_MODEL", str(getattr(chat, "DEFAULT_LLM_MODEL", "llama3"))),
        ollama_url=os.environ.get("OLLAMA_URL", str(getattr(chat, "DEFAULT_OLLAMA_URL", "http://localhost:11434"))),
        top_k=int(os.environ.get("TOP_K", str(getattr(chat, "DEFAULT_TOP_K", 6)))),
        min_score=float(os.environ.get("MIN_SCORE", str(getattr(chat, "DEFAULT_MIN_SCORE", 0.15)))),
        max_source_chars=int(os.environ.get("MAX_SOURCE_CHARS", str(getattr(chat, "DEFAULT_MAX_SOURCE_CHARS", 900)))),
        max_total_context_chars=int(os.environ.get("MAX_CONTEXT_CHARS", str(getattr(chat, "DEFAULT_MAX_TOTAL_CONTEXT_CHARS", 5500)))),
        history_turns=int(os.environ.get("HISTORY_TURNS", str(getattr(chat, "DEFAULT_HISTORY_TURNS", 4)))),
        temperature=float(os.environ.get("TEMPERATURE", str(getattr(chat, "DEFAULT_TEMPERATURE", 0.2)))),
        seed=int(os.environ.get("SEED", str(getattr(chat, "DEFAULT_SEED", 42)))),
        trust_env=_to_bool(os.environ.get("TRUST_ENV", "0"), False),
        debug=_to_bool(os.environ.get("DEBUG", "0"), False),
    )

    kwargs = _filter_kwargs_for_callable(chat.ThreatExpertRAG, kwargs)
    RAG = chat.ThreatExpertRAG(**kwargs)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "agent_loaded": RAG is not None}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if RAG is None:
        return ChatResponse(answer="Agent not initialized.", sources=[])
    answer = RAG.answer(req.message)
    return ChatResponse(answer=answer, sources=_extract_sources(answer))


# --- Minimal HTML UI (single file) ---

_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Appdome Threat‑Expert</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b1220; color: #e5e7eb; margin: 0; }
    .wrap { max-width: 920px; margin: 0 auto; padding: 24px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 16px; }
    .title { display:flex; align-items:center; gap:10px; font-weight: 700; font-size: 18px; }
    .title span { font-size: 22px; }
    .chat { margin-top: 14px; display:flex; flex-direction: column; gap: 10px; min-height: 60vh; }
    .msg { padding: 12px 14px; border-radius: 12px; max-width: 90%; white-space: pre-wrap; line-height: 1.35; }
    .user { background: #1d4ed8; align-self: flex-end; }
    .bot { background: #0f172a; border: 1px solid #1f2937; align-self: flex-start; }
    .row { display:flex; gap:10px; margin-top: 14px; }
    input { flex:1; padding: 12px 12px; border-radius: 12px; border: 1px solid #334155; background: #0f172a; color: #e5e7eb; }
    button { padding: 12px 14px; border-radius: 12px; border: 1px solid #334155; background: #111827; color: #e5e7eb; cursor:pointer; }
    button:hover { background: #0f172a; }
    .hint { margin-top: 10px; font-size: 13px; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title"><span>🔐</span> Appdome Threat‑Expert</div>
      <div class="hint">Grounded answers from your local KB index. Every answer includes sources.</div>
      <div id="chat" class="chat"></div>
      <div class="row">
        <input id="q" placeholder="Ask a threat/security question…" />
        <button onclick="send()">Send</button>
      </div>
    </div>
  </div>

<script>
const chat = document.getElementById("chat");
const q = document.getElementById("q");

function add(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + (role === "user" ? "user" : "bot");
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const msg = (q.value || "").trim();
  if (!msg) return;
  add("user", msg);
  q.value = "";

  add("bot", "…");
  const last = chat.lastChild;

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: msg})
    });
    const data = await resp.json();
    last.textContent = data.answer || "(empty)";
  } catch (e) {
    last.textContent = "Request failed: " + e;
  }
}

q.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});

</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_HTML)
