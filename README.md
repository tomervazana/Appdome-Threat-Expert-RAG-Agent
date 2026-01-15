# Appdome Threat Expert RAG Agent (Offline)

A terminal-based (and optional browser-based) chat agent that answers **threat/security questions** using **Retrieval Augmented Generation (RAG)** grounded in **Appdome’s How-To KB**: [https://www.appdome.com/how-to/](https://www.appdome.com/how-to/)

This solution is designed to be **fully offline for embeddings + retrieval**, and uses a **local open-source LLM** via **Ollama** (e.g., `llama3`) for generation. The system enforces **grounded answers** and **requires at least one KB source URL** per response.

---

## High-Level Architecture

**Data plane**

1. **Crawl** Appdome How-To KB → structured JSON docs in `data/raw/`
2. **Ingest** JSON docs → chunking + sentence-transformer embeddings → FAISS index in `data/index/`
3. **Serve** chat agent → FAISS retrieve → grounded prompt → local LLM generate → answer + citations

**Control plane**

* Resume-safe crawler state (idempotent)
* Reproducible indexing (`manifest.json`)
* Deterministic / mockable generation in tests
* Proxy-safe localhost calls (critical in corporate environments)

---

## Repo Structure

```
.
├── crawler.py                # Scrape Appdome How-To KB into JSON
├── ingest.py                 # Build embeddings + FAISS index from crawled JSON
├── chat.py                   # CLI RAG agent (retrieve → prompt → generate)
├── web_ui.py                 # (Bonus) Streamlit web chat UI
├── api_server.py             # (Bonus) FastAPI backend + minimal HTML chat
├── data/
│   ├── raw/                  # Output of crawler (JSON per article)
│   └── index/                # Output of ingest (FAISS + chunk metadata)
└── tests/                    # pytest suite (unit + end-to-end mocked + metrics)
```

---

## Quickstart (End-to-End)

### 0) Create venv + install baseline deps

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/WSL
source .venv/bin/activate

pip install -U pip
```

### 1) Crawl Appdome How-To KB → `data/raw/`

```bash
pip install requests beautifulsoup4 lxml
python crawler.py --out data/raw --workers 4 --delay 0.4
```

### 2) Build embeddings + FAISS index → `data/index/`

```bash
pip install sentence-transformers faiss-cpu numpy
python ingest.py --raw-dir data/raw --out-dir data/index --rebuild true
```

### 3) Chat (CLI)

```bash
pip install requests numpy faiss-cpu sentence-transformers
python chat.py
```

---

## `crawler.py`

### How the crawler works (high level)

* **Discovery loop**: BFS over links under the `/how-to` prefix.
* **Article gating**: saves only pages containing the canonical “Last updated …” marker (separates KB articles from landing pages).
* **Structured extraction** (semantic, not HTML soup):

  * Preserves **semantic blocks**: heading, paragraph, list (nested), code, table, blockquote, image.
  * Handles Appdome’s code samples rendered via **CodeMirror** (avoids capturing each line as a separate block).
* **Resume-safe**: stores visited URLs and errors in `_crawl_state.json`.

### Run instructions

From repo root:

```bash
pip install requests beautifulsoup4 lxml
python crawler.py
```

Useful options:

```bash
python crawler.py --out data/raw --workers 4 --delay 0.4
python crawler.py --max-pages 50          # smoke run
python crawler.py --no-resume             # ignore previous crawl state
python crawler.py --seed https://www.appdome.com/how-to/
```

Outputs:

* `data/raw/<slug>_<sha1>.json` per article
* `data/raw/index.json` (quick lookup of saved docs)
* `data/raw/_crawl_state.json` (resume state + errors)

### JSON schema (what you’ll get per article)

Each file looks like:

```json
{
  "url": "...",
  "title": "...",
  "breadcrumbs": ["How to", "...", "...", "..."],
  "last_updated": "YYYY-MM-DD",
  "author": "Appdome",
  "scraped_at": "YYYY-MM-DDTHH:MM:SSZ",
  "source": "appdome-how-to",
  "content": {
    "blocks": [ { "type": "...", "...": "..." } ],
    "text": "plain text stitched from blocks",
    "images": [ { "src": "...", "alt": "..." } ]
  }
}
```

---

## `ingest.py`

### What `ingest.py` produces

By default it writes into `data/index/`:

* `index.faiss` — FAISS index (**IndexFlatIP** + normalized vectors → cosine similarity)
* `chunks.jsonl` — one JSON per chunk (metadata + text), aligned with FAISS vector order
* `docs.jsonl` — original document metadata
* `manifest.json` — reproducibility metadata (model, params, counts, etc.)

### Install deps

```bash
pip install sentence-transformers faiss-cpu numpy
```

(If you have CUDA and want GPU FAISS, use `faiss-gpu` instead.)

### Run

Smoke run:

```bash
python ingest.py --raw-dir data/raw --out-dir data/index --limit 20 --rebuild true
```

Full run:

```bash
python ingest.py --raw-dir data/raw --out-dir data/index --rebuild true
```

Full-offline run (recommended for “offline” evaluation):

```bash
python ingest.py --model /absolute/path/to/sentence-transformer-model --rebuild true
```

---

## `chat.py` (CLI RAG Agent)

### What it does

* Loads `data/index/index.faiss` and `data/index/chunks.jsonl`
* Embeds the user query using sentence-transformers (offline-capable)
* Retrieves top-k with FAISS
* Builds a **grounded prompt** that includes:

  * chunk excerpts
  * relevance scores
  * source URLs
* Calls **local Ollama** (`llama3` by default)
* Enforces:

  * **No hallucinated steps** (prompt rules)
  * **At least one URL citation** (post-processing safety net)

### Run

Interactive mode:

```bash
python chat.py
```

Single question:

```bash
python chat.py --question "How do I integrate Threat Remediation Center in my iOS app?"
```

Useful knobs:

```bash
python chat.py --index-dir data/index --ollama http://localhost:11434 --llm llama3
python chat.py --top-k 8 --min-score 0.12
python chat.py --debug true
```

### Ollama requirements

Make sure Ollama is running and the model exists locally:

```bash
ollama serve
ollama pull llama3
```

**Common pitfall (strong opinion):** if you’re in a proxy-heavy environment, `requests` can route `localhost` through the proxy and fail even though the browser says “Ollama is running”.
This project defaults to **proxy-safe** behavior (do not trust proxy env vars). If needed:

* set `NO_PROXY=localhost,127.0.0.1`
* or run `chat.py` with `--trust-env true` (only if you actually want env proxies)

---

## Bonus: Web UI (Browser Chat)

Two minimal options are included so you can demo like a product, not a script.

### Option A — Streamlit (fastest demo)

```bash
pip install -r requirements_webui.txt
streamlit run web_ui.py
```

### Option B — FastAPI (best for plugging into any web chat template)

```bash
pip install -r requirements_api.txt
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

* Open: `http://127.0.0.1:8000/`
* API endpoint:

  * `POST /chat` → `{ "message": "..." }` → `{ "answer": "...", "sources": [...] }`

This is the clean integration point if you found a web template and want to “swap in” our agent behind it.

---

## Tests (`pytest`)

### Install

```bash
pip install pytest
```

### Run

```bash
pytest -q
```

### What’s covered

* **Crawler correctness**

  * URL canonicalization + `/how-to` prefix gating regression (prevents “Visited=0” failures)
* **Ingest correctness**

  * heading-aware sectioning
  * chunk sizing guarantees
* **RAG core**

  * retrieval ordering
  * prompt includes source URLs
  * citations enforced if LLM forgets
* **End-to-end (mocked)**

  * deterministic embeddings
  * mocked generation (no network)
* **Agent correctness metrics**

  * classic IR metrics: **Hit@k, Recall@k, Precision@k, MRR, nDCG@k**
  * RAG proxies: context precision/recall
  * grounding proxy: sentence-level faithfulness (citation lines excluded)
  * answer correctness proxy: token-F1 against lightweight gold

---

## Design Notes (why these choices)

* **Semantic scraping > HTML scraping**: The crawler extracts *meaning*, not markup.
* **FAISS IndexFlatIP + normalized vectors**: simple, fast, reproducible cosine similarity baseline.
* **Strict grounding + citations**: makes the agent defensible in security workflows (and aligns with evaluation criteria).
* **Mockable generation**: tests don’t depend on Ollama runtime; only the demo does.