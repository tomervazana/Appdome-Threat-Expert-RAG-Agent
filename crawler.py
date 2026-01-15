#!/usr/bin/env python3
"""
Appdome How‑To KB crawler (structured scraper)

Outputs: one JSON file per KB article into data/raw/

Design goals:
- Crawl /how-to/ pages and discover all KB article URLs by following internal links.
- Extract *structured* content (headings, paragraphs, lists, code blocks, tables, images) instead of dumping raw HTML.
- Be polite: global rate limiting + retry/backoff.
- Simple CLI parsing (no argparse).

Dependencies:
  pip install requests beautifulsoup4 lxml

Usage examples:
  python crawler.py
  python crawler.py --out data/raw --workers 4 --delay 0.4
  python crawler.py --max-pages 50   # smoke run

Notes:
- The crawler saves only pages that look like KB articles (detected by "Last updated ...").
- Category/landing pages under /how-to/ are crawled for link discovery but are not persisted by default.

JSON schema:
{
  "url": "...",
  "title": "...",
  "breadcrumbs": ["How to", "...", "...", "..."],
  "last_updated": "YYYY-MM-DD",
  "author": "Appdome",
  "scraped_at": "YYYY-MM-DDTHH:MM:SSZ",
  "source": "appdome-how-to",
  "content": {
    "blocks": [ { "type": "...", ... } ],
    "text": "plain text stitched from blocks",
    "images": [ { "src": "...", "alt": "..." } ]
  }
}

"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


BASE_URL = "https://www.appdome.com"
START_URL = f"{BASE_URL}/how-to"
ALLOWED_PREFIX = START_URL  # canonical prefix (no trailing slash)


# -----------------------------
# Utilities
# -----------------------------

def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat() + "Z"


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    """Normalize URL: https scheme, drop query+fragment, squash //, drop trailing slash."""
    p = urlparse(urljoin(BASE_URL, url))
    scheme = "https"
    netloc = p.netloc or urlparse(BASE_URL).netloc
    path = re.sub(r"//+", "/", p.path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, netloc, path, "", "", ""))


def safe_filename_for_url(url: str) -> str:
    """Stable filename from URL: <slug>_<sha1_10>.json"""
    p = urlparse(url)
    slug = (p.path.strip("/").split("/")[-1] or "index").lower()
    slug = re.sub(r"[^a-z0-9_-]+", "_", slug)[:80]
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{h}.json"


class RateLimiter:
    """Global min-interval rate limiter (thread-safe)."""
    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        sleep_s = 0.0
        with self._lock:
            now = time.time()
            if now < self._next_allowed:
                sleep_s = self._next_allowed - now
                self._next_allowed = self._next_allowed + self.min_interval_s
            else:
                self._next_allowed = now + self.min_interval_s
        if sleep_s > 0:
            time.sleep(sleep_s)


# -----------------------------
# Structured parsing
# -----------------------------

_CODEMIRROR_LINE_CLASS = "CodeMirror-line"
_CODEMIRROR_BLOCK_CLASS = "CodeMirror"

_HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = list(_HEADING_TAGS) + ["p", "ul", "ol", "pre", "table", "blockquote", "figure", "img"]


def _detect_code_language(code_tag: Optional[Tag]) -> Optional[str]:
    if not code_tag:
        return None
    cls = " ".join(code_tag.get("class", []) or [])
    m = re.search(r"(?:language|lang)-([a-z0-9_+-]+)", cls, flags=re.I)
    return m.group(1).lower() if m else None


def _clean_code(code: str) -> str:
    # Remove zero-width spaces & strip noisy line-number-only lines.
    code = code.replace("\u200b", "")
    lines = [ln.rstrip() for ln in code.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        if re.fullmatch(r"\s*\d+\s*", ln):
            continue
        if re.fullmatch(r"\s*x{5,}\s*", ln, flags=re.I):
            continue
        cleaned.append(ln)

    # Trim outer blanks
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _parse_list(list_tag: Tag) -> Dict[str, Any]:
    ordered = list_tag.name == "ol"
    items: List[Dict[str, Any]] = []
    for li in list_tag.find_all("li", recursive=False):
        # text excluding nested lists
        parts: List[str] = []
        for child in li.contents:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                continue
            if isinstance(child, NavigableString):
                if str(child).strip():
                    parts.append(str(child))
            elif isinstance(child, Tag):
                parts.append(child.get_text(" ", strip=True))
        text = clean_ws(" ".join(parts))

        nested = li.find(["ul", "ol"], recursive=False)
        children = _parse_list(nested)["items"] if nested else None

        # images inside the li (not ordered in the main flow, but captured semantically)
        images: List[Dict[str, Any]] = []
        for img in li.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                continue
            images.append(
                {
                    "src": urljoin(BASE_URL, src),
                    "alt": clean_ws(img.get("alt") or "") or None,
                }
            )

        items.append({"text": text, "children": children, "images": images or None})
    return {"type": "list", "ordered": ordered, "items": items}


def _parse_table(table: Tag) -> Dict[str, Any]:
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells: List[str] = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            cells.append(clean_ws(cell.get_text(" ", strip=True)))
        if cells:
            rows.append(cells)
    return {"type": "table", "rows": rows}


def _parse_figure(fig: Tag) -> Optional[Dict[str, Any]]:
    img = fig.find("img")
    if not img:
        return None
    src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
    if not src:
        return None
    cap = fig.find("figcaption")
    caption = clean_ws(cap.get_text(" ", strip=True)) if cap else None
    return {
        "type": "image",
        "src": urljoin(BASE_URL, src),
        "alt": clean_ws(img.get("alt") or "") or None,
        "caption": caption,
    }


def _parse_img(img: Tag) -> Optional[Dict[str, Any]]:
    src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
    if not src:
        return None
    return {
        "type": "image",
        "src": urljoin(BASE_URL, src),
        "alt": clean_ws(img.get("alt") or "") or None,
        "caption": None,
    }


def _strip_noise(root: Tag) -> None:
    for name in ("nav", "header", "footer", "aside", "form", "script", "style", "noscript"):
        for t in root.find_all(name):
            t.decompose()


def _extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        txt = clean_ws(h1.get_text(" ", strip=True))
        if txt:
            return txt
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return clean_ws(str(og["content"]))
    if soup.title and soup.title.string:
        return clean_ws(str(soup.title.string))
    return ""


def _extract_breadcrumbs(soup: BeautifulSoup) -> List[str]:
    # Primary: Appdome KB breadcrumb wrapper
    bc_wrapper = soup.find(class_=re.compile(r"kb-breadcrumbs-wrapper", re.I))
    if bc_wrapper:
        links = [clean_ws(a.get_text(" ", strip=True)) for a in bc_wrapper.find_all("a")]
        links = [l for l in links if l]
        if links:
            return links

    # Secondary: schema.org BreadcrumbList inside JSON-LD
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        txt = (script.string or script.get_text() or "").strip()
        if not txt:
            continue
        try:
            data = json.loads(txt)
        except Exception:
            continue

        objs: List[Dict[str, Any]] = []
        if isinstance(data, dict) and isinstance(data.get("@graph"), list):
            objs.extend([o for o in data["@graph"] if isinstance(o, dict)])
        elif isinstance(data, dict):
            objs.append(data)
        elif isinstance(data, list):
            objs.extend([o for o in data if isinstance(o, dict)])

        for obj in objs:
            if obj.get("@type") != "BreadcrumbList":
                continue
            items = obj.get("itemListElement")
            if not isinstance(items, list):
                continue
            crumbs: List[str] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                item = it.get("item")
                name = None
                if isinstance(item, dict):
                    name = item.get("name")
                if not name:
                    name = it.get("name")
                if name:
                    crumbs.append(clean_ws(str(name)))
            if crumbs:
                return crumbs

    return []


def _extract_last_updated(soup: BeautifulSoup, h1: Optional[Tag]) -> Tuple[Optional[str], Optional[str]]:
    # Search near the H1 first to avoid false positives.
    candidates: List[str] = []
    if h1:
        cur: Optional[Tag] = h1
        for _ in range(60):
            nxt = cur.find_next() if cur else None
            if nxt is None or not isinstance(nxt, Tag):
                break
            cur = nxt
            tx = clean_ws(nxt.get_text(" ", strip=True))
            if re.search(r"\bLast updated\b", tx, flags=re.I):
                candidates.append(tx)
                break

    if not candidates:
        for s in soup.find_all(string=re.compile(r"\bLast updated\b", re.I)):
            tx = clean_ws(str(s))
            if tx:
                candidates.append(tx)
                break

    for tx in candidates:
        m = re.search(
            r"Last updated\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})(?:\s+by\s+(.+?))?$",
            tx,
            flags=re.I,
        )
        if not m:
            continue
        date_str = m.group(1)
        author = clean_ws(m.group(2)) if m.group(2) else None
        try:
            dt = _dt.datetime.strptime(date_str, "%B %d, %Y").date()
            return dt.isoformat(), author
        except Exception:
            return date_str, author

    return None, None


def _find_content_root(soup: BeautifulSoup) -> Optional[Tag]:
    h1 = soup.find("h1")
    if h1:
        cand: Optional[Tag] = h1
        for _ in range(14):
            if cand is None:
                break
            txt = cand.get_text(" ", strip=True)
            if len(txt) > 500 and re.search(r"\bLast updated\b", txt, flags=re.I):
                return cand
            cand = cand.parent if isinstance(cand.parent, Tag) else None

    # Typical KB article root
    art = soup.find("article")
    if art and len(art.get_text(" ", strip=True)) > 500:
        return art
    _main = soup.find("main")
    if _main and len(_main.get_text(" ", strip=True)) > 500:
        return _main
    return None


def _iter_structural_elements(root: Tag) -> Iterable[Tag]:
    """Yield block-level elements in document order, skipping nested duplicates."""
    elements = root.find_all(_BLOCK_TAGS)
    captured: Set[int] = set()

    for el in elements:
        # Ignore per-line CodeMirror <pre> tags.
        if el.name == "pre":
            cls = set(el.get("class", []) or [])
            if _CODEMIRROR_LINE_CLASS in cls:
                continue

        # Avoid capturing elements nested inside a previously captured block
        if any(id(p) in captured for p in el.parents):
            continue

        # Ignore img inside figure (we capture the figure)
        if el.name == "img" and el.find_parent("figure"):
            continue

        captured.add(id(el))
        yield el


def _element_to_block(el: Tag) -> Optional[Dict[str, Any]]:
    if el.name in _HEADING_TAGS:
        return {"type": "heading", "level": int(el.name[1]), "text": clean_ws(el.get_text(" ", strip=True))}

    if el.name == "p":
        text = clean_ws(el.get_text(" ", strip=True))
        return {"type": "paragraph", "text": text} if text else None

    if el.name in ("ul", "ol"):
        return _parse_list(el)

    if el.name == "pre":
        cls = set(el.get("class", []) or [])
        if _CODEMIRROR_LINE_CLASS in cls:
            return None

        code_el = el.find("code")
        lang = _detect_code_language(code_el) if code_el else None
        raw = (el.get_text("\n", strip=False) or "")
        text = _clean_code(raw)
        # Discard tiny/noise blocks
        if len(text.strip()) < 10:
            return None
        return {"type": "code", "language": lang, "text": text}

    if el.name == "table":
        tbl = _parse_table(el)
        return tbl if tbl["rows"] else None

    if el.name == "blockquote":
        text = clean_ws(el.get_text(" ", strip=True))
        return {"type": "blockquote", "text": text} if text else None

    if el.name == "figure":
        return _parse_figure(el)

    if el.name == "img":
        return _parse_img(el)

    return None


def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []

    def _add_list_items(items: List[Dict[str, Any]]) -> None:
        for it in items:
            if it.get("text"):
                parts.append(str(it["text"]))
            if it.get("children"):
                _add_list_items(it["children"])

    for b in blocks:
        t = b.get("type")
        if t == "heading":
            parts.append(str(b.get("text", "")))
        elif t == "paragraph":
            parts.append(str(b.get("text", "")))
        elif t == "list":
            _add_list_items(b.get("items", []))
        elif t == "code":
            parts.append(str(b.get("text", "")))
        elif t == "table":
            for row in b.get("rows", []):
                parts.append(" | ".join(row))
        elif t == "blockquote":
            parts.append(str(b.get("text", "")))
        elif t == "image":
            if b.get("alt"):
                parts.append(str(b["alt"]))
            if b.get("caption"):
                parts.append(str(b["caption"]))

    return "\n".join([p for p in parts if p]).strip()


def parse_kb_article(url: str, html: str) -> Optional[Dict[str, Any]]:
    """
    Returns a structured doc dict for KB *article* pages.
    Returns None for landing pages.
    """
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)
    breadcrumbs = _extract_breadcrumbs(soup)

    h1 = soup.find("h1")
    last_updated, author = _extract_last_updated(soup, h1)

    # Filter: persist only KB articles
    if not last_updated:
        return None

    root = _find_content_root(soup)
    if not root:
        return None

    _strip_noise(root)

    blocks: List[Dict[str, Any]] = []
    for el in _iter_structural_elements(root):
        b = _element_to_block(el)
        if b:
            blocks.append(b)

    # Document-level image list (deduped)
    images: List[Dict[str, Any]] = []
    for img in root.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src:
            continue
        images.append(
            {
                "src": urljoin(BASE_URL, src),
                "alt": clean_ws(img.get("alt") or "") or None,
            }
        )
    seen_src: Set[str] = set()
    images_dedup: List[Dict[str, Any]] = []
    for im in images:
        if im["src"] in seen_src:
            continue
        seen_src.add(im["src"])
        images_dedup.append(im)

    doc_text = _blocks_to_text(blocks)

    return {
        "url": canonicalize_url(url),
        "title": title,
        "breadcrumbs": breadcrumbs,
        "last_updated": last_updated,
        "author": author,
        "scraped_at": _utc_now_iso(),
        "source": "appdome-how-to",
        "content": {
            "blocks": blocks,
            "text": doc_text,
            "images": images_dedup,
        },
    }


# -----------------------------
# Crawling
# -----------------------------

def extract_howto_links(soup: BeautifulSoup) -> Set[str]:
    urls: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        u = canonicalize_url(href)
        if u == ALLOWED_PREFIX or u.startswith(ALLOWED_PREFIX + "/"):
            urls.add(u)
    return urls


def fetch_html(
    session: requests.Session,
    url: str,
    rate_limiter: RateLimiter,
    timeout_s: float = 25.0,
    max_retries: int = 4,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Returns (html, final_url, status_code).
    """
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            resp = session.get(url, timeout=timeout_s, allow_redirects=True)
            status = resp.status_code

            if status in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)
                continue

            if status != 200:
                return None, resp.url, status

            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                return None, resp.url, status

            resp.encoding = resp.encoding or "utf-8"
            return resp.text, resp.url, status

        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(30.0, backoff * 2.0)

    return None, None, None


@dataclass
class CrawlerConfig:
    out_dir: Path = Path("data/raw")
    seeds: List[str] = None  # type: ignore[assignment]
    workers: int = 4
    delay_s: float = 0.35
    max_pages: int = 0  # 0 = unlimited
    resume: bool = True

    def __post_init__(self) -> None:
        if self.seeds is None:
            self.seeds = [START_URL]


class AppdomeKBCrawler:
    def __init__(self, cfg: CrawlerConfig) -> None:
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.cfg.out_dir / "_crawl_state.json"
        self.index_path = self.cfg.out_dir / "index.json"

        self.rate_limiter = RateLimiter(self.cfg.delay_s)
        self.todo: "queue.Queue[Optional[str]]" = queue.Queue()

        self.visited: Set[str] = set()
        self._visited_lock = threading.Lock()

        self.index: List[Dict[str, Any]] = []
        self._index_lock = threading.Lock()

        self.errors: List[Dict[str, Any]] = []
        self._errors_lock = threading.Lock()

        if self.cfg.resume:
            self._load_state()

        for seed in self.cfg.seeds:
            self._enqueue(seed)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            urls = data.get("visited_urls", [])
            if isinstance(urls, list):
                self.visited.update([canonicalize_url(u) for u in urls if isinstance(u, str)])
            idx = data.get("index", [])
            if isinstance(idx, list):
                self.index = idx
            errs = data.get("errors", [])
            if isinstance(errs, list):
                self.errors = errs
            logging.info("Resumed: %d visited, %d docs indexed, %d errors", len(self.visited), len(self.index), len(self.errors))
        except Exception as e:
            logging.warning("Failed loading state: %s", e)

    def _save_state(self) -> None:
        payload = {
            "visited_urls": sorted(self.visited),
            "index": self.index,
            "errors": self.errors,
            "updated_at": _utc_now_iso(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

        # Keep index as a separate convenient file
        tmp2 = self.index_path.with_suffix(".tmp")
        tmp2.write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp2.replace(self.index_path)

    def _enqueue(self, url: str) -> None:
        u = canonicalize_url(url)
        if not (u == ALLOWED_PREFIX or u.startswith(ALLOWED_PREFIX + "/")):
            return

        with self._visited_lock:
            if u in self.visited:
                return
            if self.cfg.max_pages and len(self.visited) >= self.cfg.max_pages:
                return
            self.visited.add(u)

        self.todo.put(u)

    def _record_error(self, url: str, status: Optional[int], message: str) -> None:
        with self._errors_lock:
            self.errors.append(
                {"url": url, "status": status, "message": message, "at": _utc_now_iso()}
            )

    def _save_doc(self, doc: Dict[str, Any]) -> None:
        url = doc["url"]
        path = self.cfg.out_dir / safe_filename_for_url(url)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        entry = {
            "url": url,
            "file": str(path.relative_to(self.cfg.out_dir)),
            "title": doc.get("title"),
            "last_updated": doc.get("last_updated"),
            "breadcrumbs": doc.get("breadcrumbs"),
        }
        with self._index_lock:
            self.index.append(entry)

    def _worker(self) -> None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "AppdomeThreatExpertRAG/1.0 (educational crawler; contact: none)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

        processed = 0
        while True:
            url = self.todo.get()
            if url is None:
                self.todo.task_done()
                return

            html, final_url, status = fetch_html(session, url, self.rate_limiter)
            if not html:
                self._record_error(url, status, "fetch_failed")
                self.todo.task_done()
                continue

            final = canonicalize_url(final_url or url)
            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(html, "html.parser")

            # Discover new links
            for link in extract_howto_links(soup):
                self._enqueue(link)

            # Parse & persist article docs
            try:
                doc = parse_kb_article(final, html)
                if doc:
                    self._save_doc(doc)
            except Exception as e:
                self._record_error(final, status, f"parse_failed: {e}")

            processed += 1
            if processed % 25 == 0:
                # Periodic state flush
                try:
                    self._save_state()
                except Exception:
                    pass

            self.todo.task_done()

    def run(self) -> None:
        logging.info("Starting crawl with %d workers. Output: %s", self.cfg.workers, self.cfg.out_dir)
        threads = []
        for i in range(max(1, int(self.cfg.workers))):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        # Wait until the queue drains
        self.todo.join()

        # Stop workers
        for _ in threads:
            self.todo.put(None)
        for t in threads:
            t.join()

        self._save_state()
        logging.info("Done. Visited=%d, SavedDocs=%d, Errors=%d", len(self.visited), len(self.index), len(self.errors))


# -----------------------------
# CLI (no argparse)
# -----------------------------

def _usage() -> str:
    return (
        "Usage: python crawler.py [--out DIR] [--workers N] [--delay SEC] [--max-pages N] [--no-resume] [--seed URL ...]\n"
        "Defaults: --out data/raw --workers 4 --delay 0.35 --seed https://www.appdome.com/how-to\n"
    )


def _parse_cli(argv: List[str]) -> CrawlerConfig:
    out_dir = Path("data/raw")
    workers = 4
    delay_s = 0.35
    max_pages = 0
    resume = True
    seeds: List[str] = []

    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(_usage())
            sys.exit(0)
        if a == "--out":
            i += 1
            out_dir = Path(argv[i])
        elif a == "--workers":
            i += 1
            workers = int(argv[i])
        elif a == "--delay":
            i += 1
            delay_s = float(argv[i])
        elif a == "--max-pages":
            i += 1
            max_pages = int(argv[i])
        elif a == "--no-resume":
            resume = False
        elif a == "--seed":
            i += 1
            seeds.append(argv[i])
        else:
            print(f"Unknown arg: {a}\n{_usage()}")
            sys.exit(2)
        i += 1

    if not seeds:
        seeds = [START_URL]

    return CrawlerConfig(out_dir=out_dir, seeds=seeds, workers=workers, delay_s=delay_s, max_pages=max_pages, resume=resume)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cfg = _parse_cli(sys.argv)
    crawler = AppdomeKBCrawler(cfg)
    crawler.run()


if __name__ == "__main__":
    main()
