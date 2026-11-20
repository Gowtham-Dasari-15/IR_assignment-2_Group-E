"""
IR Studio - Information Retrieval Assignment 2 (AIMLCZG537 / DSECLZG537), Group E

End-to-end IR system in one Streamlit file:
    acquisition (dataset / crawler / API) -> preprocessing & text mining -> indexing
    -> search & ranking (BM25, TF-IDF, Boolean, PageRank, HITS) -> recommendation
    -> evaluation (P, R, F1, P@K, R@K, MAP, MRR, NDCG) -> performance analytics

Run:  streamlit run app.py
"""
import hashlib
import json
import math
import re
import time
import urllib.robotparser
from collections import Counter, defaultdict, deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer, SnowballStemmer
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent
SAMPLE_DIR = BASE / "data" / "sample"
STORE_DIR = BASE / "data" / "store"
USER_AGENT = "IR-Assignment-Crawler/1.0 (educational)"


# =============================================================================
# 1. TEXT PREPROCESSING
# =============================================================================
# scikit-learn's stop list contains content words ("interest rates", "computer system"),
# so we keep those as index terms.
CONTENT_WORDS = {
    "interest", "computer", "system", "bill", "fire", "find", "found", "show", "call", "describe",
    "detail", "mill", "thick", "thin", "cry", "front", "back", "bottom", "top", "side", "full",
    "part", "amount", "name", "move", "fill", "empty", "serious", "con", "de",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - CONTENT_WORDS)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
porter, snowball = PorterStemmer(), SnowballStemmer("english")


@lru_cache(maxsize=200_000)
def stem_word(token, algo):
    if algo == "porter":
        return porter.stem(token)
    if algo == "snowball":
        return snowball.stem(token)
    return token


def tokenize(text, stop=True, stem="porter", min_len=2, drop_numbers=False):
    """clean -> lowercase -> tokenize -> filter -> stem."""
    text = re.sub(r"<[^>]+>|https?://\S+", " ", str(text)).lower()
    out = []
    for t in TOKEN_RE.findall(text):
        if len(t) < min_len or (drop_numbers and t.isdigit()) or (stop and t in STOP_WORDS):
            continue
        out.append(stem_word(t, stem))
    return out


PRESETS = {  # preprocessing strategies compared on the Text mining page
    "Raw tokens": dict(stop=False, stem="none", min_len=1),
    "Stop-word removal": dict(stop=True, stem="none"),
    "Stop words + Porter": dict(stop=True, stem="porter"),
    "Stop words + Snowball": dict(stop=True, stem="snowball"),
}


# =============================================================================
# 2. DATA ACQUISITION AND DUPLICATE HANDLING
# =============================================================================
DOC_COLS = ["doc_id", "title", "text"]
META_COLS = ["doc_id", "url", "source", "source_type", "category", "depth", "fetched_at",
             "n_words", "content_hash", "simhash"]


def normalize_url(url):
    """Canonical URL: lowercase host, no www., no default port, no trailing slash, no utm_*."""
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    host = re.sub(r":(80|443)$", "", host)
    path = re.sub(r"/+", "/", p.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")))
    return urlunparse(((p.scheme or "http").lower(), host, path, "", query, ""))


def content_hash(text):
    """MD5 of the normalised text: identical documents get identical hashes."""
    norm = re.sub(r"\W+", " ", str(text).lower()).strip()
    return hashlib.md5(norm.encode()).hexdigest()


def simhash(text, bits=64):
    """SimHash over word 3-shingles: similar documents get fingerprints differing in few bits."""
    words = re.findall(r"\w+", str(text).lower())
    shingles = [" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))]
    v = np.zeros(bits)
    for sh in shingles:
        h = int(hashlib.md5(sh.encode()).hexdigest()[:16], 16)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    return sum(1 << b for b in range(bits) if v[b] > 0)


def hamming(a, b):
    return bin(a ^ b).count("1")


def near_duplicates(meta, max_dist=6):
    """All pairs of documents whose SimHash fingerprints differ in <= max_dist bits."""
    ids = meta["doc_id"].tolist()
    hashes = []
    for h in meta["simhash"]:
        try:
            hashes.append(int(h))
        except (ValueError, TypeError):
            hashes.append(0)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = hamming(hashes[i], hashes[j])
            if d <= max_dist:
                pairs.append({"doc_a": ids[i], "doc_b": ids[j], "hamming": d, "similarity": round(1 - d / 64, 3)})
    return pd.DataFrame(pairs, columns=["doc_a", "doc_b", "hamming", "similarity"])


def empty_corpus():
    return {"docs": pd.DataFrame(columns=DOC_COLS), "meta": pd.DataFrame(columns=META_COLS),
            "edges": pd.DataFrame(columns=["src", "dst"])}


def add_documents(rows, source, source_type, dedup=True):
    """Add rows (dicts with title, text, url?, category?, doc_id?, links?) to the corpus.
    Skips duplicate URLs and (optionally) exact-duplicate content. Returns a report."""
    c = st.session_state.corpus
    seen_urls = {normalize_url(u) for u in c["meta"]["url"] if u}
    seen_hash = set(c["meta"]["content_hash"])
    used_ids = set(c["docs"]["doc_id"])
    next_num = len(c["docs"]) + 1
    new_docs, new_meta, id_map, skipped = [], [], {}, []
    clean = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
    for r in rows:
        text = clean(r.get("text")).strip()
        if not text:
            continue
        url = clean(r.get("url"))
        url_n = normalize_url(url) if url else ""
        h = content_hash(text)
        if url_n and url_n in seen_urls:
            skipped.append({"title": clean(r.get("title"))[:80], "reason": "duplicate URL", "key": url_n})
            continue
        if dedup and h in seen_hash:
            skipped.append({"title": clean(r.get("title"))[:80], "reason": "duplicate content (MD5)", "key": h})
            continue
        doc_id = str(r.get("doc_id") or "")
        if not doc_id or doc_id in used_ids:
            while f"D{next_num:05d}" in used_ids:
                next_num += 1
            doc_id = f"D{next_num:05d}"
        used_ids.add(doc_id)
        seen_hash.add(h)
        for key in (r.get("doc_id"), url, url_n):
            if key:
                id_map[str(key)] = doc_id
        if url_n:
            seen_urls.add(url_n)
        new_docs.append({"doc_id": doc_id, "title": clean(r.get("title")) or text[:60], "text": text})
        new_meta.append({"doc_id": doc_id, "url": url, "source": source, "source_type": source_type,
                         "category": clean(r.get("category")), "depth": int(r.get("depth") or 0),
                         "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "n_words": len(text.split()), "content_hash": h, "simhash": str(simhash(text))})
    # resolve hyperlinks (targets may be doc ids or URLs) into edges between documents
    url_to_id = {normalize_url(u): d for u, d in zip(c["meta"]["url"], c["meta"]["doc_id"]) if u}
    edges = []
    for r in rows:
        src = id_map.get(str(r.get("doc_id") or "")) or id_map.get(str(r.get("url") or ""))
        for dst in r.get("links") or []:
            tgt = id_map.get(dst) or url_to_id.get(normalize_url(dst)) or id_map.get(normalize_url(dst))
            if tgt is None and dst in used_ids:
                tgt = dst
            if src and tgt and src != tgt:
                edges.append({"src": src, "dst": tgt})
    c["docs"] = pd.concat([c["docs"], pd.DataFrame(new_docs, columns=DOC_COLS)], ignore_index=True)
    c["meta"] = pd.concat([c["meta"], pd.DataFrame(new_meta, columns=META_COLS)], ignore_index=True)
    c["edges"] = pd.concat([c["edges"], pd.DataFrame(edges, columns=["src", "dst"])]).drop_duplicates().reset_index(drop=True)
    report = {"added": len(new_docs), "dup_url": sum(s["reason"] == "duplicate URL" for s in skipped),
              "dup_content": sum(s["reason"].startswith("duplicate content") for s in skipped),
              "links": len(edges), "skipped": pd.DataFrame(skipped)}
    st.session_state.ingest_log.append({"time": datetime.now().strftime("%H:%M:%S"), "source": source,
                                        "type": source_type, "received": len(rows), "added": report["added"],
                                        "dup_url": report["dup_url"], "dup_content": report["dup_content"]})
    corpus_changed()
    return report


def save_corpus():
    """Content and metadata are stored in SEPARATE files, joined by doc_id."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    c = st.session_state.corpus
    with open(STORE_DIR / "documents.jsonl", "w", encoding="utf-8") as f:
        for r in c["docs"].to_dict("records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c["meta"].to_csv(STORE_DIR / "metadata.csv", index=False)
    c["edges"].to_csv(STORE_DIR / "links.csv", index=False)
    if "qrels" in st.session_state and not st.session_state.qrels.empty:
        st.session_state.qrels.to_csv(STORE_DIR / "qrels.csv", index=False)
    if "ratings" in st.session_state and not st.session_state.ratings.empty:
        st.session_state.ratings.to_csv(STORE_DIR / "ratings.csv", index=False)


def load_corpus():
    if not (STORE_DIR / "documents.jsonl").exists() or (STORE_DIR / "documents.jsonl").stat().st_size == 0:
        if (SAMPLE_DIR / "documents.csv").exists():
            df = pd.read_csv(SAMPLE_DIR / "documents.csv", dtype={"doc_id": str})
            rows = [{**r, "links": str(r["links"]).split(";") if isinstance(r["links"], str) else []}
                    for r in df.to_dict("records")]
            add_documents(rows, "sample collection", "bundled dataset", dedup=True)
            if (SAMPLE_DIR / "qrels.csv").exists():
                st.session_state.qrels = pd.read_csv(SAMPLE_DIR / "qrels.csv", dtype={"doc_id": str})
            if (SAMPLE_DIR / "ratings.csv").exists():
                st.session_state.ratings = pd.read_csv(SAMPLE_DIR / "ratings.csv", dtype={"doc_id": str, "user_id": str})
            save_corpus()
        return
    docs = pd.read_json(STORE_DIR / "documents.jsonl", lines=True, dtype={"doc_id": str})
    meta = pd.read_csv(STORE_DIR / "metadata.csv", dtype={"doc_id": str, "simhash": str}).fillna({"url": "", "category": ""})
    edges = pd.read_csv(STORE_DIR / "links.csv", dtype=str)
    st.session_state.corpus = {"docs": docs[DOC_COLS], "meta": meta[META_COLS], "edges": edges}
    st.session_state.corpus_version += 1
    if (STORE_DIR / "qrels.csv").exists():
        st.session_state.qrels = pd.read_csv(STORE_DIR / "qrels.csv", dtype={"doc_id": str})
    elif (SAMPLE_DIR / "qrels.csv").exists():
        st.session_state.qrels = pd.read_csv(SAMPLE_DIR / "qrels.csv", dtype={"doc_id": str})
    if (STORE_DIR / "ratings.csv").exists():
        st.session_state.ratings = pd.read_csv(STORE_DIR / "ratings.csv", dtype={"doc_id": str, "user_id": str})
    elif (SAMPLE_DIR / "ratings.csv").exists():
        st.session_state.ratings = pd.read_csv(SAMPLE_DIR / "ratings.csv", dtype={"doc_id": str, "user_id": str})


def parse_html(html, base_url):
    """Return (title, main text, absolute links) of an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else base_url
    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["h1", "h2", "h3", "p", "li"])]
    text = " ".join(b for b in blocks if len(b.split()) > 3) or soup.get_text(" ", strip=True)
    links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)
             if not a["href"].startswith(("mailto:", "javascript:", "tel:", "#"))]
    return title, re.sub(r"\s+", " ", text).strip(), [u for u in links if urlparse(u).scheme in ("http", "https")]


SKIP_EXT = re.compile(r"\.(jpg|jpeg|png|gif|svg|pdf|zip|mp4|mp3|css|js|ico|xml)$", re.I)


def crawl(seeds, max_depth=1, max_pages=25, same_domain=True, delay=0.3, respect_robots=True,
          min_words=50, url_regex="", progress=None):
    """Breadth-first crawler: multiple seeds, depth limit, robots.txt, duplicate URL/content checks."""
    seeds = [s.strip() for s in seeds if s.strip()]
    seed_domains = {urlparse(s).netloc.lower().removeprefix("www.") for s in seeds}
    url_filter = re.compile(url_regex) if url_regex else None
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    robots = {}
    frontier = deque((s, 0) for s in seeds)  # FIFO queue -> breadth-first order
    visited, seen_content, pages, log = set(), set(), [], []

    def allowed(url):
        if not respect_robots:
            return True
        root = "{0.scheme}://{0.netloc}".format(urlparse(url))
        if root not in robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = session.get(root + "/robots.txt", timeout=8)
                rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            except requests.RequestException:
                rp = None
            robots[root] = rp
        return robots[root] is None or robots[root].can_fetch(USER_AGENT, url)

    while frontier and len(pages) < max_pages:
        url, depth = frontier.popleft()
        norm = normalize_url(url)
        if norm in visited:
            log.append({"url": url, "depth": depth, "status": "skipped: duplicate URL"})
            continue
        visited.add(norm)
        if SKIP_EXT.search(urlparse(url).path):
            continue
        if not allowed(url):
            log.append({"url": url, "depth": depth, "status": "blocked by robots.txt"})
            continue
        t0 = time.perf_counter()
        try:
            resp = session.get(url, timeout=8)
        except requests.RequestException as exc:
            log.append({"url": url, "depth": depth, "status": f"error: {type(exc).__name__}"})
            continue
        ms = round((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200 or "html" not in resp.headers.get("Content-Type", ""):
            log.append({"url": url, "depth": depth, "status": f"HTTP {resp.status_code}", "ms": ms})
            continue
        title, text, links = parse_html(resp.text, resp.url)
        h = content_hash(text)
        if h in seen_content:
            log.append({"url": url, "depth": depth, "status": "skipped: duplicate content", "ms": ms})
            continue
        seen_content.add(h)
        out_links = []
        for link in links:
            if same_domain and urlparse(link).netloc.lower().removeprefix("www.") not in seed_domains:
                continue
            if (url_filter and not url_filter.search(link)) or SKIP_EXT.search(urlparse(link).path):
                continue
            out_links.append(normalize_url(link))
        out_links = list(dict.fromkeys(out_links))
        if len(text.split()) >= min_words:
            pages.append({"title": title, "text": text, "url": resp.url, "depth": depth,
                          "links": out_links, "category": urlparse(resp.url).netloc})
            status = "indexed"
        else:
            status = "fetched (too short, not indexed)"
        log.append({"url": url, "depth": depth, "status": status, "ms": ms, "outlinks": len(out_links)})
        if progress:
            progress(len(pages), max_pages, url)
        if depth < max_depth:
            frontier.extend((link, depth + 1) for link in out_links if link not in visited)
        time.sleep(delay)
    return pages, pd.DataFrame(log)


def fetch_wikipedia(topics, per_topic=5):
    """Public API source: MediaWiki search + plain-text extracts. Each topic becomes a category."""
    api = "https://en.wikipedia.org/w/api.php"
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    def call(params):
        time.sleep(0.2)  # be polite: Wikipedia rate-limits clients that send requests too fast
        resp = sess.get(api, params={**params, "action": "query", "format": "json"}, timeout=20)
        if resp.status_code == 429:
            raise requests.HTTPError("HTTP 429: Wikipedia says too many requests; wait a minute and try again")
        resp.raise_for_status()
        return resp.json()

    rows = []
    for topic in [t.strip() for t in topics if t.strip()]:
        hits = call({"list": "search", "srsearch": topic, "srlimit": per_topic})
        for hit in hits.get("query", {}).get("search", []):
            res = call({"prop": "extracts|links|info", "explaintext": 1, "pllimit": "max", "inprop": "url",
                        "titles": hit["title"]})
            for page in res.get("query", {}).get("pages", {}).values():
                if page.get("extract"):
                    links = [f"https://en.wikipedia.org/wiki/{l['title'].replace(' ', '_')}" for l in page.get("links", [])]
                    rows.append({"title": page["title"], "text": page["extract"][:20000],
                                 "url": page.get("fullurl", ""), "category": topic, "links": links})
    return rows


# =============================================================================
# 3. INVERTED INDEX AND RETRIEVAL MODELS
# =============================================================================
MODELS = ["BM25", "TF-IDF cosine", "BM25 + PageRank", "BM25 + PRF", "Boolean"]


def minmax(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class InvertedIndex:
    """Positional inverted index:  term -> {doc number: [positions]}  plus TF-IDF matrix."""

    def __init__(self, prep=None, title_boost=2, k1=1.5, b=0.75):
        self.prep = prep or dict(stop=True, stem="porter", min_len=2, drop_numbers=False)
        self.title_boost, self.k1, self.b = title_boost, k1, b

    def tok(self, text):
        return tokenize(text, **self.prep)

    def build(self, docs):
        t0 = time.perf_counter()
        self.doc_ids = docs["doc_id"].astype(str).tolist()
        self.titles = docs["title"].astype(str).tolist()
        self.texts = docs["text"].astype(str).tolist()
        n = len(self.doc_ids)
        postings = defaultdict(dict)
        self.doc_len = np.zeros(n)
        self.surface_vocab = Counter()  # unstemmed words, used for spelling correction
        doc_tfs = []
        for i, (title, text) in enumerate(zip(self.titles, self.texts)):
            body = (title + " ") * self.title_boost + text  # title boost = repeat the title
            toks = self.tok(body)
            self.surface_vocab.update(w for w in TOKEN_RE.findall(body.lower())
                                      if w not in STOP_WORDS and len(w) > 2 and not w.isdigit())
            self.doc_len[i] = len(toks)
            for pos, t in enumerate(toks):
                postings[t].setdefault(i, []).append(pos)
            doc_tfs.append(Counter(toks))
        self.postings = dict(postings)
        self.vocab = sorted(self.postings)  # sorted -> binary search for wildcards
        self.term_id = {t: j for j, t in enumerate(self.vocab)}
        self.avgdl = self.doc_len.mean() if n else 1.0
        # TF-IDF matrix: (1 + log tf) * idf, rows L2-normalised -> dot product = cosine similarity
        self.idf = {t: math.log((1 + n) / (1 + len(p))) + 1 for t, p in self.postings.items()}
        rows, cols, vals = [], [], []
        for i, tf in enumerate(doc_tfs):
            for t, c in tf.items():
                rows.append(i)
                cols.append(self.term_id[t])
                vals.append((1 + math.log(c)) * self.idf[t])
        m = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)))
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1
        self.tfidf = sparse.diags(1 / norms) @ m
        self.n_postings = sum(len(p) for p in self.postings.values())
        self.build_ms = (time.perf_counter() - t0) * 1000
        self.prior = None
        return self

    # ------------------------------------------------------------ query tools
    def expand_wildcard(self, pattern, limit=30):
        """'comput*' -> binary search in the sorted vocabulary; other patterns -> regex scan."""
        pat = pattern.lower()
        if pat.endswith("*") and "*" not in pat[:-1]:
            out = []
            for prefix in dict.fromkeys([pat[:-1]] + self.tok(pat[:-1])):
                lo = np.searchsorted(self.vocab, prefix)
                for t in self.vocab[lo:]:
                    if not t.startswith(prefix) or len(out) >= limit:
                        break
                    out.append(t)
            return list(dict.fromkeys(out))
        rx = re.compile("^" + re.escape(pat).replace(r"\*", ".*") + "$")
        return [t for t in self.vocab if rx.match(t)][:limit]

    def spell_correct(self, query):
        """Replace words that are not in the index by the closest known word (edit-distance ratio)."""
        import difflib
        fixes, out = {}, []
        vocab = list(self.surface_vocab)
        for w in re.findall(r"\"|\w+\*?|\*\w+|[()]", query):
            lw = w.lower()
            known = self.tok(lw)
            if (w in '"()' or "*" in w or lw in ("and", "or", "not") or lw in STOP_WORDS or lw.isdigit()
                    or (known and known[0] in self.postings)):
                out.append(w)
                continue
            match = difflib.get_close_matches(lw, vocab, n=1, cutoff=0.75)
            if match:
                fixes[w] = match[0]
            out.append(match[0] if match else w)
        return " ".join(out).replace('" ', '"'), fixes

    def query_terms(self, query):
        terms = []
        for raw in re.findall(r"\w+\*?|\*\w+", query):
            if raw.lower() not in ("and", "or", "not"):
                terms += self.expand_wildcard(raw) if "*" in raw else self.tok(raw)
        return terms

    # ---------------------------------------------------------------- scoring
    def bm25(self, terms, weights=None):
        """Okapi BM25:  sum over query terms of  idf * tf*(k1+1) / (tf + k1*(1 - b + b*len/avglen))."""
        n = len(self.doc_ids)
        scores = np.zeros(n)
        for t, qf in Counter(terms).items():
            if t not in self.postings:
                continue
            docs = np.fromiter(self.postings[t].keys(), dtype=int)
            tf = np.array([len(p) for p in self.postings[t].values()], dtype=float)
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[docs] / self.avgdl)
            scores[docs] += (weights or {}).get(t, 1.0) * qf * idf * tf * (self.k1 + 1) / norm
        return scores

    def tfidf_cosine(self, terms):
        q = np.zeros(len(self.vocab))
        for t, c in Counter(t for t in terms if t in self.term_id).items():
            q[self.term_id[t]] = (1 + math.log(c)) * self.idf[t]
        if not q.any():
            return np.zeros(len(self.doc_ids))
        return np.asarray(self.tfidf @ (q / np.linalg.norm(q))).ravel()

    def phrase_docs(self, phrase):
        """Documents where the phrase terms occur at consecutive positions (positional index)."""
        toks = self.tok(phrase)
        if not toks or any(t not in self.postings for t in toks):
            return set()
        candidates = set.intersection(*(set(self.postings[t]) for t in toks))
        hits = set()
        for d in candidates:
            starts = set(self.postings[toks[0]][d])
            for offset, t in enumerate(toks[1:], 1):
                starts &= {p - offset for p in self.postings[t][d]}
            if starts:
                hits.add(d)
        return hits

    def boolean_docs(self, query):
        """AND / OR / NOT with parentheses; adjacent words = implicit AND (shunting-yard parser)."""
        tokens = re.findall(r"\"[^\"]+\"|\(|\)|\w+\*?|\*\w+", query)
        seq = []
        for tok in tokens:
            up = tok.upper()
            if seq and seq[-1] not in ("AND", "OR", "NOT", "(") and up not in ("AND", "OR", ")"):
                seq.append("AND")
            seq.append(up if up in ("AND", "OR", "NOT") else tok)
        prec, out, ops = {"NOT": 3, "AND": 2, "OR": 1}, [], []
        for tok in seq:  # infix -> postfix
            if tok in prec:
                while ops and ops[-1] in prec and (prec[ops[-1]] > prec[tok] or (prec[ops[-1]] == prec[tok] and tok != "NOT")):
                    out.append(ops.pop())
                ops.append(tok)
            elif tok == "(":
                ops.append(tok)
            elif tok == ")":
                while ops and ops[-1] != "(":
                    out.append(ops.pop())
                if ops:
                    ops.pop()
            else:
                out.append(tok)
        out += [op for op in reversed(ops) if op != "("]
        everything, stack = set(range(len(self.doc_ids))), []
        for tok in out:  # evaluate postfix with sets of doc numbers
            if tok == "NOT":
                stack.append(everything - (stack.pop() if stack else set()))
            elif tok in ("AND", "OR"):
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a & b if tok == "AND" else a | b)
            elif tok.startswith('"'):
                stack.append(self.phrase_docs(tok.strip('"')))
            else:
                terms = self.expand_wildcard(tok) if "*" in tok else self.tok(tok)
                stack.append(set().union(*(set(self.postings.get(t, {})) for t in terms)) if terms else set())
        return stack[-1] if stack else set()

    def prf_expand(self, terms, k_docs=5, n_terms=5, beta=0.5):
        """Pseudo-relevance feedback (Rocchio): add top TF-IDF terms of the first k BM25 results."""
        base = self.bm25(terms)
        top = [d for d in np.argsort(-base)[:k_docs] if base[d] > 0]
        if not top:
            return terms, {}
        centroid = np.asarray(self.tfidf[top].mean(axis=0)).ravel()
        new = [self.vocab[j] for j in np.argsort(-centroid) if self.vocab[j] not in terms][:n_terms]
        return terms + new, {**{t: 1.0 for t in terms}, **{t: beta for t in new}}

    def search(self, query, model="BM25", top_k=10, pr_weight=0.3, prf_docs=5, prf_terms=5):
        t0 = time.perf_counter()
        n = len(self.doc_ids)
        phrases = re.findall(r"\"([^\"]+)\"", query)
        terms = self.query_terms(query.replace('"', " "))
        info = {"terms": terms, "expanded": []}
        if model == "Boolean":
            mask = np.zeros(n, bool)
            mask[list(self.boolean_docs(query))] = True
            text_scores = (self.bm25(terms) + 1e-6) * mask  # matched set, ordered by BM25
        else:
            mask = np.ones(n, bool)
            for ph in phrases:  # phrase queries filter the candidates
                allowed = np.zeros(n, bool)
                allowed[list(self.phrase_docs(ph))] = True
                mask &= allowed
            if model == "TF-IDF cosine":
                text_scores = self.tfidf_cosine(terms)
            elif model == "BM25 + PRF":
                exp, weights = self.prf_expand(terms, prf_docs, prf_terms)
                info["expanded"] = exp[len(terms):]
                text_scores = self.bm25(exp, weights)
            else:
                text_scores = self.bm25(terms)
            text_scores = text_scores * mask
        pr = self.prior if self.prior is not None else np.zeros(n)
        final = text_scores.copy()
        if model == "BM25 + PageRank":  # (1-w)*BM25 + w*PageRank, both scaled to [0, 1]
            matched = final > 0
            final = np.where(matched, (1 - pr_weight) * minmax(final) + pr_weight * minmax(pr), 0)
        cand = np.flatnonzero(final > 0)
        if len(cand) > top_k:  # partial sort: only the top-k need ordering
            cand = cand[np.argpartition(-final[cand], top_k - 1)[:top_k]]
        order = cand[np.argsort(-final[cand])]
        info["n_matched"] = int((final > 0).sum())
        results = pd.DataFrame({
            "rank": range(1, len(order) + 1), "doc_id": [self.doc_ids[i] for i in order],
            "score": final[order].round(4), "text_score": text_scores[order].round(4),
            "pagerank": pr[order].round(5), "title": [self.titles[i] for i in order],
            "snippet": [self.snippet(self.texts[i], terms) for i in order]})
        return results, info, (time.perf_counter() - t0) * 1000

    def snippet(self, text, terms, n_words=40):
        words = text.split()
        stems = set(terms)
        hit = lambda w: (lambda t: bool(t) and t[0] in stems)(self.tok(w))
        start = next((max(0, i - 8) for i, w in enumerate(words) if hit(w)), 0)
        part = " ".join(f"**{w}**" if hit(w) else w for w in words[start:start + n_words])
        return ("..." if start else "") + part + ("..." if start + n_words < len(words) else "")


# =============================================================================
# 4. LINK ANALYSIS: PAGERANK AND HITS
# =============================================================================
def adjacency(doc_ids, edges):
    pos = {d: i for i, d in enumerate(doc_ids)}
    pairs = {(pos[s], pos[d]) for s, d in zip(edges["src"], edges["dst"]) if s in pos and d in pos and s != d}
    rows, cols = zip(*pairs) if pairs else ((), ())
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(doc_ids), len(doc_ids)))


def similarity_graph(tfidf, k=5, threshold=0.1):
    """For collections without hyperlinks: link each doc to its k most similar docs."""
    sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(sim, 0)
    rows, cols = [], []
    for i in range(sim.shape[0]):
        for j in np.argsort(-sim[i])[:k]:
            if sim[i, j] >= threshold:
                rows.append(i)
                cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=sim.shape)


def pagerank(adj, d=0.85, tol=1e-8, max_iter=200):
    """Power iteration:  PR = d * (M^T PR + dangling/N) + (1-d)/N."""
    n = adj.shape[0]
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = out_deg == 0  # pages with no out-links share their rank with everyone
    m_t = (sparse.diags(np.divide(1.0, out_deg, out=np.zeros(n), where=out_deg > 0)) @ adj).T
    r, history = np.full(n, 1 / n), []
    for it in range(max_iter):
        new = d * (m_t @ r + r[dangling].sum() / n) + (1 - d) / n
        change = np.abs(new - r).sum()
        history.append({"iteration": it + 1, "l1_change": change})
        r = new
        if change < tol:
            break
    return r / r.sum(), history


def hits(adj, tol=1e-8, max_iter=200):
    """Kleinberg HITS: authority = sum of hub scores pointing in; hub = sum of authorities pointed to."""
    n = adj.shape[0]
    hub, auth, history = np.ones(n), np.ones(n), []
    for it in range(max_iter):
        a = adj.T @ hub
        a /= np.linalg.norm(a) or 1
        h = adj @ a
        h /= np.linalg.norm(h) or 1
        change = np.abs(a - auth).sum() + np.abs(h - hub).sum()
        history.append({"iteration": it + 1, "l1_change": change})
        auth, hub = a, h
        if change < tol:
            break
    return hub, auth, history


# =============================================================================
# 5. TEXT MINING
# =============================================================================
def with_bigrams(analyzer):
    return lambda doc: (lambda t: t + [f"{a}_{b}" for a, b in zip(t, t[1:])])(analyzer(doc))


FEATURES = {
    "Bag of words (counts)": lambda an: CountVectorizer(analyzer=an),
    "Binary BoW": lambda an: CountVectorizer(analyzer=an, binary=True),
    "TF-IDF (unigrams)": lambda an: TfidfVectorizer(analyzer=an, sublinear_tf=True),
    "TF-IDF (uni+bigrams)": lambda an: TfidfVectorizer(analyzer=with_bigrams(an), sublinear_tf=True),
}
CLASSIFIERS = {
    "Multinomial NB": lambda: MultinomialNB(alpha=0.3),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, C=5),
    "Linear SVM": lambda: LinearSVC(C=1.0),
}


def compare_classifiers(texts, labels, presets, features, classifiers, folds=5, progress=None):
    """Cross-validated accuracy / macro-F1 for every preprocessing x feature x classifier combination."""
    y = np.array(labels)
    folds = max(2, min(folds, min(Counter(y).values())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows, preds, step = [], {}, 0
    total = len(presets) * len(features) * len(classifiers)
    for p in presets:
        analyzer = lambda d, p=p: tokenize(d, **PRESETS[p])
        for f in features:
            x = FEATURES[f](analyzer).fit_transform(texts)
            for c in classifiers:
                yp = cross_val_predict(CLASSIFIERS[c](), x, y, cv=cv)
                rows.append({"preprocessing": p, "features": f, "classifier": c, "n_features": x.shape[1],
                             "accuracy": accuracy_score(y, yp), "macro_F1": f1_score(y, yp, average="macro")})
                preds[(p, f, c)] = yp
                step += 1
                if progress:
                    progress(step / total)
    return pd.DataFrame(rows), y, preds, folds


def doc_keywords(index, i, k=8):
    row = index.tfidf.getrow(i)
    order = np.argsort(-row.data)[:k]
    return [(index.vocab[row.indices[j]], float(row.data[j])) for j in order]


def profile(text, title, index, keywords):
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words)
    toks = index.tok(text)
    n_w, n_s = max(1, len(words)), max(1, len(sents))
    return {"Title": title, "Words": len(words), "Sentences": len(sents),
            "Avg sentence length": round(n_w / n_s, 1), "Unique terms": len(set(toks)),
            "Type-token ratio": round(len(set(toks)) / max(1, len(toks)), 3),
            "Flesch reading ease": round(206.835 - 1.015 * n_w / n_s - 84.6 * syllables / n_w, 1),
            "Top keywords": ", ".join(t for t, _ in keywords)}


# =============================================================================
# 6. RECOMMENDER SYSTEM
# =============================================================================
def rating_matrix(ratings, doc_ids):
    pos = {d: i for i, d in enumerate(doc_ids)}
    r = ratings[ratings["doc_id"].astype(str).isin(pos)]
    users = sorted(r["user_id"].astype(str).unique())
    upos = {u: i for i, u in enumerate(users)}
    m = sparse.csr_matrix((r["rating"].astype(float), (r["user_id"].astype(str).map(upos),
                                                       r["doc_id"].astype(str).map(pos))),
                          shape=(len(users), len(doc_ids)))
    return m, users


def item_similarity(m):
    """Adjusted cosine between item columns, after subtracting each user's mean rating."""
    m = m.tocsr().astype(float)
    counts = np.diff(m.indptr)
    means = np.divide(np.asarray(m.sum(axis=1)).ravel(), counts, out=np.zeros(m.shape[0]), where=counts > 0)
    c = m.copy()
    c.data -= np.repeat(means, counts)
    c.data[c.data == 0] = 1e-6  # keep a rating equal to the mean as an explicit entry
    norms = np.sqrt(np.asarray(c.multiply(c).sum(axis=0)).ravel())
    norms[norms == 0] = 1
    sim = (c.T @ c).toarray() / np.outer(norms, norms)
    np.fill_diagonal(sim, 0)
    return sim


def content_profile_scores(tfidf, liked, exclude=()):
    """Cosine of every document with the centroid of the user's liked documents."""
    if not liked:
        return np.zeros(tfidf.shape[0])
    centroid = np.asarray(tfidf[liked].mean(axis=0)).ravel()
    scores = np.asarray(tfidf @ (centroid / (np.linalg.norm(centroid) or 1))).ravel()
    scores[list(liked) + list(exclude)] = -np.inf
    return scores


def collaborative_scores(m, sim, u):
    """Predicted rating = similarity-weighted average of the user's own ratings."""
    row = m[u].toarray().ravel()
    rated = np.flatnonzero(row)
    if not len(rated):
        return np.zeros(m.shape[1])
    s = sim[:, rated]
    den = np.abs(s).sum(axis=1)
    scores = np.divide(s @ row[rated], den, out=np.zeros(m.shape[1]), where=den > 0)
    scores[rated] = -np.inf
    return scores


def scale01(x):
    ok = np.isfinite(x)
    out = np.full(len(x), -np.inf)
    if ok.any():
        lo, hi = x[ok].min(), x[ok].max()
        out[ok] = 0 if hi - lo < 1e-12 else (x[ok] - lo) / (hi - lo)
    return out


def hybrid(content, collab, alpha):
    """alpha * content + (1 - alpha) * collaborative, each min-max scaled."""
    c, f = scale01(content), scale01(collab)
    excluded = ~np.isfinite(c) if alpha == 1 else ~np.isfinite(f) if alpha == 0 else ~np.isfinite(c) | ~np.isfinite(f)
    out = alpha * np.nan_to_num(c, neginf=0) + (1 - alpha) * np.nan_to_num(f, neginf=0)
    out[excluded] = -np.inf
    return out


def leave_one_out(m, tfidf, k=5, alphas=(0.0, 0.5, 1.0), seed=0):
    """Hide one highly rated (>=4) item per user and check if each method ranks it in the top-K."""
    rng = np.random.default_rng(seed)
    m = m.tolil()
    held = {}
    for u in range(m.shape[0]):
        good = [j for j, v in zip(m.rows[u], m.data[u]) if v >= 4]
        if len(good) >= 3:
            held[u] = int(rng.choice(good))
            m[u, held[u]] = 0
    m = m.tocsr()
    m.eliminate_zeros()
    sim = item_similarity(m)
    res = {a: [0, 0.0] for a in alphas}
    for u, target in held.items():
        row = m[u].toarray().ravel()
        liked = list(np.flatnonzero(row >= 4)) or list(np.flatnonzero(row))
        cb = content_profile_scores(tfidf, liked, exclude=np.flatnonzero(row))
        cf = collaborative_scores(m, sim, u)
        for a in alphas:
            top = list(np.argsort(-hybrid(cb, cf, a))[:k])
            if target in top:
                res[a][0] += 1
                res[a][1] += 1 / (top.index(target) + 1)
    n = max(1, len(held))
    name = {0.0: "Collaborative", 1.0: "Content-based"}
    return pd.DataFrame([{"method": name.get(a, "Hybrid"), "alpha": a, f"HitRate@{k}": h / n,
                          f"MRR@{k}": rr / n, "test_users": len(held)} for a, (h, rr) in res.items()])


def simulate_ratings(doc_ids, categories, n_users=40, per_user=12, seed=42):
    """Synthetic ratings when none are uploaded: each user prefers 1-2 categories."""
    rng = np.random.default_rng(seed)
    cats = pd.Series(categories).replace("", "general")
    groups = {c: list(idx) for c, idx in cats.groupby(cats).groups.items()}
    rows = []
    for u in range(n_users):
        favs = rng.choice(list(groups), size=min(len(groups), int(rng.integers(1, 3))), replace=False)
        pool = [i for c in favs for i in groups[c]]
        for _ in range(per_user):
            fav = rng.random() < 0.8
            d = int(rng.choice(pool)) if fav else int(rng.integers(0, len(doc_ids)))
            rows.append({"user_id": f"U{u + 1:03d}", "doc_id": doc_ids[d],
                         "rating": int(rng.integers(4, 6)) if fav else int(rng.integers(1, 4))})
    return pd.DataFrame(rows).groupby(["user_id", "doc_id"], as_index=False)["rating"].max()


# =============================================================================
# 7. EVALUATION METRICS
# =============================================================================
def precision(ret, rel):
    return sum(d in rel for d in ret) / len(ret) if ret else 0.0


def precision_at_k(ret, rel, k):
    return sum(d in rel for d in ret[:k]) / k if k > 0 else 0.0


def recall(ret, rel):
    return sum(d in rel for d in ret) / len(rel) if rel else 0.0


def average_precision(ret, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ret, 1):
        if d in rel:
            hits += 1
            total += hits / i  # precision at each relevant document
    return total / len(rel) if rel else 0.0


def reciprocal_rank(ret, rel):
    return next((1 / i for i, d in enumerate(ret, 1) if d in rel), 0.0)


def ndcg(ret, grades, k):
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ret[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def evaluate_run(ret, grades, k):
    rel = {d for d, g in grades.items() if g > 0}
    p, r = precision(ret, rel), recall(ret, rel)
    return {"Precision": p, "Recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
            f"P@{k}": precision_at_k(ret, rel, k), f"R@{k}": recall(ret[:k], rel),
            "AP": average_precision(ret, rel), "RR": reciprocal_rank(ret, rel), f"NDCG@{k}": ndcg(ret, grades, k)}


def pr_curve(ret, rel):
    """11-point interpolated precision-recall curve."""
    pts, hits = [], 0
    for i, d in enumerate(ret, 1):
        hits += d in rel
        pts.append((hits / len(rel) if rel else 0, hits / i))
    return pd.DataFrame([{"recall": round(lvl, 1), "precision": max([p for r, p in pts if r >= lvl - 1e-9], default=0.0)}
                         for lvl in np.linspace(0, 1, 11)])


def auto_judgements(index, meta, n_terms=3):
    """Pseudo test collection from category labels: query = most distinctive terms of a category,
    relevant = documents of that category (grade 2 if they contain all query terms)."""
    cats = meta.set_index("doc_id").reindex(index.doc_ids)["category"].fillna("").values
    rows = []
    for c in [c for c in pd.unique(cats) if c]:
        inside, outside = np.flatnonzero(cats == c), np.flatnonzero(cats != c)
        if len(inside) < 2:
            continue
        diff = np.asarray(index.tfidf[inside].mean(axis=0)).ravel()
        if len(outside):
            diff -= np.asarray(index.tfidf[outside].mean(axis=0)).ravel()
        terms = [index.vocab[j] for j in np.argsort(-diff)[:n_terms]]
        words = [next((w for w, _ in index.surface_vocab.most_common() if w.startswith(t[:max(3, len(t) - 1)])), t) for t in terms]
        for d in inside:
            rows.append({"query": " ".join(words), "doc_id": index.doc_ids[d],
                         "relevance": 2 if all(d in index.postings.get(t, {}) for t in terms) else 1})
    return pd.DataFrame(rows, columns=["query", "doc_id", "relevance"])


# =============================================================================
# 8. SESSION STATE HELPERS
# =============================================================================
def init_state():
    defaults = {"corpus": empty_corpus(), "corpus_version": 0, "index": None, "index_version": -1,
                "index_cfg": dict(stop=True, stem="porter", min_len=2, drop_numbers=False, title_boost=2, k1=1.5, b=0.75),
                "links": {}, "search_log": [], "crawl_log": pd.DataFrame(), "ingest_log": [],
                "qrels": pd.DataFrame(columns=["query", "doc_id", "relevance"]),
                "ratings": pd.DataFrame(columns=["user_id", "doc_id", "rating"]),
                "eval": None, "mining": None, "rec_eval": None, "bench": None, "last_search": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "loaded" not in st.session_state:
        st.session_state.loaded = True
        load_corpus()


def corpus_changed():
    st.session_state.corpus_version += 1
    st.session_state.links = {}
    st.session_state.eval = None
    save_corpus()


def n_docs():
    return len(st.session_state.corpus["docs"])


def get_index(force=False):
    """(Re)build the index when the corpus or the settings changed."""
    ss = st.session_state
    if n_docs() == 0:
        return None
    if force or ss.index is None or ss.index_version != ss.corpus_version:
        cfg = ss.index_cfg
        prep = {k: cfg[k] for k in ("stop", "stem", "min_len", "drop_numbers")}
        with st.spinner("Building inverted index..."):
            ss.index = InvertedIndex(prep, cfg["title_boost"], cfg["k1"], cfg["b"]).build(ss.corpus["docs"])
        ss.index_version = ss.corpus_version
        ss.links = {}
    return ss.index


def get_links(graph="auto", damping=0.85, knn=5):
    """PageRank + HITS for the current corpus, cached per settings."""
    idx = get_index()
    key = (st.session_state.corpus_version, graph, damping, knn)
    if key not in st.session_state.links:
        adj = adjacency(idx.doc_ids, st.session_state.corpus["edges"])
        used = "hyperlinks"
        if graph == "similarity" or (graph == "auto" and adj.nnz < max(5, len(idx.doc_ids) // 4)):
            adj, used = similarity_graph(idx.tfidf, knn), f"content similarity (k={knn})"
        t0 = time.perf_counter()
        pr, pr_hist = pagerank(adj, damping)
        hub, auth, h_hist = hits(adj)
        st.session_state.links[key] = {"adj": adj, "pagerank": pr, "hub": hub, "authority": auth, "graph": used,
                                       "pr_history": pr_hist, "hits_history": h_hist, "ms": (time.perf_counter() - t0) * 1000}
    links = st.session_state.links[key]
    idx.prior = links["pagerank"]
    return links


def categories(doc_ids):
    return st.session_state.corpus["meta"].set_index("doc_id").reindex(doc_ids)["category"].fillna("").astype(str).values


def doc_label(doc_id):
    d = st.session_state.corpus["docs"]
    return f"{doc_id} - {d.loc[d['doc_id'] == doc_id, 'title'].iloc[0][:70]}"


def need_corpus():
    if n_docs() == 0:
        st.info("No documents yet. Open **Data acquisition** to load the sample collection, upload a dataset or crawl the web.")
        st.stop()


def show_ingest_report(rep):
    st.success(f"Added **{rep['added']}** documents - skipped {rep['dup_url']} duplicate URLs and "
               f"{rep['dup_content']} duplicate documents - {rep['links']} links resolved.")
    if len(rep["skipped"]):
        with st.expander("Skipped duplicates"):
            st.dataframe(rep["skipped"], hide_index=True)


# =============================================================================
# 9. PAGES
# =============================================================================
def page_dashboard():
    st.caption("Lifecycle: acquire -> preprocess & mine -> index -> search & rank -> recommend -> evaluate")
    need_corpus()
    c, idx = st.session_state.corpus, get_index()
    meta = c["meta"]
    cols = st.columns(6)
    cols[0].metric("Documents", len(c["docs"]))
    cols[1].metric("Sources", meta["source"].nunique())
    cols[2].metric("Categories", meta["category"].replace("", pd.NA).nunique())
    cols[3].metric("Vocabulary", f"{len(idx.vocab):,}")
    cols[4].metric("Hyperlinks", len(c["edges"]))
    cols[5].metric("Searches run", len(st.session_state.search_log))
    left, right = st.columns(2)
    left.markdown("**Documents by source**")
    left.altair_chart(alt.Chart(meta.groupby(["source", "source_type"]).size().reset_index(name="documents")).mark_bar().encode(
        x="documents:Q", y=alt.Y("source:N", sort="-x", title=None), color="source_type:N"))
    right.markdown("**Documents by category**")
    right.altair_chart(alt.Chart(meta.assign(category=meta["category"].replace("", "(none)")).groupby("category").size()
                                 .reset_index(name="documents")).mark_arc(innerRadius=60).encode(theta="documents:Q", color="category:N"))
    st.markdown("**Document length (words)**")
    st.altair_chart(alt.Chart(meta).mark_bar().encode(x=alt.X("n_words:Q", bin=alt.Bin(maxbins=30), title="Words"),
                                                      y=alt.Y("count()", title="Documents")).properties(height=220))
    if st.session_state.ingest_log:
        st.markdown("**Ingestion history**")
        st.dataframe(pd.DataFrame(st.session_state.ingest_log), hide_index=True)


def page_acquisition():
    st.caption("Every source goes through: URL normalisation -> duplicate-URL check -> MD5 content check -> "
               "separate document / metadata storage.")
    dedup = st.toggle("Drop exact-duplicate documents (MD5)", value=True)
    t_up, t_sample, t_crawl, t_api, t_judge = st.tabs(["Upload dataset", "Sample collection", "Web crawler",
                                                       "Wikipedia API", "Judgements & ratings"])
    with t_up:
        kind = st.radio("File type", ["CSV table", "Text / HTML files"], horizontal=True)
        if kind == "CSV table":
            up = st.file_uploader("Upload a CSV document collection", type=["csv"])
            if up is not None:
                df = pd.read_csv(up)
                st.dataframe(df.head(), hide_index=True)
                cols, none = list(df.columns), "(none)"
                guess = lambda *names: next((c for n in names for c in cols if c.lower() == n), none)
                opt = [none] + cols
                with st.form("map_cols"):
                    a, b, c3 = st.columns(3)
                    text_col = a.selectbox("Text column *", cols, index=cols.index(guess("text", "content", "body")) if guess("text", "content", "body") != none else 0)
                    title_col = b.selectbox("Title", opt, index=opt.index(guess("title", "headline")))
                    id_col = c3.selectbox("Document id", opt, index=opt.index(guess("doc_id", "id")))
                    a, b, c3 = st.columns(3)
                    cat_col = a.selectbox("Category", opt, index=opt.index(guess("category", "label", "topic")))
                    url_col = b.selectbox("URL", opt, index=opt.index(guess("url", "link")))
                    link_col = c3.selectbox("Out-links (; separated)", opt, index=opt.index(guess("links", "outlinks")))
                    go = st.form_submit_button("Ingest dataset", type="primary")
                if go:
                    get = lambda r, col: "" if col == none or pd.isna(r[col]) else str(r[col])
                    rows = [{"doc_id": get(r, id_col), "title": get(r, title_col), "text": get(r, text_col),
                             "category": get(r, cat_col), "url": get(r, url_col),
                             "links": [x for x in re.split(r"[;,|\s]+", get(r, link_col)) if x]} for _, r in df.iterrows()]
                    show_ingest_report(add_documents(rows, up.name, "dataset upload", dedup))
        else:
            files = st.file_uploader("Upload .txt / .html files", type=["txt", "html", "htm"], accept_multiple_files=True)
            if files and st.button("Ingest files", type="primary"):
                rows = []
                for f in files:
                    raw = f.getvalue().decode("utf-8", errors="ignore")
                    if f.name.lower().endswith((".html", ".htm")):
                        title, text, _ = parse_html(raw, "")
                    else:
                        title, text = f.name.rsplit(".", 1)[0], raw
                    rows.append({"title": title, "text": text, "url": f"file://{f.name}"})
                show_ingest_report(add_documents(rows, "uploaded files", "file upload", dedup))
    with t_sample:
        st.markdown("Bundled synthetic collection: **54 short articles** in 5 categories with hyperlinks, relevance judgements "
                    "for 12 queries and user ratings. Three planted rows test de-duplication: an exact copy under "
                    "another URL (MD5), the same URL written differently (`www.`, trailing `/`, `utm_` parameter) and a "
                    "near-duplicate (SimHash, see Index management -> Duplicates).")
        if st.button("Load sample collection", type="primary"):
            df = pd.read_csv(SAMPLE_DIR / "documents.csv", dtype={"doc_id": str})
            rows = [{**r, "links": str(r["links"]).split(";") if isinstance(r["links"], str) else []}
                    for r in df.to_dict("records")]
            show_ingest_report(add_documents(rows, "sample collection", "bundled dataset", dedup))
            st.session_state.qrels = pd.read_csv(SAMPLE_DIR / "qrels.csv", dtype={"doc_id": str})
            st.session_state.ratings = pd.read_csv(SAMPLE_DIR / "ratings.csv", dtype={"doc_id": str, "user_id": str})
            st.info("Relevance judgements and user ratings loaded too.")
    with t_crawl:
        with st.form("crawl"):
            seeds = st.text_area("Seed URLs (one per line)", "https://en.wikipedia.org/wiki/Information_retrieval\n"
                                                             "https://en.wikipedia.org/wiki/Web_crawler")
            a, b, c3 = st.columns(3)
            depth = a.slider("Crawl depth", 0, 4, 1, help="0 = only the seeds, 1 = seeds + the pages they link to, ...")
            max_pages = b.slider("Max pages", 5, 300, 25)
            delay = c3.slider("Politeness delay (s)", 0.0, 3.0, 0.3, 0.1)
            a, b, c3 = st.columns(3)
            same_domain = a.checkbox("Stay on seed domains", True)
            robots = b.checkbox("Respect robots.txt", True)
            min_words = c3.number_input("Min words to index", 0, 1000, 50)
            url_regex = st.text_input("URL regex filter (only follow matching links)", r"/wiki/[^:]+$")
            go = st.form_submit_button("Start crawl", type="primary")
        if go:
            bar = st.progress(0.0, "Starting crawler...")
            pages, log = crawl(seeds.splitlines(), depth, max_pages, same_domain, delay, robots, min_words, url_regex,
                               progress=lambda n, tot, url: bar.progress(min(1.0, n / tot), f"{n}/{tot} {url[:80]}"))
            bar.empty()
            st.session_state.crawl_log = log
            if pages:
                show_ingest_report(add_documents(pages, "crawl: " + seeds.splitlines()[0][:50], "web crawl", dedup))
            else:
                st.warning("No pages indexed. Check the seeds, network access and filters.")
        log = st.session_state.crawl_log
        if len(log):
            a, b, c3, d = st.columns(4)
            a.metric("URLs processed", len(log))
            b.metric("Indexed", int((log["status"] == "indexed").sum()))
            c3.metric("Duplicate URL / content", int(log["status"].str.startswith("skipped").sum()))
            d.metric("Max depth reached", int(log["depth"].max()))
            st.dataframe(log, hide_index=True)
    with t_api:
        with st.form("wiki"):
            topics = st.text_input("Topics (comma separated)", "information retrieval, pagerank, cricket, inflation")
            per = st.slider("Articles per topic", 1, 20, 3)
            go = st.form_submit_button("Fetch from Wikipedia", type="primary")
        if go:
            try:
                with st.spinner("Calling the Wikipedia API..."):
                    rows = fetch_wikipedia(topics.split(","), per)
                show_ingest_report(add_documents(rows, "Wikipedia API", "API", dedup))
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
    with t_judge:
        a, b = st.columns(2)
        q = a.file_uploader("Relevance judgements CSV: query, doc_id, relevance", type=["csv"])
        if q is not None and a.button("Load judgements"):
            st.session_state.qrels = pd.read_csv(q, dtype={"doc_id": str})
        a.caption(f"{len(st.session_state.qrels)} judgements loaded")
        r = b.file_uploader("User ratings CSV: user_id, doc_id, rating", type=["csv"])
        if r is not None and b.button("Load ratings"):
            st.session_state.ratings = pd.read_csv(r, dtype={"doc_id": str, "user_id": str})
        b.caption(f"{len(st.session_state.ratings)} ratings loaded")
    st.divider()
    st.caption(f"Corpus: {n_docs()} documents, stored in data/store/documents.jsonl (content) and "
               "data/store/metadata.csv (metadata).")
    if st.button("Reset corpus"):
        st.session_state.corpus = empty_corpus()
        st.session_state.qrels = st.session_state.qrels.iloc[0:0]
        st.session_state.ratings = st.session_state.ratings.iloc[0:0]
        corpus_changed()
        st.rerun()


def page_index():
    need_corpus()
    cfg = st.session_state.index_cfg
    with st.form("index_settings"):
        st.markdown("**Index configuration**")
        a, b, c, d = st.columns(4)
        stem = a.selectbox("Stemmer", ["porter", "snowball", "none"], index=["porter", "snowball", "none"].index(cfg["stem"]))
        stop = b.checkbox("Remove stop words", cfg["stop"])
        nums = c.checkbox("Remove numbers", cfg["drop_numbers"])
        min_len = d.number_input("Min token length", 1, 5, cfg["min_len"])
        a, b, c = st.columns(3)
        boost = a.slider("Title boost (title repeated n times)", 1, 5, cfg["title_boost"])
        k1 = b.slider("BM25 k1", 0.5, 3.0, cfg["k1"], 0.1)
        bb = c.slider("BM25 b", 0.0, 1.0, cfg["b"], 0.05)
        if st.form_submit_button("Rebuild index", type="primary"):
            st.session_state.index_cfg = dict(stop=stop, stem=stem, min_len=int(min_len), drop_numbers=nums,
                                              title_boost=boost, k1=k1, b=bb)
            get_index(force=True)
            st.session_state.eval = None
            st.toast("Index rebuilt")
    idx = get_index()
    a, b, c, d, e = st.columns(5)
    a.metric("Documents", len(idx.doc_ids))
    b.metric("Unique terms", f"{len(idx.vocab):,}")
    c.metric("Postings", f"{idx.n_postings:,}")
    d.metric("Avg doc length", f"{idx.avgdl:.1f} tokens")
    e.metric("Build time", f"{idx.build_ms:.0f} ms")
    t_term, t_vocab, t_store, t_dup = st.tabs(["Term lookup", "Vocabulary", "Document & metadata store", "Duplicates"])
    with t_term:
        word = st.text_input("Look up a word in the inverted index (wildcards allowed)", "learning")
        for t in (idx.expand_wildcard(word) if "*" in word else idx.tok(word))[:5]:
            post = idx.postings.get(t)
            if not post:
                st.warning(f"`{t}` is not in the index.")
                continue
            st.markdown(f"Term **`{t}`**: df = {len(post)}, idf = {idx.idf[t]:.3f}")
            st.dataframe(pd.DataFrame([{"doc_id": idx.doc_ids[d], "tf": len(p), "positions": str(p[:15]), "title": idx.titles[d]}
                                       for d, p in post.items()]), hide_index=True)
    with t_vocab:
        vocab = pd.DataFrame({"term": idx.vocab, "df": [len(idx.postings[t]) for t in idx.vocab],
                              "cf": [sum(len(p) for p in idx.postings[t].values()) for t in idx.vocab],
                              "idf": [round(idx.idf[t], 3) for t in idx.vocab]})
        st.dataframe(vocab.sort_values("df", ascending=False), hide_index=True, height=350)
    with t_store:
        c = st.session_state.corpus
        st.markdown("Content and metadata are kept in **separate tables / files**, joined by `doc_id`.")
        st.markdown("**documents.jsonl** (content)")
        st.dataframe(c["docs"], hide_index=True, height=230)
        st.markdown("**metadata.csv** (metadata)")
        st.dataframe(c["meta"], hide_index=True, height=230)
        st.markdown(f"**links.csv** ({len(c['edges'])} hyperlinks)")
        st.dataframe(c["edges"], hide_index=True, height=180)
    with t_dup:
        st.markdown("Exact duplicates and duplicate URLs are blocked at ingestion. **Near-duplicates** are found by "
                    "comparing 64-bit SimHash fingerprints (word 3-shingles).")
        max_dist = st.slider("Max Hamming distance", 0, 16, 6)
        pairs = near_duplicates(st.session_state.corpus["meta"], max_dist)
        if pairs.empty:
            st.success("No near-duplicate pairs at this threshold.")
        else:
            titles = dict(zip(idx.doc_ids, idx.titles))
            pairs["title_a"], pairs["title_b"] = pairs["doc_a"].map(titles), pairs["doc_b"].map(titles)
            st.dataframe(pairs, hide_index=True)
            if st.button(f"Remove {len(pairs)} near-duplicates (keep the first of each pair)"):
                drop = set(pairs["doc_b"])
                c = st.session_state.corpus
                c["docs"] = c["docs"][~c["docs"]["doc_id"].isin(drop)].reset_index(drop=True)
                c["meta"] = c["meta"][~c["meta"]["doc_id"].isin(drop)].reset_index(drop=True)
                c["edges"] = c["edges"][~c["edges"]["src"].isin(drop) & ~c["edges"]["dst"].isin(drop)].reset_index(drop=True)
                corpus_changed()
                st.rerun()


def page_mining():
    need_corpus()
    idx, docs = get_index(), st.session_state.corpus["docs"]
    texts = (docs["title"] + ". " + docs["text"]).tolist()
    cats = categories(idx.doc_ids)
    section = st.radio("Section", ["Corpus characteristics", "Keywords", "Document profiling", "Classification"], horizontal=True)

    if section == "Corpus characteristics":
        rows, stats = [], {}
        raw_count = sum(len(re.findall(r"\w+", t.lower())) for t in texts)
        for name, prep in PRESETS.items():
            toks = [tokenize(t, **prep) for t in texts]
            stats[name] = toks
            n_tok = sum(map(len, toks))
            rows.append({"strategy": name, "tokens": n_tok, "vocabulary": len({t for d in toks for t in d}),
                         "token reduction": round(1 - n_tok / raw_count, 3)})
        table = pd.DataFrame(rows)
        st.markdown("**Effect of preprocessing strategies**")
        st.dataframe(table, hide_index=True)
        strategy = st.selectbox("Strategy for the plots", list(PRESETS), index=2)
        toks = stats[strategy]
        tf = Counter(t for d in toks for t in d)
        zipf = pd.DataFrame([{"rank": i, "term": t, "frequency": f} for i, (t, f) in enumerate(tf.most_common(500), 1)])
        seen, total, heaps = set(), 0, []
        for d in toks:
            seen.update(d)
            total += len(d)
            heaps.append({"tokens seen": total, "vocabulary": len(seen)})
        a, b = st.columns(2)
        a.markdown("**Zipf's law** (log-log rank vs frequency)")
        a.altair_chart(alt.Chart(zipf).mark_line(point=True).encode(
            x=alt.X("rank:Q", scale=alt.Scale(type="log")), y=alt.Y("frequency:Q", scale=alt.Scale(type="log")),
            tooltip=["rank", "term", "frequency"]).properties(height=260))
        b.markdown("**Heaps' law** (vocabulary growth)")
        b.altair_chart(alt.Chart(pd.DataFrame(heaps)).mark_line().encode(x="tokens seen:Q", y="vocabulary:Q").properties(height=260))
        st.markdown("**Top 25 terms**")
        st.altair_chart(alt.Chart(zipf.head(25)).mark_bar().encode(x=alt.X("term:N", sort="-y", title=None),
                                                                   y="frequency:Q").properties(height=240))

    elif section == "Keywords":
        weight = np.asarray(idx.tfidf.sum(axis=0)).ravel()
        top = np.argsort(-weight)[:25]
        kw = pd.DataFrame({"term": [idx.vocab[j] for j in top], "tfidf_weight": weight[top].round(3),
                           "doc_freq": [len(idx.postings[idx.vocab[j]]) for j in top]})
        a, b = st.columns(2)
        a.markdown("**Corpus keywords** (summed TF-IDF)")
        a.altair_chart(alt.Chart(kw).mark_bar().encode(x="tfidf_weight:Q", y=alt.Y("term:N", sort="-x", title=None),
                                                       tooltip=list(kw.columns)).properties(height=480))
        if any(cats):
            rows = []
            for c in [c for c in pd.unique(cats) if c]:
                mean = np.asarray(idx.tfidf[np.flatnonzero(cats == c)].mean(axis=0)).ravel()
                rows.append({"category": c, "keywords": ", ".join(idx.vocab[j] for j in np.argsort(-mean)[:8])})
            b.markdown("**Most characteristic terms per category**")
            b.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown("**Per-document keywords**")
        st.dataframe(pd.DataFrame({"doc_id": idx.doc_ids, "title": idx.titles,
                                   "keywords": [", ".join(t for t, _ in doc_keywords(idx, i, 6)) for i in range(len(idx.doc_ids))]}),
                     hide_index=True, height=300)

    elif section == "Document profiling":
        did = st.selectbox("Document", [doc_label(d) for d in idx.doc_ids]).split(" - ")[0]
        i = idx.doc_ids.index(did)
        a, b = st.columns([2, 3])
        a.table(pd.Series(profile(idx.texts[i], idx.titles[i], idx, doc_keywords(idx, i)), name="value").astype(str))
        b.altair_chart(alt.Chart(pd.DataFrame(doc_keywords(idx, i, 10), columns=["term", "tfidf"])).mark_bar().encode(
            x="tfidf:Q", y=alt.Y("term:N", sort="-x", title=None)).properties(title="Keyword weights", height=300))
        with b.expander("Raw vs processed text"):
            st.write(idx.texts[i][:1200])
            st.code(" ".join(idx.tok(idx.texts[i]))[:1200], language=None)
        allp = pd.DataFrame([profile(idx.texts[j], idx.titles[j], idx, doc_keywords(idx, j)) for j in range(len(idx.doc_ids))])
        allp.insert(0, "doc_id", idx.doc_ids)
        allp["category"] = cats
        st.markdown("**Profiles of all documents**")
        st.dataframe(allp, hide_index=True, height=260)
        st.altair_chart(alt.Chart(allp).mark_circle(size=70).encode(x="Words:Q", y="Type-token ratio:Q", color="category:N",
                                                                    tooltip=["doc_id", "Title"]).properties(height=280))

    else:
        labeled = cats != ""
        X = [t for t, ok in zip(texts, labeled) if ok]
        y = [c for c in cats if c]
        counts = Counter(y)
        X = [t for t, l in zip(X, y) if counts[l] >= 2]
        y = [l for l in y if counts[l] >= 2]
        if len(set(y)) < 2:
            st.warning("Classification needs category labels with at least 2 documents in 2 or more categories.")
            return
        st.caption(f"{len(X)} labelled documents in {len(set(y))} classes")
        with st.form("clf"):
            presets = st.multiselect("Preprocessing strategies", list(PRESETS), default=list(PRESETS))
            feats = st.multiselect("Feature extraction", list(FEATURES), default=list(FEATURES))
            clfs = st.multiselect("Classifiers", list(CLASSIFIERS), default=list(CLASSIFIERS))
            folds = st.slider("Cross-validation folds", 2, 10, 5)
            run = st.form_submit_button("Run comparison", type="primary")
        if run and presets and feats and clfs:
            bar = st.progress(0.0)
            table, y_true, preds, used_folds = compare_classifiers(X, y, presets, feats, clfs, folds, bar.progress)
            bar.empty()
            st.session_state.mining = {"table": table, "y": y_true, "preds": preds, "folds": used_folds, "X": X, "labels": y}
        res = st.session_state.mining
        if res:
            table = res["table"].sort_values("macro_F1", ascending=False)
            best = table.iloc[0]
            st.success(f"Best: **{best['preprocessing']} / {best['features']} / {best['classifier']}** - accuracy "
                       f"{best['accuracy']:.3f}, macro-F1 {best['macro_F1']:.3f} ({res['folds']}-fold CV)")
            st.dataframe(table.round(3), hide_index=True)
            a, b = st.columns(2)
            a.altair_chart(alt.Chart(table).mark_rect().encode(
                x=alt.X("features:N", title=None), y=alt.Y("preprocessing:N", title=None),
                color=alt.Color("mean(macro_F1):Q", title="Macro-F1"), tooltip=["preprocessing", "features", "mean(macro_F1):Q"]
            ).properties(title="Macro-F1 averaged over classifiers", height=260))
            labs = sorted(set(res["y"]))
            cm = confusion_matrix(res["y"], res["preds"][(best["preprocessing"], best["features"], best["classifier"])], labels=labs)
            cm_df = pd.DataFrame([{"actual": labs[i], "predicted": labs[j], "count": int(cm[i, j])}
                                  for i in range(len(labs)) for j in range(len(labs))])
            b.altair_chart((alt.Chart(cm_df).mark_rect().encode(x="predicted:N", y="actual:N", color="count:Q")
                            + alt.Chart(cm_df).mark_text().encode(x="predicted:N", y="actual:N", text="count:Q")
                            ).properties(title="Confusion matrix (best configuration)", height=260))
            new = st.text_area("Classify new text with the best configuration", "The central bank raised interest rates to fight inflation.")
            if st.button("Predict"):
                prep = PRESETS[best["preprocessing"]]
                vec = FEATURES[best["features"]](lambda d: tokenize(d, **prep))
                model = CLASSIFIERS[best["classifier"]]().fit(vec.fit_transform(res["X"]), res["labels"])
                st.metric("Predicted category", model.predict(vec.transform([new]))[0])


def page_search():
    need_corpus()
    idx = get_index()
    with st.form("search"):
        q = st.text_input("Query", placeholder='machine learning  |  "interest rates"  |  invest*  |  travel AND NOT budget', key="query")
        a, b, c, d = st.columns([2, 1, 1, 1])
        model = a.selectbox("Retrieval model", MODELS, index=2)
        top_k = b.number_input("Top K", 1, 100, 10)
        pr_w = c.slider("PageRank weight", 0.0, 1.0, 0.3, 0.05)
        spell = d.toggle("Spell correction", True)
        with st.expander("Query optimisation settings"):
            a, b, c = st.columns(3)
            prf_docs = a.slider("PRF feedback docs", 1, 20, 5)
            prf_terms = b.slider("PRF expansion terms", 1, 20, 5)
            graph = c.selectbox("Link graph", ["auto", "hyperlinks", "similarity"])
        go = st.form_submit_button("Search", type="primary")
    st.caption('Syntax: free text (ranked) - "exact phrase" - wildcard comput* - Boolean AND / OR / NOT with ( ) '
               "when model = Boolean.")
    if go and q.strip():
        query = q
        if spell:
            corrected, fixes = idx.spell_correct(q)
            if fixes:
                st.info("Did you mean: **" + corrected + "**  (" + ", ".join(f"{a} -> {b}" for a, b in fixes.items()) + ")")
                query = corrected
        get_links(graph)
        res, info, ms = idx.search(query, model, int(top_k), pr_w, prf_docs, prf_terms)
        st.session_state.search_log.append({"time": datetime.now().strftime("%H:%M:%S"), "query": query, "model": model,
                                            "latency_ms": round(ms, 3), "results": len(res)})
        st.session_state.last_search = {"query": query, "model": model, "res": res, "info": info, "ms": ms,
                                        "top_k": int(top_k), "pr_w": pr_w, "graph": graph}
    ls = st.session_state.last_search
    if not ls:
        return
    res, info = ls["res"], ls["info"]
    a, b, c, d = st.columns(4)
    a.metric("Matching documents", info["n_matched"])
    b.metric("Returned", len(res))
    c.metric("Latency", f"{ls['ms']:.2f} ms")
    d.metric("Model", ls["model"])
    st.markdown(f"**Processed query terms:** `{' '.join(info['terms']) or '-'}`"
                + (f"  \n**PRF expansion terms:** `{' '.join(info['expanded'])}`" if info["expanded"] else ""))
    if res.empty:
        st.warning("No documents matched.")
        return
    meta = st.session_state.corpus["meta"].set_index("doc_id")
    res = res.join(meta[["url", "category", "source", "n_words", "fetched_at"]], on="doc_id")
    for r in res.itertuples():
        with st.container(border=True):
            st.markdown(f"**{r.rank}. {r.title}**  `{r.doc_id}`")
            st.markdown(r.snippet)
            st.caption(f"score {r.score:.4f} - text {r.text_score:.4f} - PageRank {r.pagerank:.5f} - "
                       f"{r.category or 'no category'} - {r.source} - {r.n_words} words - {r.fetched_at}"
                       + (f" - {r.url}" if r.url else ""))

    st.subheader("Why ranking matters: BM25 vs BM25 + PageRank")
    get_links(ls["graph"])
    base = idx.search(ls["query"], "BM25", ls["top_k"])[0][["doc_id", "title", "rank"]]
    mixed = idx.search(ls["query"], "BM25 + PageRank", ls["top_k"], ls["pr_w"])[0][["doc_id", "rank", "pagerank"]]
    cmp = base.merge(mixed, on="doc_id", how="outer", suffixes=("_bm25", "_with_pagerank"))
    cmp["rank change"] = cmp["rank_bm25"] - cmp["rank_with_pagerank"]
    st.dataframe(cmp.sort_values("rank_with_pagerank"), hide_index=True)
    long = pd.concat([cmp.assign(system="BM25", rank=cmp["rank_bm25"]),
                      cmp.assign(system="BM25 + PageRank", rank=cmp["rank_with_pagerank"])]).dropna(subset=["rank"])
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("system:N", title=None), y=alt.Y("rank:Q", scale=alt.Scale(reverse=True)),
        color=alt.Color("doc_id:N", legend=None), tooltip=["doc_id", "rank", "pagerank"]).properties(height=300))

    with st.expander("Add relevance judgements for this query"):
        judge = res[["doc_id", "title"]].assign(relevance=0)
        edited = st.data_editor(judge, hide_index=True, disabled=["doc_id", "title"], key=f"judge_{ls['query']}",
                                column_config={"relevance": st.column_config.SelectboxColumn(options=[0, 1, 2, 3])})
        if st.button("Save judgements"):
            qr = st.session_state.qrels
            qr = qr[~((qr["query"] == ls["query"]) & qr["doc_id"].isin(edited["doc_id"]))]
            new = edited[edited["relevance"] > 0][["doc_id", "relevance"]].assign(query=ls["query"])
            st.session_state.qrels = pd.concat([qr, new[["query", "doc_id", "relevance"]]], ignore_index=True)
            st.toast(f"Saved {len(new)} judgements")


def page_ranking():
    need_corpus()
    idx = get_index()
    a, b, c = st.columns(3)
    graph = a.selectbox("Link graph", ["auto", "hyperlinks", "similarity"])
    damping = b.slider("Damping factor d", 0.50, 0.99, 0.85, 0.01)
    knn = c.slider("k for the similarity graph", 2, 15, 5)
    L = get_links(graph, damping, knn)
    adj = L["adj"]
    a, b, c, d, e = st.columns(5)
    a.metric("Graph", L["graph"].split(" (")[0])
    b.metric("Edges", int(adj.nnz))
    c.metric("PageRank iterations", len(L["pr_history"]))
    d.metric("HITS iterations", len(L["hits_history"]))
    e.metric("Compute time", f"{L['ms']:.1f} ms")
    scores = pd.DataFrame({"doc_id": idx.doc_ids, "title": idx.titles, "category": categories(idx.doc_ids),
                           "pagerank": L["pagerank"], "authority": L["authority"], "hub": L["hub"],
                           "in_links": np.asarray(adj.sum(axis=0)).ravel().astype(int),
                           "out_links": np.asarray(adj.sum(axis=1)).ravel().astype(int)})
    t_scores, t_graph, t_conv, t_query = st.tabs(["Scores", "Graph", "Convergence", "Effect on a query"])
    with t_scores:
        by = st.radio("Rank by", ["pagerank", "authority", "hub", "in_links"], horizontal=True)
        top = scores.sort_values(by, ascending=False).head(20)
        st.altair_chart(alt.Chart(top).mark_bar().encode(x=f"{by}:Q", y=alt.Y("title:N", sort="-x", title=None),
                                                         color="category:N", tooltip=list(top.columns)).properties(height=460))
        st.dataframe(scores.sort_values(by, ascending=False), hide_index=True, height=280)
        st.markdown("**Spearman correlation between the scores**")
        st.dataframe(scores[["pagerank", "authority", "hub", "in_links"]].corr(method="spearman").round(3))
    with t_graph:
        import networkx as nx
        keep = np.argsort(-L["pagerank"])[:300]
        sub = adj[keep][:, keep]
        layout = nx.spring_layout(nx.from_scipy_sparse_array(sub).to_undirected(), seed=7)
        pos = np.array([layout[i] for i in range(sub.shape[0])])
        nodes = scores.iloc[keep].reset_index(drop=True).assign(x=pos[:, 0], y=pos[:, 1])
        coo = sub.tocoo()
        lines = pd.DataFrame({"x": pos[coo.row, 0], "y": pos[coo.row, 1], "x2": pos[coo.col, 0], "y2": pos[coo.col, 1]})
        st.altair_chart((alt.Chart(lines).mark_rule(opacity=0.15).encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2", y2="y2")
                         + alt.Chart(nodes).mark_circle().encode(x="x:Q", y="y:Q", size=alt.Size("pagerank:Q", scale=alt.Scale(range=[20, 900])),
                                                                 color="category:N", tooltip=["doc_id", "title", "pagerank", "authority", "hub"])
                         ).properties(height=520).interactive())
        st.caption("Node size = PageRank.")
    with t_conv:
        hist = pd.concat([pd.DataFrame(L["pr_history"]).assign(algorithm="PageRank"),
                          pd.DataFrame(L["hits_history"]).assign(algorithm="HITS")])
        st.altair_chart(alt.Chart(hist).mark_line().encode(x="iteration:Q", y=alt.Y("l1_change:Q", scale=alt.Scale(type="log"), title="L1 change (log)"),
                                                           color="algorithm:N").properties(height=300, title="Power-iteration convergence"))
        rows = []
        for dd in [0.5, 0.65, 0.75, 0.85, 0.95]:
            pr, h = pagerank(adj, dd)
            rows.append({"damping": dd, "iterations": len(h), "max PageRank": round(pr.max(), 4),
                         "top-5 documents": ", ".join(idx.doc_ids[i] for i in np.argsort(-pr)[:5])})
        st.markdown("**Sensitivity to the damping factor**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with t_query:
        q = st.text_input("Query", "learning")
        rows = []
        for w in [0.0, 0.2, 0.4, 0.6, 0.8]:
            for r in idx.search(q, "BM25 + PageRank", 10, w)[0].itertuples():
                rows.append({"PageRank weight": w, "doc_id": r.doc_id, "title": r.title, "rank": r.rank})
        if rows:
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
                x="PageRank weight:O", y=alt.Y("rank:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("title:N", legend=alt.Legend(orient="bottom", columns=2)), tooltip=["doc_id", "title", "rank"]
            ).properties(height=420, title="Rank of each document as the PageRank weight grows"), use_container_width=True)
        else:
            st.info("No documents matched this query.")


def page_recommend():
    need_corpus()
    idx = get_index()
    ids, titles = idx.doc_ids, dict(zip(idx.doc_ids, idx.titles))
    cats = dict(zip(ids, categories(ids)))
    ratings = st.session_state.ratings
    ratings = ratings[ratings["doc_id"].astype(str).isin(set(ids))]
    if ratings.empty:
        st.warning("No user ratings loaded; collaborative filtering needs interaction data.")
        if st.button("Simulate ratings for 40 users"):
            st.session_state.ratings = simulate_ratings(ids, list(categories(ids)))
            st.rerun()
    k = st.slider("Top-K", 1, 20, 5)
    mode = st.radio("Mode", ["Item-to-item", "Personalised for a user", "Offline evaluation"], horizontal=True)
    m, users, sim = None, [], None
    if len(ratings):
        m, users = rating_matrix(ratings, ids)
        sim = item_similarity(m)
    table = lambda order, score_cols: pd.DataFrame([{"rank": r + 1, "doc_id": ids[j], "title": titles[ids[j]],
                                                     "category": cats[ids[j]], **{n: round(float(s[j]), 4) for n, s in score_cols.items()}}
                                                    for r, j in enumerate(order)])
    if mode == "Item-to-item":
        did = st.selectbox("Seed document", [doc_label(d) for d in ids]).split(" - ")[0]
        i = ids.index(did)
        content = np.asarray((idx.tfidf @ idx.tfidf[i].T).todense()).ravel()
        content[i] = -np.inf
        collab = sim[i].copy() if sim is not None else np.zeros(len(ids))
        collab[i] = -np.inf
        a, b = st.columns(2)
        a.markdown("**Content-based** (cosine of TF-IDF vectors)")
        a.dataframe(table(np.argsort(-content)[:k], {"cosine": content}), hide_index=True)
        b.markdown("**Collaborative** (adjusted cosine of ratings)")
        if sim is not None:
            b.dataframe(table(np.argsort(-collab)[:k], {"similarity": collab}), hide_index=True)
        alpha = st.slider("Hybrid weight alpha (1 = content only, 0 = collaborative only)", 0.0, 1.0, 0.6, 0.05)
        hy = hybrid(content, collab, alpha)
        hyt = table(np.argsort(-hy)[:k], {"content": content, "collaborative": collab, "hybrid": hy})
        st.markdown("**Hybrid recommendations**")
        st.dataframe(hyt, hide_index=True)
        long = hyt.melt(id_vars=["title"], value_vars=["content", "collaborative", "hybrid"], var_name="score")
        st.altair_chart(alt.Chart(long).mark_bar().encode(y=alt.Y("title:N", sort=None, title=None), x="value:Q",
                                                          color="score:N", yOffset="score:N").properties(height=60 * len(hyt) + 40))
    elif mode == "Personalised for a user":
        if m is None:
            st.info("User ratings are required for collaborative personalization. Please load or simulate ratings above.")
        else:
            u = st.selectbox("User", users)
            uidx = users.index(u)
            hist = ratings[ratings["user_id"].astype(str) == u].assign(title=lambda x: x["doc_id"].map(titles),
                                                                       category=lambda x: x["doc_id"].map(cats))
            st.markdown(f"**History of {u}** ({len(hist)} rated items)")
            st.dataframe(hist.sort_values("rating", ascending=False), hide_index=True, height=200)
            row = m[uidx].toarray().ravel()
            liked = list(np.flatnonzero(row >= 4)) or list(np.flatnonzero(row))
            cb = content_profile_scores(idx.tfidf, liked, exclude=np.flatnonzero(row))
            cf = collaborative_scores(m, sim, uidx)
            alpha = st.slider("Hybrid weight alpha", 0.0, 1.0, 0.5, 0.05)
            hy = hybrid(cb, cf, alpha)
            for col, name, s in zip(st.columns(3), ["Content-based", "Collaborative", f"Hybrid (alpha={alpha})"], [cb, cf, hy]):
                col.markdown(f"**{name}**")
                col.dataframe(table([j for j in np.argsort(-s) if np.isfinite(s[j])][:k], {"score": s}), hide_index=True)
    elif mode == "Offline evaluation":
        if m is None:
            st.info("User ratings are required for offline evaluation. Please load or simulate ratings above.")
        else:
            st.markdown("Leave-one-out: for each user one item rated 4 or 5 is hidden; does each method put it in the top-K?")
            if st.button("Run offline evaluation", type="primary"):
                st.session_state.rec_eval = leave_one_out(m, idx.tfidf, k, (0.0, 0.25, 0.5, 0.75, 1.0))
            ev = st.session_state.rec_eval
            if ev is not None:
                st.dataframe(ev.round(3), hide_index=True)
                long = ev.melt(id_vars=["alpha"], value_vars=[c for c in ev.columns if "@" in c], var_name="metric")
                st.altair_chart(alt.Chart(long).mark_line(point=True).encode(x=alt.X("alpha:Q", title="alpha (0 = collaborative, 1 = content)"),
                                                                             y="value:Q", color="metric:N").properties(height=300), use_container_width=True)


def page_evaluation():
    need_corpus()
    idx = get_index()
    qrels = st.session_state.qrels
    qrels = qrels[qrels["doc_id"].astype(str).isin(set(idx.doc_ids))].reset_index(drop=True)
    st.caption("Judgements come from the sample collection, an uploaded CSV, the Search page, or are generated from category labels.")
    a, b = st.columns([3, 1])
    if b.button("Auto-generate from categories"):
        st.session_state.qrels = auto_judgements(idx, st.session_state.corpus["meta"])
        st.rerun()
    edited = a.data_editor(qrels, num_rows="dynamic", hide_index=True, height=220, key="qrels_editor")
    a.caption(f"{edited['query'].nunique()} queries, {len(edited)} judgements")
    if not edited.empty and not edited.equals(st.session_state.qrels):
        st.session_state.qrels = edited.copy()
        if (STORE_DIR / "documents.jsonl").exists():
            edited.to_csv(STORE_DIR / "qrels.csv", index=False)
    if edited.empty:
        st.info("Add or generate relevance judgements to evaluate.")
        return
    with st.form("eval_form"):
        a, b, c = st.columns([3, 1, 1])
        models = a.multiselect("Systems to compare", MODELS, default=MODELS)
        k = int(b.number_input("K", 1, 50, 5))
        depth = int(c.number_input("Results per query", 5, 200, 20))
        run = st.form_submit_button("Run evaluation", type="primary")
    if run:
        get_links()
        rows, runs = [], {}
        for model in models:
            for q, grp in edited.groupby("query"):
                grades = dict(zip(grp["doc_id"].astype(str), grp["relevance"].astype(float)))
                res, _, ms = idx.search(q, model, depth)
                ret = res["doc_id"].tolist()
                runs[(model, q)] = (ret, grades)
                rows.append({"system": model, "query": q, "latency_ms": ms, **evaluate_run(ret, grades, k)})
        st.session_state.eval = {"per_query": pd.DataFrame(rows), "runs": runs, "k": k}
    ev = st.session_state.eval
    if ev is None:
        return
    pq, k = ev["per_query"], ev["k"]
    metric_cols = ["Precision", "Recall", "F1", f"P@{k}", f"R@{k}", "AP", "RR", f"NDCG@{k}"]
    summary = pq.groupby("system")[metric_cols].mean().rename(columns={"AP": "MAP", "RR": "MRR"}).sort_values("MAP", ascending=False)
    st.success(f"Best system by MAP: **{summary.index[0]}** (MAP {summary['MAP'].iloc[0]:.3f}, NDCG@{k} {summary[f'NDCG@{k}'].iloc[0]:.3f})")
    st.dataframe(summary.style.format("{:.3f}").highlight_max(axis=0, color="#c8e6c9"))
    long = summary.reset_index().melt(id_vars="system", var_name="metric")
    st.altair_chart(alt.Chart(long).mark_bar().encode(x=alt.X("metric:N", sort=list(summary.columns), title=None),
                                                      y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])), color="system:N",
                                                      xOffset="system:N").properties(height=320, title="System comparison"))
    a, b = st.columns(2)
    curves = [pr_curve(ret, {d for d, g in grades.items() if g > 0}).assign(system=m) for (m, _), (ret, grades) in ev["runs"].items()]
    a.altair_chart(alt.Chart(pd.concat(curves).groupby(["system", "recall"], as_index=False)["precision"].mean()).mark_line(point=True).encode(
        x="recall:Q", y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])), color="system:N").properties(height=300, title="11-point precision-recall curve"))
    atk = []
    for (m, _), (ret, grades) in ev["runs"].items():
        rel = {d for d, g in grades.items() if g > 0}
        atk += [{"system": m, "K": kk, "NDCG@K": ndcg(ret, grades, kk), "P@K": precision_at_k(ret, rel, kk)} for kk in range(1, 21)]
    b.altair_chart(alt.Chart(pd.DataFrame(atk).groupby(["system", "K"], as_index=False).mean(numeric_only=True)).mark_line().encode(
        x="K:Q", y=alt.Y("NDCG@K:Q", scale=alt.Scale(domain=[0, 1])), color="system:N").properties(height=300, title="NDCG@K as K grows"))
    metric = st.selectbox("Per-query metric", metric_cols, index=metric_cols.index("AP"))
    st.altair_chart(alt.Chart(pq).mark_rect().encode(x=alt.X("system:N", title=None), y=alt.Y("query:N", title=None),
                                                     color=alt.Color(f"{metric}:Q", scale=alt.Scale(domain=[0, 1])),
                                                     tooltip=["system", "query", f"{metric}:Q"]).properties(height=max(200, 24 * pq["query"].nunique())))
    with st.expander("Per-query table"):
        st.dataframe(pq.round(4), hide_index=True)


def page_performance():
    need_corpus()
    idx = get_index()
    log = pd.DataFrame(st.session_state.search_log)
    a, b, c, d = st.columns(4)
    a.metric("Index build", f"{idx.build_ms:.0f} ms")
    b.metric("Queries logged", len(log))
    c.metric("Median latency", f"{log['latency_ms'].median():.2f} ms" if len(log) else "-")
    d.metric("p95 latency", f"{log['latency_ms'].quantile(0.95):.2f} ms" if len(log) else "-")
    st.markdown("**Query log**")
    if log.empty:
        st.caption("Run some searches to fill the log.")
    else:
        st.altair_chart(alt.Chart(log.reset_index()).mark_line(point=True).encode(
            x=alt.X("index:Q", title="query #"), y="latency_ms:Q", color="model:N", tooltip=["query", "model", "latency_ms"]).properties(height=240))
        st.dataframe(log.iloc[::-1], hide_index=True)
    st.markdown("**Latency benchmark**: every model on a query workload")
    default = "\n".join(st.session_state.qrels["query"].unique()[:10]) or "machine learning\nhealth\n\"search engine\"\ntravel*"
    queries = [x for x in st.text_area("Workload (one query per line)", default, height=140).splitlines() if x.strip()]
    reps = st.slider("Repetitions", 1, 20, 5)
    if st.button("Run benchmark", type="primary"):
        if not queries:
            st.warning("Please provide at least one query in the workload.")
        else:
            get_links()
            rows = []
            for model in MODELS:
                for q in queries:
                    for _ in range(reps):
                        rows.append({"model": model, "ms": idx.search(q, model, 10)[2]})
            st.session_state.bench = pd.DataFrame(rows)
    bench = st.session_state.bench
    if bench is not None and not bench.empty:
        agg = bench.groupby("model")["ms"].agg(mean="mean", median="median", p95=lambda s: s.quantile(0.95)).round(3)
        agg["queries/sec"] = (1000 / agg["mean"]).round(0)
        st.dataframe(agg)
        st.altair_chart(alt.Chart(bench).mark_boxplot().encode(x=alt.X("model:N", title=None), y=alt.Y("ms:Q", title="latency (ms)"),
                                                               color="model:N").properties(height=300), use_container_width=True)


def page_inference():
    st.caption("Answers to Section G. The 'Our results' boxes use the experiments run in this session "
               "(run Text mining -> Classification, Evaluation and Recommendations -> Offline evaluation first).")
    ss = st.session_state
    facts = {}
    if n_docs():
        idx = get_index()
        facts["corpus"] = f"{n_docs()} documents, {len(idx.vocab):,} terms, {len(ss.corpus['edges'])} links"
        facts["dups"] = (f"{sum(x['dup_url'] for x in ss.ingest_log)} duplicate URLs and {sum(x['dup_content'] for x in ss.ingest_log)} "
                         f"exact duplicates skipped at ingestion; {len(near_duplicates(ss.corpus['meta'], 6))} near-duplicate pairs (Hamming <= 6)")
    if ss.eval is not None:
        k = ss.eval["k"]
        table = ss.eval["per_query"].groupby("system")[["AP", "RR", f"NDCG@{k}", "Recall"]].mean().rename(columns={"AP": "MAP", "RR": "MRR"})
        facts["eval"], facts["best"] = table.round(3), table["MAP"].idxmax()

    def results(lines):
        lines = [l for l in lines if l]
        if lines:
            st.info("**Our results:** " + "  \n".join(lines))

    st.subheader("1. Relevant documents retrieved but ranked poorly")
    st.markdown("""
**Causes:** term-frequency bias towards long documents (no length normalisation); vocabulary mismatch between query and
document; title matches weighted like body matches; no query-independent quality signal; duplicates filling the top slots;
query expansion that drifts.

**Improvements:** BM25 with tuned k1 and b; field weighting and phrase / proximity scoring; adding link authority
(BM25 + PageRank, HITS) with a weight tuned on judged queries; learning-to-rank or a neural re-ranker on the top results;
collapsing duplicates and diversifying results.""")
    if "eval" in facts:
        t = facts["eval"]
        lines = [f"Best system by MAP: {facts['best']} ({t.loc[facts['best'], 'MAP']:.3f})."]
        if {"BM25", "BM25 + PageRank"} <= set(t.index):
            lines.append(f"BM25 MAP {t.loc['BM25', 'MAP']:.3f} -> BM25 + PageRank {t.loc['BM25 + PageRank', 'MAP']:.3f} with the same recall: "
                         "only the order changed.")
        results(lines)
        st.dataframe(t)

    st.subheader("2. Effect of duplicate and near-duplicate documents")
    st.markdown("""
| Stage | Effect | Mitigation |
|---|---|---|
| Indexing | Bigger index, inflated document frequencies distort IDF, wasted crawl budget | URL normalisation, MD5 at ingestion, SimHash before indexing |
| Ranking | Copies fill the top-K and reduce diversity; copies linking to each other inflate PageRank | Collapse duplicate clusters, diversify results |
| Recommendation | "More like this" returns copies of the seed; ratings split across copies | Similarity threshold filter, merge ratings onto one id |
| Evaluation | Unjudged copies count as non-relevant, or judged copies inflate scores | De-duplicate the test collection, judge at cluster level |
""")
    results([facts.get("dups", "")])

    st.subheader("3. Content-based vs collaborative recommendation")
    st.markdown("""
| | Content-based | Collaborative |
|---|---|---|
| Uses | item text (TF-IDF) | user ratings |
| New item | works immediately | fails (no ratings) |
| Novelty | low, more of the same | high, items liked by similar users |
| Needs | only the corpus | a dense enough rating matrix |

**Content-based** suits fast-changing catalogues (news, crawled pages), sparse interactions and explainable results.
**Collaborative** suits large interaction logs where taste matters (shopping, music, films). A **hybrid** usually works best.""")
    if ss.rec_eval is not None:
        hr = [c for c in ss.rec_eval.columns if c.startswith("HitRate")][0]
        results([", ".join(f"{m} (alpha={a}): {v:.3f}" for m, a, v in zip(ss.rec_eval["method"], ss.rec_eval["alpha"], ss.rec_eval[hr]))])

    st.subheader("4. How the components work together")
    st.markdown("""
- **Crawling** sets the ceiling on recall: a page never collected can never be found.
- **Preprocessing and text mining** turn raw text into consistent terms and add metadata (keywords, categories).
- **Indexing** makes search fast and stores the statistics ranking needs (df, document length, positions).
- **Search and ranking** combine query processing with BM25 relevance and PageRank / HITS authority.
- **Recommendation** reuses the same TF-IDF vectors and adds user behaviour.
- **Evaluation** shows which stage limits quality: low recall points to crawling or expansion, low NDCG to ranking.""")
    results([facts.get("corpus", "")])

    st.subheader("5. Learnings")
    learnings = ["BM25's length normalisation and saturating term frequency beat raw TF-IDF.",
                 "Recall is decided early (acquisition, preprocessing, expansion); the top of the ranking is decided by the ranking function.",
                 "PageRank helps as a moderate prior; too much weight lets popular but off-topic pages rise.",
                 "Pseudo-relevance feedback raises recall but drifts on vague queries.",
                 "Hashing and SimHash make de-duplication cheap and protect every later stage.",
                 "With sparse ratings, content-based recommendation is the safer default.",
                 "Several metrics are needed: MRR, MAP and NDCG each see a different part of the ranking."]
    if "eval" in facts:
        learnings.insert(0, f"{facts['best']} achieved the highest MAP on our collection.")
    st.markdown("\n".join(f"- {x}" for x in learnings))
    report = ["# Inference summary", ""] + [f"- {v}" for key, v in facts.items() if key in ("corpus", "dups")]
    if "eval" in facts:
        report += ["", "## Retrieval evaluation", "```", facts["eval"].to_string(), "```"]
    report += ["", "## Learnings"] + [f"- {x}" for x in learnings]
    st.download_button("Download inference summary (Markdown)", "\n".join(report), "inference_summary.md")


# =============================================================================
# 10. MAIN
# =============================================================================
PAGES = {
    "Dashboard": page_dashboard,
    "Data acquisition (crawler)": page_acquisition,
    "Index management": page_index,
    "Text mining": page_mining,
    "Search": page_search,
    "Ranking (PageRank / HITS)": page_ranking,
    "Recommendations": page_recommend,
    "Evaluation": page_evaluation,
    "Performance analytics": page_performance,
    "Inference & discussion": page_inference,
}

st.set_page_config(page_title="IR Studio", layout="wide")
init_state()
with st.sidebar:
    st.title("IR Studio")
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.metric("Documents", n_docs())
    st.caption("Group E - IR Assignment 2")
st.header(choice)
PAGES[choice]()
