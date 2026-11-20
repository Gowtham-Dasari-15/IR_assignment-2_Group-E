# IR Studio – Information Retrieval Assignment 2 (Group E)

A Streamlit application for **Information Retrieval Assignment 2 (AIMLCZG537 / DSECLZG537)**.
The whole workflow runs from the Streamlit front end: data acquisition and crawling, indexing, text mining,
search, ranking (PageRank / HITS), recommendations, evaluation and performance analytics.

## Files

```text
.
├── app.py                            # the complete Streamlit application
├── requirements.txt                  # Python dependencies
├── IR_Assignment2_Report_Group-E.docx # comprehensive assignment report
├── Group-E_Contribution.xlsx         # student contribution sheet
├── screenshots/                      # 18 high-resolution execution screenshots from Virtual Lab
├── data/sample/                      # benchmark dataset
│   ├── documents.csv                 #   54 articles (5 categories) with hyperlinks
│   ├── qrels.csv                     #   relevance judgements for 12 queries
│   └── ratings.csv                   #   user ratings for the recommender
└── .streamlit/config.toml            # UI theme configuration
```

At runtime the app stores the corpus in `data/store/`: `documents.jsonl` (content), `metadata.csv` (metadata)
and `links.csv` (hyperlinks), kept as separate files and joined by `doc_id`.

## 1. Install dependencies

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## 2. Run the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

### Running on the BITS Virtual Lab (RunPod / JupyterLab terminal)

The Virtual Lab pod has no browser, so Streamlit is exposed through a Cloudflare Tunnel.

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the standalone `cloudflared` binary, make it executable and put it on the PATH:
   ```bash
   curl -L --output /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared
   ```
3. Launch the app, listening on all interfaces:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
4. In a second terminal, start the tunnel. It prints a public `https://….trycloudflare.com` URL:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8501
   ```
5. Open that URL in any browser to use the app.

## 3. Using the app

1. **Data acquisition** – load the sample collection, upload a CSV or text/HTML files, crawl websites
   (multiple seeds, crawl depth, max pages, politeness delay, robots.txt, domain and URL filters) or fetch
   articles from the Wikipedia API. Duplicate URLs and duplicate documents are skipped.
2. **Index management** – preprocessing options, title boost and BM25 parameters; inspect postings, the
   vocabulary, the separate document / metadata / link stores, and near-duplicates (SimHash).
3. **Text mining** – compare preprocessing strategies, Zipf's and Heaps' laws, keywords, document profiles,
   and classification over preprocessing × features × classifier.
4. **Search** – BM25, TF-IDF cosine, BM25 + PageRank, BM25 + pseudo-relevance feedback and Boolean;
   phrase queries, wildcards, spelling correction, and a BM25 vs BM25 + PageRank comparison.
5. **Ranking** – PageRank and HITS scores, link graph, convergence, damping sensitivity.
6. **Recommendations** – content-based, collaborative and hybrid Top-K lists with similarity scores, and a
   leave-one-out evaluation.
7. **Evaluation** – Precision, Recall, F1, P@K, R@K, MAP, MRR and NDCG@K per system and per query.
8. **Performance analytics** – query latency log and a latency benchmark.
9. **Inference & discussion** – answers to Section G with the session's results.
