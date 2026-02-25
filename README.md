# task2vec

Not all engineering work is equal. Some tickets are repetitive patterns the team has solved dozens of times. Some sit at the edge of what anyone knows. Most are somewhere in between. task2vec makes the difference visible — so AI handles the routine, and engineers own the frontier.

Live demo: **[task2vec.com](https://task2vec.com)**

---

## The idea

Not all engineering work is equal — but until now there has been no reliable way to tell it apart at scale.

task2vec embeds every ticket into a semantic map built from your project's full history. That map makes three things visible at once: which work is safe to hand to an AI agent, which needs an AI draft with a human checking the result, and which problems are genuinely hard — the ones that require deep expertise, careful judgment, and the kind of engagement that actually grows an engineer's skills.

The goal is not to replace engineers. It is the opposite: free them from repetitive work so their time is spent entirely on the problems worth solving. **AI handles the routine. Engineers own the frontier.**

No manual tagging. No surveys. No process changes. Connect your Jira or GitHub, and the data speaks for itself.

---

## Vision

Software teams produce a continuous stream of tickets, comments, and commits. Buried in that stream are three signals that are almost never made explicit:

| Signal | Question it answers | How task2vec surfaces it |
|---|---|---|
| **Semantic map** | What is the team actually working on, in their own words? | Embed every ticket with LLM embeddings, cluster with UMAP + HDBSCAN, render as an interactive map |
| **Employee trajectory** | How is each person growing? Are they deepening expertise or scattered? | Track per-assignee paths through the embedding space over time — a trajectory through clusters is a growth story |
| **Real strategy** | Where is the project actually heading, regardless of what the roadmap says? | Compute the temporal drift vector of resolved tickets — the direction the work has moved is the real strategy |

These three signals together define a **capability envelope**: the semantic territory where the team knows what it's doing, who owns what, and where momentum is building.

---

## Proactive risk detection

The next layer uses the capability envelope to assess incoming work before it is assigned.

A ticket is a candidate for human review when it satisfies multiple of the following:

1. **Strategic drift** — it lands far from the strategy vector, pulling work away from the direction the project has been moving.
2. **Capability gap** — no team member has a resolved-ticket trajectory through the semantic area the ticket occupies. The team has never successfully delivered anything like it.
3. **Knowledge island** — only one person has ever worked in this cluster. Assigning the ticket to anyone else creates single-point-of-failure risk.

Comments matter as much as the title. A ticket titled "Fix login button" whose comments are full of OAuth token refresh and SAML assertions is not a UI ticket — it is an auth platform ticket in disguise. Re-embedding the ticket description together with its comments gives an honest placement on the map. If that placement drifts significantly from the title-only position, scope is expanding into unknown territory.

The goal is to surface the question *"does this belong here, and can we actually do it?"* at intake — when redirection is cheap — rather than six weeks into a stalled sprint.

---

## What it does

1. **ETL** — loads tickets from Jira (JSONL export or MongoDB), embeds each ticket summary with OpenAI `text-embedding-3-large`, and stores them in a NumPy search index.
2. **Clustering** — runs UMAP dimensionality reduction and HDBSCAN clustering to group tickets by semantic similarity.
3. **Explorer** — generates `spring_explorer.html`: an interactive Plotly map where you can explore clusters, see per-assignee work stories, and analyze a new ticket against the full history.
4. **Ticket Advisor API** — given a ticket description, returns the closest historical tickets, likely cluster, team experts, and a GPT-4o-mini narrative of how the ticket fits the project direction.
5. **AI Work Cockpit** — prototype oversight UI (`cockpit.html`) for monitoring AI agents working a backlog: drift detection, override hotspots, cluster deep-dive, and a natural-language Ask interface.

---

## Project layout

```
.
├── ticketing_intel/          # ETL package
│   ├── config.py             # paths, model names
│   ├── etl/
│   │   ├── loader.py         # JSONL → tickets
│   │   ├── mongo_loader.py   # MongoDB → tickets
│   │   ├── embedder.py       # OpenAI embeddings
│   │   └── pipeline.py       # orchestrates ETL
│   └── store/
│       └── sqlite_store.py   # local SQLite cache
│
├── build_search_index.py     # build .cache/search_index.npy (850 MB)
├── generate_umap_app.py      # generate stories_spring/spring_explorer.html
│
├── api/
│   ├── server.py             # Flask API (gunicorn, port 5001)
│   ├── analyze.php           # PHP proxy → /api/analyze
│   └── health.php            # PHP proxy → /api/health
│
├── cockpit.html              # AI Work Cockpit prototype (self-contained)
├── index.html                # Landing page
│
├── deploy/
│   ├── DEPLOY.md             # full deployment guide
│   ├── restart_api.sh        # keepalive cron script
│   ├── task2vec.conf         # Apache virtual host config
│   └── task2vec-api.service  # systemd service (if available)
│
├── .cache/                   # generated data (not in git)
│   ├── search_index.npy      # (N × 3072) float32, L2-normalised
│   ├── search_keys.npy       # ticket keys
│   ├── search_meta.json      # ticket metadata
│   └── umap_app_data.json    # cluster labels, strategy, UMAP coords
│
└── stories_spring/           # generated work stories + spring_explorer.html
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install openai numpy scikit-learn umap-learn hdbscan flask flask-cors gunicorn
```

### 2. Set your OpenAI key

```bash
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file (see `.env.example`).

### 3. Run the ETL pipeline

```bash
# Expects Jira export as JSONL in jira_100.jsonl (or configure MongoDB in ticketing_intel/config.py)
python run_etl.py
```

This embeds all tickets and writes `.cache/search_index.npy`.

### 4. Generate the explorer

```bash
python generate_umap_app.py
# → stories_spring/spring_explorer.html
```

Open the HTML file directly in a browser — no server needed for the explorer.

### 5. Run the API server

```bash
python api/server.py
# Listening on http://0.0.0.0:5001
```

Test:
```bash
curl http://localhost:5001/api/health
curl -X POST http://localhost:5001/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "Add reactive MongoDB support for Spring Data repositories"}'
```

---

## API

### `GET /api/health`
```json
{"status": "ok", "indexed": 69156}
```

### `POST /api/analyze`
Request body:
```json
{"text": "ticket description here"}
```

Response:
```json
{
  "umap_pos":       {"x": 1.2, "y": -0.4},
  "cluster":        "Database / ORM",
  "top_similarity": 0.912,
  "similar": [
    {"key": "SPR-1234", "summary": "...", "assignee": "...", "year": 2023, "similarity": 0.912}
  ],
  "experts": [
    {"name": "Jane Smith", "count": 7}
  ],
  "advice": "This ticket belongs to the data access layer..."
}
```

---

## Deployment

The live site runs on a Zoner.fi DirectAdmin VPS (nginx + PHP-FPM). Because `mod_proxy` is not available, the API is bridged via PHP proxy scripts (`api/analyze.php`, `api/health.php`) that forward requests to gunicorn on port 5001.

Gunicorn is kept alive by a cron job every 5 minutes:
```
*/5 * * * * /home/adloccbvmx/task2vec/restart_api.sh >> .../cron.log 2>&1
```

For a full step-by-step guide see **[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

### Memory note

`search_index.npy` is 850 MB in RAM (69k tickets × 3072 floats). To halve it:
```python
import numpy as np
vecs = np.load('.cache/search_index.npy')
np.save('.cache/search_index.npy', vecs.astype(np.float16))
```

---

## Cockpit prototype

`cockpit.html` is a self-contained prototype for the **AI Work Cockpit** — an oversight dashboard for teams where AI agents are actively solving tickets.

Features:
- UMAP map coloured by agent type (AI-agent, Human, Override)
- Strategy arrow vs. AI drift arrow
- Timeline scrubber (Jan 2024 – Feb 2026)
- Cluster deep-dive panel (work breakdown, override log, AI reasoning)
- Alert cards: drift warnings, override hotspots, aligned zones
- Ask modal (Ctrl+K) for natural-language queries

Live at: **[task2vec.com/cockpit.html](https://task2vec.com/cockpit.html)**

---

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-large` (3072-dim) |
| Dimensionality reduction | UMAP |
| Clustering | HDBSCAN |
| Similarity search | NumPy dot product (L2-normalised vectors) |
| LLM advice | OpenAI `gpt-4o-mini` |
| API server | Flask + gunicorn |
| Frontend | Plotly.js, vanilla JS |
| Web proxy | nginx + PHP-FPM (PHP curl proxy to gunicorn) |
