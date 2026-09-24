# 🎬 KinoAgent: Agentic Film Intelligence System

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph%20v0.2-FF6F00?style=for-the-badge&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-FF4F00?style=for-the-badge)](https://www.trychroma.com/)
[![Observability](https://img.shields.io/badge/Monitoring-Prometheus%20%2B%20Grafana-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Tracking](https://img.shields.io/badge/Evaluation-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)

**KinoAgent** is an agentic AI system engineered with **LangGraph**, combining semantic vector retrieval, Text-to-SQL generation over historical awards dataset, real-time REST API mutations with Human-in-the-Loop (HITL) approval, and MLOps telemetry.

**KinoAgent** creates personal vector embeddings directly from a user's Letterboxd export data (capturing personal ratings, diary dates, and reviews) while referencing an enriched TMDb semantic catalog. For watchlist updates, it links to the user's live TMDb account to execute watchlist additions and removals—gated explicitly by stateful HITL authorization via LangGraph interrupt() checkpoints. The system enforces reliability through an automated offline evaluation gate in CI/CD, defensive SQL validation, stream-level token isolation, and real-time metric instrumentation via Prometheus, Grafana, and MLflow.

---

## 🏗️ System Architecture

```text
                                  +---------------------------------------+
                                  |         Streamlit Frontend UI         |
                                  |                                       |
                                  +-------------------+-------------------+
                                                      |
                                                      | Webhook / Chat Stream
                                                      v
                                      +-------------------------------+
                                      |   LangGraph Stateful Agent    |
                                      |      (chat_node Router)       |
                                      +---------------+---------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
                    v                                 v                                 v
     [ Semantic Vector Search ]             [ Constrained Text-to-SQL ]       [ Live Services & Web ]
                    |                                 |                                 |
        +-----------+-----------+                     | (Regex Guard)                   |
        |                       |                     v                                 |
        v                       v         +-----------------------+                     |
  (Collection A)        (Collection B)    |      SQLite DB        |                     |
letterboxd_personal     tmdb_synopsis     | (oscars_nominations)  |                     |
  (User Diary/Taste)   (Factual Enriched) +-----------------------+                     |
                                                                                        |
                    +-------------------------------------------------------------------+
                    |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
         +--------------------+                  +--------------------+
         |   TMDb REST API    |                  |    Tavily Search   |
         | (Catalog Lookups & |                  | (Web Fallback for  |
         | HITL Watchlist)    |                  |  External News)    |
         +--------------------+                  +--------------------+
                    |
                    v
    [ Human-in-the-Loop Interrupt to update TMDB watchlist ]
 (Gated write execution awaiting UI signal)
```

---

##  Core Engineering Highlights

### 1. Multi-Tool Agentic Routing & Graph Orchestration
Built on **LangGraph (`StateGraph`)** using message state accumulation (`add_messages`) and persistent thread checkpointing (`SqliteSaver`):
* **Dual ChromaDB Retrievers:** Decouples subjective personal taste (`letterboxd_personal`) from objective film metadata (`tmdb_synopsis`), both indexed with local `all-MiniLM-L6-v2` embeddings.
* **TMDb REST Integration:** Live catalog resolution and metadata lookup via TMDb v3 API.
* **Tavily Fallback:** Dynamic web search routing when queries exceed local knowledge boundaries.

### 2. Hardened, Schema-Constrained Text-to-SQL Engine
To query 90+ years of Academy Awards records without risking SQL injection or LLM hallucinations:
* **Dedicated SQL Generator:** An isolated, zero-temperature model converts natural-language queries into SQLite `SELECT` statements against the `oscars_nominations` schema.
* **Multi-Layer Defensive Guardrails:**
  * Strict syntax and DDL/DML rejection (`DROP`, `INSERT`, `ALTER`, `VACUUM`, `PRAGMA`, etc.).
  * Multi-statement and semicolon blocking.
  * Schema lock requiring explicit queries against `oscars_nominations`.
  * Multi-provider payload normalization (handling Gemini nested dictionary/thought payloads vs. Ollama bare strings).
  * Executed over explicit read-only SQLite URIs (`file:...mode=ro`).

### 3. Stateful Human-in-the-Loop (HITL) Execution
State mutations (modifying TMDb account watchlists) require explicit user sign-off:
* Utilizes LangGraph's `interrupt()` primitive inside mutating tools.
* Streamlit surfaces confirmation modals with pending action payloads (`film_title`, `tmdb_id`, `action`).
* Graph resumes execution strictly upon explicit user authorization via `Command(resume="yes"|"no")`.

### 4. Streaming Token Isolation & Metadata Filtering
Standard multi-LLM workflows risk leaking internal chain-of-thought or raw SQL into the user interface during token streaming (`stream_mode="messages"`).
* Engineered a generator filter checking `metadata["langgraph_node"] == "chat_node"`.
* Suppresses intermediate tool-level LLM invocations (e.g., Text-to-SQL generation spans) while providing real-time UX status updates for active tools.

---

## 📊 MLOps, Observability & Telemetry

```text
   +-------------------+          Scrapes (5s)          +-------------------+
   |  KinoAgent App    | <----------------------------- |    Prometheus     |
   | (Port 8000 Metric |                                | (Metrics Storage) |
   |  Daemon Thread)   |                                +---------+---------+
   +-------------------+                                          |
                                                                  | PromQL Queries
                                                                  v
   +-------------------+       Experiment Tracking      +-------------------+
   |      MLflow       | <----------------------------- |      Grafana      |
   | (Golden Dataset   |   (Tracks SQL validity, tool   | (p50/p95 Latency, |
   |  Eval Logging)    |    routing accuracy & latency) |  Error Rates)     |
   +-------------------+                                +-------------------+
```

* **Prometheus Metrics Exporter:** Background daemon thread exposing real-time operational metrics at `/metrics` (port 8000):
  * `kinoagent_tool_calls_total` (labeled by `tool_name` and `status`).
  * `kinoagent_tool_latency_seconds` (Histogram with custom buckets: $0.05\text{s}$ to $120.0\text{s}$ for calculating $P_{50}, P_{95}, P_{99}$).
  * `kinoagent_sql_generation_failures_total` (Tracks malicious, malformed, or rejected SQL).
  * `kinoagent_chat_node_latency_seconds` (Monitors router decision time).
* **MLflow Tracking:** Logs model evaluation parameters, routing accuracy, SQL correctness, and evaluation artifacts across versions.
* **LangSmith Observability:** Distributed span and trace monitoring across every graph transition, tool call, and token chunk.

---

## 🚦 CI/CD Pipeline & Automated Model Evaluation Gates

To prevent model drift and prompt regressions, KinoAgent integrates a **Model Evaluation Gate** directly into GitHub Actions:

```text
git push / PR
      │
      ▼
┌──────────────────────────────────────────────┐
│  GitHub Actions: test-and-evaluate (CI Gate) │
│  ├── Python 3.11 Environment Setup           │
│  ├── Dependency Validation                   │
│  └── python -m evals.run_evals               │
│        ├── Golden Dataset (Queries & Tools)  │
│        ├── Routing Accuracy Scoring          │
│        ├── SQL Safety & Schema Verification  │
│        └── Latency Profiling                 │
└──────────────────────┬───────────────────────┘
                       │
                  Green Passed?
                 /             \
               YES              NO ──► Abort PR & Alert
                │
                ▼
┌──────────────────────────────────────────────┐
│   GitHub Actions: deploy-to-render (CD)      │
│   └── Validated Webhook Curl Trigger         │
└──────────────────────────────────────────────┘
```

* **Golden Evaluation Dataset:** Runs multi-intent assertions testing vector retrieval, Text-to-SQL generation, live API tools, and web searches.
* **Automated CD Delivery:** Upon merging verified code into `main`, GitHub Actions triggers a Render Deploy Hook to roll out the updated container.

---

## 🛠️ Tech Stack

| Domain | Technologies |
|---|---|
| **Agent Framework** | LangGraph, LangChain Core, Pydantic |
| **LLM Backends** | Google Gemini (`gemini-2.5-flash`, `gemini-3.1-flash-lite`), Ollama (`qwen3:4b`), Groq |
| **Vector DB & Search**| ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`), SQLite |
| **Data & APIs** | TMDb API v3, Kaggle Oscars Dataset, Letterboxd Exports, Tavily API |
| **Monitoring & MLOps**| Prometheus, Grafana, MLflow, LangSmith |
| **Frontend** | Streamlit (Custom token streaming, Session State persistence, HITL UI) |
| **DevOps & Cloud** | Docker (multi-stage build), GitHub Actions CI/CD, Render Cloud |

---

## 🔬 Systems Engineering Case Study: Memory Optimization & Cloud OOM

During cloud deployment to Render's free tier (512 MB RAM ceiling), KinoAgent encountered an immediate Out-Of-Memory (OOM) kernel kill:

### Root Cause Analysis
* Python 3.11 Base + Streamlit: `~200 MB`
* C++ PyTorch CPU Shared Libraries: `~220 MB`
* ChromaDB Rust Bindings + Dual HNSW Caches: `~120 MB`
* In-Memory `all-MiniLM-L6-v2` SentenceTransformer Model: `~100 MB`
* Background Prometheus Telemetry Thread: `~40 MB`
* **Total RSS at Startup:** `~680 MB` $\rightarrow$ **Exceeded 512 MB Free Tier**

### Architectural Solutions & Trade-Offs

| Approach | Architecture Change | RAM Impact | Trade-Off |
|---|---|---|---|
| **A. Offload Embeddings to Managed Cloud API** | Replace local `sentence-transformers` with Google Gemini / Vertex Embeddings; strip `torch` from container. | Reduces image RSS to **~180 MB** (fits comfortably in 512 MB). | Introduces network latency per embedding and external API token cost. |
| **B. Vertical Infrastructure Scaling** | Deploy container to Render Starter / AWS EC2 with configured Swap Space. | Accommodates local PyTorch execution (`1–2 GB` overhead). | Compute infrastructure hosting cost ($7/mo or cloud instance). |

---

## 💻 Local Setup & Development

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/KinoAgent.git
cd KinoAgent

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
TMDB_V3_API_KEY=your_tmdb_api_key
TMDB_ACCOUNT_ID=your_tmdb_account_id
TMDB_SESSION_ID=your_tmdb_session_id
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run Locally
```bash
streamlit run src/kinoagent/app.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with KinoAgent.  
Prometheus telemetry metrics are exposed concurrently on [http://localhost:8000/metrics](http://localhost:8000/metrics).

### 4. Run Evaluation Suite
```bash
python -m evals.run_evals
```

### 5. Run via Docker
```bash
docker build -t kinoagent:latest .
docker run -p 8501:8501 -p 8000:8000 --env-file .env kinoagent:latest
```

---

## 📈 Roadmap

- [ ] Transition vector embedding pipeline to lightweight hosted API embeddings to maintain `<200 MB` container footprint.
- [ ] Implement self-correcting SQL execution retry loop with database feedback.
- [ ] Add composite recommendation subgraph with multi-step candidate filtering and re-ranking.
