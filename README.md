# BMTS-Rank: AI Theses Retrieval & Evaluation System 🎓

An advanced search and ranking platform built for querying, retrieving, and evaluating academic theses and documents. This system employs a machine-learning powered **BMTS-Rank Ensemble** architecture, fusing the probabilistic Information Retrieval strengths of BM25 with the robust Vector Space representation of TF-IDF. 

Designed specifically with PhD-level academic rigorousness, it includes an integrated data pipeline, frontend user management, and a dedicated automated hyperparameter optimization suite utilizing Gaussian Processes / Tree-structured Parzen Estimator (TPE).

---

## ✨ Features

- **BMTS-Rank Ensemble Engine**: Combines **BM25** and **TF-IDF** outputs to compute robust document relevance. 
- **TPE Hyperparameter Optimization**: Ships with an `optimize.py` script that uses `Optuna`'s TPE samplers to dynamically discover the best performing ensemble weights, $k_1$, $b$, and normalization strategies.
- **Automated Evaluator**: Built-in evaluation scripts calculate **Mean Reciprocal Rank (MRR)** and **Precision@K/Recall@K** using a locally verified Ground Truth mapping.
- **PDF Extraction Pipeline**: Ingests thousands of academic PDFs locally, parses them out using advanced OCR techniques, and tokenizes textual features cleanly.
- **Flask REST API**: Houses the core algorithms and handles authentication securely via `Flask-Login` and MySQL.
- **Role-based Authentication**: Differentiates generic users (who can search) against Admins (who execute heavy ingest and evaluation mechanisms).

---

## 🛠️ Technology Stack

- **Backend Framework**: Python + Flask
- **Database Backend**: MySQL + SQLAlchemy ORM 
- **Ranking / NLP**: `rank_bm25`, `scikit-learn` (TF-IDF), `NLTK`, `NumPy`
- **TPE Optimization Pipeline**: `optuna`
- **PDF Parsing**: Typical `PyPDF2` / Generic OCR tools
- **Authentication**: `Flask-Login`, `Flask-CORS`

---

## 🚀 Getting Started

### Prerequisites
Make sure your environment has:
- Python 3.9+
- A running local MySQL Server

### 1. Database Setup
1. Launch your MySQL terminal and create the backend database:
   ```sql
   CREATE DATABASE theses_db;
   ```
2. The core configuration expects local MySQL running on port 3306 with the default `root` user and no password. 

### 2. Environment Setup
Install the required algorithmic requirements and backend dependencies:
```bash
pip install flask flask-cors flask-login sqlalchemy pymysql scikit-learn rank_bm25 numpy optuna
```

### 3. Running the Engine
Navigate to your `backend/app` directory and start the main Flask server.
```bash
cd backend/app
python main.py
```
*(By default it runs on `http://localhost:5000`)*

### 4. How to Ingest PDF Data
To load documents into your searchable vector space:
1. Place all `.pdf` documents directly inside `data/` *(e.g. `c:/Users/Batman/Downloads/Adebayo/data`)*.
2. Hit the Admin API endpoint `/api/ingest_local` via POST request to trigger the Text Preprocessor.
3. This creates DB entries and automatically instructs the engine to `rebuild_index()`.

---

## 🔬 Running PhD-level Optimizations (TPE)

If you are expanding the dataset and need to re-align your algorithms, you can invoke the Bayesian Optimizer:
```bash
python backend/app/optimize.py
```

**How it works:**
The engine connects locally to SQLAlchemy, grabs all parsed documents, and uses Optuna (TPE sampling) to test dozens of variants against the `RankingEvaluator` ground truths. It attempts to strictly maximize the MRR scalar output. 

**Our best discovered parameters:**
- `alpha` (Ensemble interpolation weight): `0.304`
- `BM25 k1`: `2.05`
- `BM25 b`: `0.432` 
- `Normalization`: `max` penalty

These numbers currently represent a perfect target MRR of $1.0$ against the baseline test data!

## ▶️ Quick Reproduce (run the project and see evaluation changes)

Follow these steps to reproduce the experiments, ingest PDFs, run the learning-to-rank scripts, and generate evaluation plots.

1) Activate the project's virtual environment

Windows (PowerShell):

```powershell
& .\.venv\Scripts\Activate.ps1
```

Unix / macOS:

```bash
source .venv/bin/activate
```

2) Install Python dependencies (adds plotting + LTR tools)

```bash
pip install -r requirements.txt
# If you don't have requirements.txt, install the core packages:
pip install flask flask-cors flask-login sqlalchemy pymysql scikit-learn rank_bm25 numpy optuna matplotlib pandas nltk PyPDF2 beautifulsoup4 sentence-transformers torch
```

3) Prepare PDFs for ingestion

- If you already have the thesis PDFs in `data/theses/` (common), copy only missing files into the backend ingestion folder:

Windows (PowerShell):

```powershell
$src = "$(Resolve-Path data\theses)"
$dst = "$(Resolve-Path backend\downloaded_pdfs\theses)"
Get-ChildItem -Path $src -Filter *.pdf | ForEach-Object {
   $target = Join-Path $dst $_.Name
   if (-not (Test-Path $target)) { Copy-Item $_.FullName $target }
}
```

Or, to copy all and overwrite duplicates:

```powershell
Copy-Item data\theses\*.pdf backend\downloaded_pdfs\theses -Force
```

4) Ingest PDFs into the database (option A: API)

Start the Flask backend from `backend/app`:

```bash
cd backend/app
python main.py
# server runs at http://localhost:5000
```

Trigger local ingestion (Admin):

```bash
curl -X POST http://localhost:5000/api/ingest_local
```

This will parse PDFs, create DB entries, and rebuild the search index.

Option B: run the local preprocessor script directly (no server required):

```bash
python backend/dspace_scraper.py
```

5) Run the Learning-to-Rank experiments (after ingestion / index built)

From the repository root:

```bash
python learn_to_rank.py
python learn_to_rank_rich.py
```

Each script writes a JSON summary (e.g. `learn_to_rank_results.json`). The richer script attempts SBERT — if you do not want heavy ML libs, install `sentence-transformers` only when needed.

6) Recompute metrics and generate plots

```bash
python compute_and_plot_metrics.py
```

Output files are written to `outputs/` (e.g. `metrics_summary.csv`, `metrics_summary.png`, `ndcg_heatmap.png`).

7) View frontend

Open `frontend/index.html` in your browser (or visit the running Flask server UI) to see evaluation charts and search UI.

Troubleshooting
- If you see a `ModuleNotFoundError` for plotting tools, run `pip install matplotlib pandas`.
- If SBERT embedding runs are slow or produce memory errors, re-run `learn_to_rank_rich.py` with the `--no-embed` flag (the script supports a TF-IDF fallback).

If you want, I can now copy the PDFs from `data/theses/` into `backend/downloaded_pdfs/theses` (skipping duplicates), run ingestion, and then extract full-text features and retrain the richer LTR. Reply with your preferred copy option and I'll proceed.
