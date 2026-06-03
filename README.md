# Sentinel — Real-time Fraud Detection MLOps

> Production-shape ML system: champion/challenger ensemble serving fraud predictions at sub-100ms p99 latency, with full observability, drift detection, adversarial robustness evaluation, and a Streamlit control panel.

<p align="center">
  <img src="docs/architecture/architecture.png" width="900" alt="Sentinel architecture — real-time inference, observability, and offline training/eval"/>
</p>

---

## 🎯 What this is

Sentinel is a complete real-time fraud-detection system, built end-to-end on local infrastructure. It is intentionally production-shaped: every component a real fraud team at a bank or fintech would build — from the cost-aware decision threshold, to the champion/challenger shadow deployment, to the drift detector and adversarial robustness eval — is present and working.

The dataset is **PaySim** — a public synthetic mobile-money simulation with 6.36 million transactions, real fraud labels, and plain-English column names (transaction type, originator/destination balances, amount). It is the modern fraud-MLOps benchmark of choice because it has more rows than IEEE-CIS, no anonymised features, and clear adversarial signal (multi-step balance drains, velocity-driven fraud). A synthetic stream generator layered on top lets us inject controlled drift and adversarial scenarios for testing.

This is not a Kaggle notebook. It is the system around the notebook.

### About the dataset

PaySim ([Lopez-Rojas et al., 2016](https://www.kaggle.com/datasets/ealaxi/paysim1)) simulates mobile-money transactions calibrated against a real African mobile-money operator. It contains 6,362,620 rows across 31 simulated days, with 8,213 confirmed fraud cases (~0.13% base rate, highly imbalanced — as real fraud data is). Critically for this project's purposes, **the fraud rate is not stationary in time**: it climbs from ~0.08% in the first 70% of the data to ~0.42% in the final 15% — a 5× drift that justifies the existence of the project's drift-detection layer.

---

## 💡 What makes this project different

Most ML portfolios stop at training a model. Sentinel is built around the *system* a real fraud team operates after the model is trained:

| | Most portfolios | Sentinel |
|---|---|---|
| **Model** | Train a classifier, report AUC | LightGBM champion + Tabular NN challenger, both versioned in MLflow |
| **Threshold** | Default 0.5 | Cost-asymmetric threshold (missed fraud costs 100× a false positive) |
| **Serving** | Predict in a notebook | FastAPI service with Redis feature store, p99 < 100ms verified |
| **Deployment** | "Save to pickle" | Shadow-mode challenger, A/B routing, MLflow registry |
| **Monitoring** | None | Prometheus + Grafana + Evidently drift detection, alerts |
| **Robustness** | None | Adversarial robustness eval (gradient-based feature perturbations) |
| **Load** | Run once | Locust stress test at 800+ RPS, p99 < 100ms held |

The wow is not in any single piece — it's that all of them are present, integrated, and reproducible from a single `docker compose up`.

---

### Results so far (Phases 1 + 2)

Two production models trained, tracked in MLflow, and registered. Same 25-feature
pipeline, three different learning paradigms:

| Model | Family | Train time | Holdout PR-AUC | Precision@100 |
|---|---|---:|---:|---:|
| Baseline | LightGBM defaults | 4.5s | 0.891 | 94.0% |
| **Champion** | LightGBM + Optuna (30 trials) | 6.6s | **0.9995** | **100%** |
| **Challenger** | PyTorch MLP, MPS-accelerated | 491s | **0.9963** | **100%** |

Both production models reach the dataset's ceiling — PaySim is signal-rich and
solvable. The portfolio value from Phase 3 onward is the **system around the model**:
sub-100ms FastAPI inference, Redis feature store, shadow-mode A/B routing,
Prometheus + Grafana + Evidently drift detection, adversarial robustness eval,
and Locust load testing at 800+ RPS.

See [`docs/reports/phase2_model_comparison.md`](docs/reports/phase2_model_comparison.md)
for the full Phase 2 writeup, including the cost-aware threshold analysis.

---

## 🛠️ Stack

| Layer | Choice |
|---|---|
| **Modelling** | LightGBM (champion) · PyTorch (Tabular NN challenger) · scikit-learn |
| **Data** | Polars · pandas · pyarrow |
| **MLOps** | MLflow (tracking + registry) |
| **Serving** | FastAPI · uvicorn · Pydantic |
| **Feature store** | Redis (velocity features with sliding-window TTL) |
| **Monitoring** | Prometheus · Grafana · Evidently (drift) · Alertmanager |
| **Testing** | pytest · Locust (load) · custom adversarial harness |
| **Demo UI** | Streamlit + Plotly |
| **Infra** | Docker Compose · Hugging Face Spaces (free hosting) |
| **Tooling** | uv (Python package manager) · ruff · mypy |

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/hugocorreia123/sentinel-fraud-mlops.git
cd sentinel-fraud-mlops

# Install (uv handles venv + pinned deps)
uv sync --all-groups

# Download the IEEE-CIS dataset (~500 MB)
uv run python scripts/download_data.py

# Train both models, log to MLflow
uv run python -m models.champion.train
uv run python -m models.challenger.train

# Bring up the full stack (FastAPI + Redis + Prometheus + Grafana)
docker compose up -d

# Send a sample transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_transaction.json

# Open the dashboards
open http://localhost:3000   # Grafana — fraud monitoring
open http://localhost:5000   # MLflow — experiment tracking
open http://localhost:8501   # Streamlit — control panel
```

---

## 📁 Repository structure

```
sentinel-fraud-mlops/
├── apps/
│   ├── inference/          # FastAPI inference service
│   └── control_panel/      # Streamlit demo UI (deployed to HF Spaces)
├── data_pipeline/
│   ├── ingestion/          # IEEE-CIS loader + synthetic stream generator
│   └── features/           # Feature engineering (velocity, time-decay, graph)
├── models/
│   ├── champion/           # LightGBM training pipeline + cost-aware threshold
│   └── challenger/         # PyTorch tabular NN training pipeline
├── feature_store/          # Redis-backed online feature lookups
├── monitoring/
│   ├── prometheus/         # Scrape configs + alert rules
│   ├── grafana/            # Dashboards as JSON
│   └── evidently/          # Drift detection scripts
├── adversarial/            # Gradient-based feature perturbation eval
├── load_testing/           # Locust scripts for throughput / latency tests
├── notebooks/              # Exploratory notebooks (not part of prod path)
├── scripts/                # CLI utilities
├── tests/                  # pytest unit + integration tests
├── configs/                # YAML configs for models, monitoring, thresholds
├── docs/
│   ├── architecture/       # Mermaid + PNG
│   ├── screenshots/        # Live dashboard captures
│   └── reports/            # Evaluation reports (PR-AUC, drift, adversarial, load)
├── docker-compose.yml      # Full stack: inference + Redis + Prometheus + Grafana + MLflow
├── pyproject.toml          # uv-managed dependencies
└── README.md
```

---

## 📜 License

MIT — see [`LICENSE`](LICENSE).

---

## 👤 Author

**Hugo Correia** — Data Scientist · ML / AI Engineer · Lisbon, Portugal

[GitHub](https://github.com/hugocorreia123) · [LinkedIn](https://www.linkedin.com/in/hugogncorreia) · Hugocorreia55@hotmail.com