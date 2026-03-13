# Churn Prediction System

Production-grade batch ML churn prediction platform that generates synthetic customer data, trains and compares models, monitors drift, and produces business-ready churn scoring outputs.

## Architecture

```
services/
├── data_generator/   # Synthetic data generation + validation
├── training/         # Feature engineering, model training, evaluation
├── scoring/          # Batch scoring + business outputs (planned)
└── monitor/          # Drift monitoring + backtesting (planned)

pipelines/            # Thin orchestrators (no business logic)
dags/                 # Airflow DAGs (planned)
```

**3-layer design:** DAGs (scheduling) → Pipelines (orchestration) → Services (business logic)

## Setup

```bash
uv sync
```

## Pipelines

Run in order for a full end-to-end flow:

```bash
# 1. Generate synthetic customer data (5000 customers)
uv run python -m services.data_generator.run

# 2. Validate raw data
uv run python -m pipelines.validate_data

# 3. Build processed dataset (validate + feature engineering)
uv run python -m pipelines.build_dataset

# 4. Train champion model (Logistic Regression)
uv run python -m pipelines.train_pipeline
```

## Data Flow

```
Generator → data/raw/churn_data.parquet
    → Validator (schema, duplicates, churn rate cap)
    → Feature Engineering (6 derived features)
    → data/processed/churn_processed.parquet
    → Time-based split (80/20 by signup_date)
    → Train Logistic Regression
    → model_artifacts/champion/model.joblib
    → model_artifacts/champion/metrics.json
```

## Engineered Features

| Feature | Description |
|---------|-------------|
| `engagement_score` | Normalized logins + session time (0-1) |
| `billing_risk_score` | Payment failures + plan changes (0-1) |
| `support_burden_score` | Support tickets normalized (0-1) |
| `inactivity_flag` | 1 if inactive >14 days |
| `tenure_bucket` | short / medium / long |
| `price_to_usage_ratio` | Monthly price relative to usage |

## Tests

```bash
# Full suite (59 tests)
uv run pytest

# Individual modules
uv run pytest tests/test_generator.py -v
uv run pytest tests/test_validation.py -v
uv run pytest tests/test_features.py -v
uv run pytest tests/test_training.py -v
```

## Progress

### Tier 1 — Production-Like Core

- [x] Synthetic churn dataset generator
- [x] Data validation layer (schema, duplicates, churn rate cap 8%)
- [x] Feature engineering (6 derived features)
- [x] Champion model (Logistic Regression, ROC-AUC 0.70)
- [x] Time-based train/validation split
- [x] Build dataset pipeline
- [x] Train pipeline with evaluation metrics
- [ ] Challenger model (Random Forest)
- [ ] Champion vs challenger promotion policy
- [ ] Batch scoring pipeline
- [ ] Business-ready retention outputs
- [ ] MLflow experiment tracking
- [ ] Docker Compose environment
- [ ] Structured JSON logging
- [ ] GitHub Actions CI

### Tier 2 — Production Maturity

- [ ] Airflow orchestration DAGs
- [ ] Data drift monitoring
- [ ] Prediction drift monitoring
- [ ] Backtesting across time windows
- [ ] Segment-level evaluation

### Tier 3 — Elite Production Patterns

- [ ] Calibration analysis
- [ ] Threshold optimization
- [ ] Explainability artifacts
- [ ] Drift history tracking

## Tech Stack

- **Python 3.11+** with **uv** (exclusive dependency manager)
- **pandas** + **pyarrow** for data handling
- **scikit-learn** for modelling
- **pytest** for testing
