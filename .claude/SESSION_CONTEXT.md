# Session Context — Churn Prediction System

Last updated: 2026-03-13

## Where we are

**Completed through Phase 4.** Next up is **Phase 5: Challenger model + promotion policy.**

## Completed Phases

| Phase | What | Commit |
|-------|------|--------|
| Phase 0 | Project scaffolding with uv | `4633d2e` |
| Phase 1 | Synthetic data generator (5000 customers, 16 columns) | `7526b53` |
| Phase 2 | Data validation layer + churn rate cap (8%) | `c444768` |
| Phase 3 | Feature engineering (6 derived features, 22 total columns) | `0d0a477` |
| Phase 4 | Champion model (Logistic Regression, ROC-AUC 0.70) + pipelines | `cd9c176` |

## Current State

### What works end-to-end
```bash
uv run python -m services.data_generator.run        # Generate data
uv run python -m pipelines.validate_data             # Validate
uv run python -m pipelines.build_dataset             # Feature engineer
uv run python -m pipelines.train_pipeline            # Train champion
uv run pytest                                        # 59 tests pass
```

### Artifacts produced
- `data/raw/churn_data.parquet` — 5000 customers, seed=42
- `data/processed/churn_processed.parquet` — 22 columns (16 raw + 6 engineered)
- `model_artifacts/champion/model.joblib` — trained LogisticRegression pipeline
- `model_artifacts/champion/metrics.json` — ROC-AUC 0.70

### Key metrics
- Churn rate: 5.1% (capped at 8% by validator)
- Champion ROC-AUC: 0.70
- Precision/Recall/F1: 0.0 at default 0.5 threshold (expected with 5% churn — needs threshold optimization in Tier 3)

## What's next — Phase 5

**Challenger model (Random Forest) + promotion policy:**
- `services/training/models/challenger_model.py` — Random Forest or Gradient Boosting
- `services/training/promotion.py` — compare champion vs challenger, promote if better
- `monitoring/promotion_policy.json` — already has rules defined (ROC-AUC +0.01, recall >= 0.50)
- `model_artifacts/challenger/` — save challenger artifacts
- `pipelines/train_pipeline.py` — update to train both models and compare
- `tests/test_training.py` — add challenger + promotion tests

After Phase 5, remaining Tier 1 items:
- Batch scoring pipeline (`services/scoring/`)
- Business retention outputs
- MLflow experiment tracking
- Docker Compose
- Structured JSON logging
- GitHub Actions CI

## Architecture reminder

```
services/ (business logic)
  ├── data_generator/  ✅ generator.py, validator.py
  ├── training/        ✅ features/, models/baseline_model.py — needs challenger + promotion
  ├── scoring/         ⬜ placeholder
  └── monitor/         ⬜ placeholder

pipelines/ (thin orchestrators)
  ├── validate_data.py   ✅
  ├── build_dataset.py   ✅
  ├── train_pipeline.py  ✅
  ├── score_pipeline.py  ⬜ placeholder
  ├── drift_pipeline.py  ⬜ placeholder
  └── backtest_pipeline.py ⬜ placeholder

dags/ (Airflow — all placeholder, Tier 2)
```

## User preferences

- Teaching mode: explain reasoning, concepts, and design decisions
- Step-by-step: don't skip ahead, wait for "continue"
- Strict output format: Goal, Concepts, Plan, Changes, Code, How to run, How to test, Stop point
- Push to GitHub after each phase
- Update README as we progress
- Uses uv exclusively (no pip, no conda)
