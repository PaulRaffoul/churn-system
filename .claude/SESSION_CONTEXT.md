# Session Context — Churn Prediction System

Last updated: 2026-03-16

## Where we are

**Completed through Phase 5.** Next up is **Phase 6: Batch scoring pipeline + business retention outputs.**

## Completed Phases

| Phase | What | Commit |
|-------|------|--------|
| Phase 0 | Project scaffolding with uv | `4633d2e` |
| Phase 1 | Synthetic data generator (5000 customers, 16 columns) | `7526b53` |
| Phase 2 | Data validation layer + churn rate cap (8%) | `c444768` |
| Phase 3 | Feature engineering (6 derived features, 22 total columns) | `0d0a477` |
| Phase 4 | Champion model (Logistic Regression, ROC-AUC 0.70) + pipelines | `cd9c176` |
| Phase 5 | Challenger model (Random Forest) + 3-gate promotion policy | `4c153ac` |

## Current State

### What works end-to-end
```bash
uv run python -m services.data_generator.run        # Generate data
uv run python -m pipelines.validate_data             # Validate
uv run python -m pipelines.build_dataset             # Feature engineer
uv run python -m pipelines.train_pipeline            # Train champion + challenger, compare, promote
uv run pytest                                        # 71 tests pass
```

### Artifacts produced
- `data/raw/churn_data.parquet` — 5000 customers, seed=42
- `data/processed/churn_processed.parquet` — 22 columns (16 raw + 6 engineered)
- `model_artifacts/champion/model.joblib` — trained LogisticRegression pipeline
- `model_artifacts/champion/metrics.json` — ROC-AUC 0.70
- `model_artifacts/challenger/model.joblib` — trained RandomForest pipeline
- `model_artifacts/challenger/metrics.json` — ROC-AUC 0.65
- `model_artifacts/model_comparison.json` — champion retained (RF didn't beat LR)

### Key metrics
- Churn rate: 5.1% (capped at 8% by validator)
- Champion ROC-AUC: 0.6988 (Logistic Regression)
- Challenger ROC-AUC: 0.6524 (Random Forest)
- Precision/Recall/F1: 0.0 at default 0.5 threshold (expected with 5% churn — needs threshold optimization in Tier 3)

### Promotion policy (churn-optimized)
Three gates — all must pass for promotion:
1. ROC-AUC: challenger >= champion + 0.01
2. Recall floor: >= 0.65 (catch at least 2/3 of churners — false negatives cost lifetime value)
3. Precision floor: >= 0.15 (flagged customers must be 3x more likely to churn than random)

Rationale: churn prediction favors recall over precision because missing a churner costs their entire lifetime value, while a false alarm costs only a retention offer.

## What's next — Phase 6

**Batch scoring pipeline + business retention outputs:**
- `services/scoring/scorer.py` — load champion model, score new customers, assign risk bands
- `services/scoring/retention_actions.py` — generate recommended actions per risk band
- `pipelines/score_pipeline.py` — thin orchestrator
- Output: `data/scored/churn_scores.parquet` + `data/scored/retention_actions.parquet`
- Tests in `tests/test_scoring.py`

After Phase 6, remaining Tier 1 items:
- MLflow experiment tracking
- Docker Compose
- Structured JSON logging
- GitHub Actions CI

## Architecture reminder

```
services/ (business logic)
  ├── data_generator/  ✅ generator.py, validator.py
  ├── training/        ✅ features/, models/ (champion + challenger), promotion.py
  ├── scoring/         ⬜ placeholder
  └── monitor/         ⬜ placeholder

pipelines/ (thin orchestrators)
  ├── validate_data.py   ✅
  ├── build_dataset.py   ✅
  ├── train_pipeline.py  ✅ (trains both models, compares, promotes)
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
