# Session Context — Churn Prediction System

Last updated: 2026-03-16

## Where we are

**Completed through Phase 6.** Next up is **Phase 7: Batch scoring pipeline + business retention outputs.**

## Completed Phases

| Phase | What | Commit |
|-------|------|--------|
| Phase 0 | Project scaffolding with uv | `4633d2e` |
| Phase 1 | Synthetic data generator (5000 customers, 16 columns) | `7526b53` |
| Phase 2 | Data validation layer + churn rate cap (8%) | `c444768` |
| Phase 3 | Feature engineering (6 derived features, 22 total columns) | `0d0a477` |
| Phase 4 | Champion model (Logistic Regression, ROC-AUC 0.70) + pipelines | `cd9c176` |
| Phase 5 | Challenger model (Random Forest) + promotion policy | `4c153ac` |
| Phase 6 | Threshold optimization + clean separation of concerns | pending |

## Current State

### What works end-to-end
```bash
uv run python -m services.data_generator.run        # Generate data
uv run python -m pipelines.validate_data             # Validate
uv run python -m pipelines.build_dataset             # Feature engineer
uv run python -m pipelines.train_pipeline            # Train both models, compare on ROC-AUC, promote
uv run pytest                                        # 77 tests pass
```

### Artifacts produced
- `data/raw/churn_data.parquet` — 5000 customers, seed=42
- `data/processed/churn_processed.parquet` — 22 columns (16 raw + 6 engineered)
- `model_artifacts/champion/model.joblib` — trained LogisticRegression pipeline
- `model_artifacts/champion/metrics.json` — ROC-AUC 0.70
- `model_artifacts/challenger/model.joblib` — trained RandomForest pipeline
- `model_artifacts/challenger/metrics.json` — ROC-AUC 0.65
- `model_artifacts/model_comparison.json` — champion retained

### Key metrics
- Churn rate: 5.1% (capped at 8% by validator)
- Champion (LR): ROC-AUC 0.6988
- Challenger (RF): ROC-AUC 0.6524

### Design decisions

**Separation of concerns for thresholds:**
- Training pipeline compares models on ROC-AUC only (threshold-independent)
- Scoring pipeline will apply threshold optimization (recall-targeted, min_recall=0.65) at deployment time
- This avoids the problem of threshold-optimized metrics being artificially equal between models

**Promotion policy:** challenger ROC-AUC >= champion ROC-AUC + 0.01

**Threshold optimization (available, used at scoring time):**
- `find_optimal_threshold()` in `services/training/models/evaluation.py`
- Two strategies: "f1" (balanced) and "recall" (churn-optimized)
- Scoring pipeline will use "recall" strategy with min_recall=0.65

## What's next — Phase 7

**Batch scoring pipeline + business retention outputs:**
- `services/scoring/scorer.py` — load champion model, apply threshold optimization (recall strategy), score new customers, assign risk bands
- `services/scoring/retention_actions.py` — generate recommended actions per risk band
- `pipelines/score_pipeline.py` — thin orchestrator
- Output: `data/scored/churn_scores.parquet` + `data/scored/retention_actions.parquet`
- Tests in `tests/test_scoring.py`

After Phase 7, remaining Tier 1 items:
- MLflow experiment tracking
- Docker Compose
- Structured JSON logging
- GitHub Actions CI

## Architecture reminder

```
services/ (business logic)
  ├── data_generator/  ✅ generator.py, validator.py
  ├── training/        ✅ features/, models/ (champion + challenger), promotion.py, evaluation.py (threshold optimization)
  ├── scoring/         ⬜ placeholder
  └── monitor/         ⬜ placeholder

pipelines/ (thin orchestrators)
  ├── validate_data.py   ✅
  ├── build_dataset.py   ✅
  ├── train_pipeline.py  ✅ (trains both, compares on ROC-AUC, promotes)
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
- Prefers recall-prioritized strategy for churn (not F1-balanced)
- Threshold optimization belongs at scoring time, not training time
- Promotion policy should be threshold-independent (ROC-AUC only)
