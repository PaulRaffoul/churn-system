# Session Context — Churn Prediction System

Last updated: 2026-03-16

## Where we are

**Completed through Phase 6 (threshold optimization).** Next up is **Phase 7: Batch scoring pipeline + business retention outputs.**

## Completed Phases

| Phase | What | Commit |
|-------|------|--------|
| Phase 0 | Project scaffolding with uv | `4633d2e` |
| Phase 1 | Synthetic data generator (5000 customers, 16 columns) | `7526b53` |
| Phase 2 | Data validation layer + churn rate cap (8%) | `c444768` |
| Phase 3 | Feature engineering (6 derived features, 22 total columns) | `0d0a477` |
| Phase 4 | Champion model (Logistic Regression, ROC-AUC 0.70) + pipelines | `cd9c176` |
| Phase 5 | Challenger model (Random Forest) + 3-gate promotion policy | `4c153ac` |
| Phase 5b | Churn-optimized promotion policy (recall 0.65, precision 0.15) | `70f077a` |
| Phase 6 | Threshold optimization (recall strategy, threshold ~0.06) | pending |

## Current State

### What works end-to-end
```bash
uv run python -m services.data_generator.run        # Generate data
uv run python -m pipelines.validate_data             # Validate
uv run python -m pipelines.build_dataset             # Feature engineer
uv run python -m pipelines.train_pipeline            # Train both models, optimize thresholds, compare, promote
uv run pytest                                        # 78 tests pass
```

### Artifacts produced
- `data/raw/churn_data.parquet` — 5000 customers, seed=42
- `data/processed/churn_processed.parquet` — 22 columns (16 raw + 6 engineered)
- `model_artifacts/champion/model.joblib` — trained LogisticRegression pipeline
- `model_artifacts/champion/metrics.json` — ROC-AUC 0.70, threshold 0.0626
- `model_artifacts/challenger/model.joblib` — trained RandomForest pipeline
- `model_artifacts/challenger/metrics.json` — ROC-AUC 0.65, threshold 0.0488
- `model_artifacts/model_comparison.json` — champion retained

### Key metrics (with optimized thresholds)
- Churn rate: 5.1% (capped at 8% by validator)
- Champion (LR): ROC-AUC 0.70, Precision 0.12, Recall 0.67, F1 0.20, Threshold 0.0626
- Challenger (RF): ROC-AUC 0.65, Precision 0.09, Recall 0.67, F1 0.17, Threshold 0.0488
- Threshold strategy: "recall" with min_recall=0.65 (prioritizes catching churners)

### Promotion policy (churn-optimized)
Three gates — all must pass for promotion:
1. ROC-AUC: challenger >= champion + 0.01
2. Recall floor: >= 0.65 (catch at least 2/3 of churners)
3. Precision floor: >= 0.15 (flagged customers 3x more likely to churn than random)

### Key design decision: threshold optimization moved to Tier 1
Threshold optimization was moved from Tier 3 to Tier 1 (before scoring) because with 5% churn rate, the default 0.5 threshold produces zero predictions. Scoring pipeline needs a working threshold to produce meaningful risk bands and retention actions.

## What's next — Phase 7

**Batch scoring pipeline + business retention outputs:**
- `services/scoring/scorer.py` — load champion model + threshold, score new customers, assign risk bands
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
  ├── training/        ✅ features/, models/ (champion + challenger), promotion.py, threshold optimization
  ├── scoring/         ⬜ placeholder
  └── monitor/         ⬜ placeholder

pipelines/ (thin orchestrators)
  ├── validate_data.py   ✅
  ├── build_dataset.py   ✅
  ├── train_pipeline.py  ✅ (trains both, optimizes thresholds, compares, promotes)
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
