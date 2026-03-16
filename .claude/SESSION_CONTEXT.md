# Session Context — Churn Prediction System

Last updated: 2026-03-16

## Where we are

**Completed through Phase 6.** Next session (2026-03-17): tackle class imbalance, then **Phase 7: Batch scoring pipeline + business retention outputs.**

## Completed Phases

| Phase | What | Commit |
|-------|------|--------|
| Phase 0 | Project scaffolding with uv | `4633d2e` |
| Phase 1 | Synthetic data generator (5000 customers, 16 columns) | `7526b53` |
| Phase 2 | Data validation layer + churn rate cap (8%) | `c444768` |
| Phase 3 | Feature engineering (6 derived features, 22 total columns) | `0d0a477` |
| Phase 4 | Champion model (Logistic Regression, ROC-AUC 0.70) + pipelines | `cd9c176` |
| Phase 5 | Challenger model (Random Forest) + promotion policy | `4c153ac` |
| Phase 6 | Threshold optimization + simplified ROC-AUC-only promotion | `9ddbc5a` |

## Current State

### What works end-to-end
```bash
uv run python -m services.data_generator.run        # Generate data
uv run python -m pipelines.validate_data             # Validate
uv run python -m pipelines.build_dataset             # Feature engineer
uv run python -m pipelines.train_pipeline            # Train both models, compare on ROC-AUC, promote
uv run pytest                                        # 77 tests pass
```

### Key metrics
- Churn rate: 5.1% (~250 churners out of 5000)
- Champion (LR): ROC-AUC 0.6988
- Challenger (RF): ROC-AUC 0.6524

### Design decisions made this session
- Promotion policy uses ROC-AUC only (threshold-independent)
- Threshold optimization deferred to scoring time (recall strategy, min_recall=0.65)
- Moved threshold optimization from Tier 3 to Tier 1

## Open item: Class Imbalance

We tried `class_weight="balanced"` on both models. Results:
- Champion ROC-AUC dropped: 0.6988 → 0.6965
- Challenger ROC-AUC dropped: 0.6524 → 0.6401
- Probabilities spread to full 0-1 range (better calibrated) but ranking quality got worse
- **Reverted** — unweighted models rank better on this dataset

Techniques still to try next session:
1. **SMOTE** (Synthetic Minority Over-sampling) — generates synthetic churner samples by interpolating between existing ones. Requires `imbalanced-learn` dependency. May help RF more than LR.
2. **Increase synthetic data size** — generate 10,000-20,000 customers instead of 5,000 to give models more churner examples to learn from. Simplest approach.
3. **Tune class_weight with custom ratios** — instead of "balanced" (which uses 1/frequency), try intermediate weights like {0: 1, 1: 5} or {0: 1, 1: 10}.
4. **Feature engineering improvements** — better features may matter more than resampling. Look at interaction features or nonlinear transforms.
5. **Gradient Boosting challenger** — XGBoost/LightGBM with `scale_pos_weight` handles imbalance natively and often outperforms RF on tabular data.

Note: The real bottleneck may be synthetic data quality, not class imbalance. With only ~250 churners and synthetic patterns, there's limited signal to learn from.

## What's next — Phase 7 (after class imbalance work)

**Batch scoring pipeline + business retention outputs:**
- `services/scoring/scorer.py` — load champion model, apply threshold optimization (recall strategy), score customers, assign risk bands
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
  ├── training/        ✅ features/, models/ (champion + challenger), promotion.py, evaluation.py
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
