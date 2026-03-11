Below is the **complete `CLAUDE_CODE.md` file** including everything we discussed:

* Rules of engagement
* 3 production tiers
* Airflow orchestration
* Champion / Challenger framework
* Data + prediction drift monitoring
* Backtesting
* Business outputs
* **uv as the exclusive dependency manager**

You can copy this directly into a file called:

```
CLAUDE_CODE.md
```

in the root of your project.

---

# CLAUDE_CODE.md — Production-Grade Batch ML Churn Prediction System

Owner: Pierre

Goal: Build a **production-grade batch churn prediction platform** that generates synthetic customer data, trains and compares models, monitors drift, orchestrates pipelines with Airflow, and produces business-ready churn scoring outputs.

This file defines the **mandatory rules and structure** for the coding agent implementing this system.

---

# 0) Rules of Engagement (MANDATORY)

These rules override everything else.

## 1) Step-by-step only

* Never produce the whole project at once.
* Work in **small, verifiable increments**.
* Each step must be runnable/testable in **< 10 minutes**.

---

## 2) Teaching Mode

For every step include:

* What we are building
* Why it matters
* What files we create/modify
* How to run
* What to expect
* Common pitfalls

Assume I am **technical but learning ML system design**.

Explain decisions clearly but concisely.

---

## 3) Output format per step (STRICT)

Every step must include exactly these sections:

1. Goal
2. Concepts (what I learn)
3. Plan
4. Changes (files)
5. Code (only files changed)
6. How to run
7. How to test
8. Stop point

---

## 4) Don’t move on without confirmation

After every step stop and wait for me to reply with:

```
continue
```

or a question/fix request.

---

## 5) Keep changes minimal

* Prefer the smallest number of files.
* Avoid refactoring until the system works.
* Avoid introducing unnecessary tools.

---

# 1) Microservices Architecture (MANDATORY)

This project follows a **microservices design pattern**.

All business logic lives inside `services/`. Each service is self-contained.

---

## Services

There are **4 services**:

| Service | Responsibility |
|---------|---------------|
| `services/data_generator` | Synthetic data generation, data validation |
| `services/training` | Feature engineering, model training, evaluation, promotion |
| `services/scoring` | Batch scoring, business output generation |
| `services/monitor` | Data drift, prediction drift, backtesting |

---

## Rules

1. **All business logic must live inside a service.** No business logic in `pipelines/`, `dags/`, or root-level directories.
2. **Each service owns its domain.** Models, features, evaluation — all belong to `services/training/`. Drift logic belongs to `services/monitor/`.
3. **No root-level `models/` or `features/` directories.** These are absorbed into their owning service.
4. **Services expose a public API** via their `__init__.py` or top-level modules. External code imports from the service package, not from internal files.
5. **Pipelines are thin orchestrators.** `pipelines/*.py` import from services and wire together I/O. They contain no ML logic, no feature transforms, no model code.
6. **DAGs are thinner still.** `dags/*.py` call pipeline scripts. They contain no business logic and no direct service imports.
7. **Services may import from other services** when needed (e.g., scoring imports the trained model from training artifacts), but circular dependencies are forbidden.

---

## Service internal structure

Each service follows this pattern:

```
services/<service_name>/
├── __init__.py          # Public API exports
├── <core_modules>.py    # Business logic
└── run.py               # CLI entry point (optional)
```

Larger services may have sub-packages:

```
services/training/
├── __init__.py
├── features/
│   ├── __init__.py
│   └── feature_engineering.py
├── models/
│   ├── __init__.py
│   ├── baseline_model.py
│   ├── challenger_model.py
│   └── evaluation.py
├── promotion.py
├── calibration.py
└── run.py
```

---

# 2) Python Environment Management (MANDATORY)

This project is **exclusively managed using** uv.

No other dependency managers are allowed.

---

## Allowed

* uv
* pyproject.toml
* uv.lock

---

## Forbidden

* requirements.txt
* pip install
* poetry
* conda
* pipenv

---

## Running commands

All commands must run with:

```
uv run
```

Examples:

```
uv run python pipelines/train_pipeline.py
uv run pytest
uv run airflow scheduler
uv run airflow webserver
```

---

## Installing dependencies

Dependencies must be added using:

```
uv add <package>
```

Examples:

```
uv add pandas
uv add scikit-learn
uv add mlflow
uv add apache-airflow
```

Dev dependencies:

```
uv add --dev pytest
uv add --dev ruff
```

---

## Lock file

The repository must maintain:

```
uv.lock
```

This ensures reproducible builds.

---

## Docker standard

Docker must install dependencies with uv.

Example:

```
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync
```

---

## CI/CD standard

CI pipelines must use uv:

```
uv sync
uv run pytest
```

---

# 2) Project Objective

Build a **batch churn prediction ML platform** that simulates a real production pipeline.

The system must:

* generate synthetic customer data
* support **time-aware churn prediction**
* build datasets
* train **champion and challenger models**
* log experiments to MLflow
* perform batch scoring
* produce business-ready outputs
* monitor **data drift**
* monitor **prediction drift**
* support **backtesting**
* orchestrate workflows with **Airflow**

---

# 3) Tiered Production Scope

The project has **3 maturity tiers**.

---

# Tier 1 — Production-Like Core

Required features:

1. Synthetic churn dataset generator
2. Time-based dataset logic
3. Data validation checks
4. Feature engineering
5. Champion model (Logistic Regression)
6. Challenger model (Random Forest / Gradient Boosting)
7. Champion vs challenger comparison
8. Explicit promotion policy
9. Batch scoring pipeline
10. Business-ready churn outputs
11. MLflow experiment tracking
12. Docker Compose environment
13. Structured JSON logging
14. pytest test suite
15. GitHub Actions CI

---

# Tier 2 — Production Maturity

Add:

1. Backtesting across time windows
2. Segment-level evaluation
3. Data drift monitoring
4. Prediction drift monitoring
5. Separate training vs scoring datasets
6. Airflow orchestration DAGs
7. Baseline training statistics storage
8. Drift reporting artifacts

---

# Tier 3 — Elite Production Patterns

Add:

1. Calibration analysis
2. Threshold optimization
3. Promotion gating rules
4. Drift history tracking
5. Performance monitoring over time
6. Explainability artifacts
7. Retention policy simulation
8. Reproducible pipeline runner
9. Backtest dashboards or summaries

---

# 4) Airflow Orchestration (MANDATORY)

Airflow orchestrates the pipeline.

Airflow must **not contain business logic**.
It should call scripts in `/pipelines`.

---

## Required DAGs

* `generate_data_dag.py`
* `build_dataset_dag.py`
* `train_models_dag.py`
* `score_customers_dag.py`
* `drift_monitoring_dag.py`

Later:

* `backtesting_dag.py`
* `full_pipeline_dag.py`

---

# 5) High-Level Architecture

```
Synthetic Data
     ↓
Dataset Builder
     ↓
Feature Engineering
     ↓
Champion + Challenger Training
     ↓
Model Comparison + Promotion Policy
     ↓
MLflow Logging
     ↓
Batch Scoring
     ↓
Business Outputs
     ↓
Drift Monitoring
```

Airflow orchestrates the pipeline steps.

---

# 6) Repository Structure

```
/project-root
├── pyproject.toml
├── uv.lock
├── services
│   ├── data_generator/           # Service: synthetic data + validation
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   └── run.py
│   ├── training/                 # Service: features, models, evaluation
│   │   ├── __init__.py
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   └── feature_engineering.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── baseline_model.py
│   │   │   ├── challenger_model.py
│   │   │   └── evaluation.py
│   │   ├── promotion.py
│   │   ├── calibration.py
│   │   └── run.py
│   ├── scoring/                  # Service: batch scoring + business output
│   │   ├── __init__.py
│   │   ├── scorer.py
│   │   ├── retention_actions.py
│   │   └── run.py
│   └── monitor/                  # Service: drift + backtesting
│       ├── __init__.py
│       ├── data_drift.py
│       ├── prediction_drift.py
│       ├── backtester.py
│       └── run.py
├── pipelines                     # Thin orchestrators (no business logic)
│   ├── build_dataset.py
│   ├── train_pipeline.py
│   ├── score_pipeline.py
│   ├── drift_pipeline.py
│   ├── backtest_pipeline.py
│   └── validate_data.py
├── dags                          # Airflow DAGs (call pipelines only)
│   ├── generate_data_dag.py
│   ├── build_dataset_dag.py
│   ├── train_models_dag.py
│   ├── score_customers_dag.py
│   ├── drift_monitoring_dag.py
│   ├── backtesting_dag.py
│   └── full_pipeline_dag.py
├── data
│   ├── raw
│   ├── processed
│   ├── scoring
│   ├── scored
│   ├── drift_reports
│   └── backtests
├── model_artifacts
│   ├── champion
│   └── challenger
├── monitoring
│   ├── baseline_stats.json
│   └── promotion_policy.json
├── tests
│   ├── test_generator.py
│   ├── test_validation.py
│   ├── test_features.py
│   ├── test_training.py
│   ├── test_scoring.py
│   └── test_drift.py
├── docker-compose.yml
├── .env.example
├── README.md
└── CLAUDE_CODE.md
```

---

# 7) Synthetic Data Requirements

The generator must simulate realistic churn patterns.

Required datasets:

### Customers

* customer_id
* signup_date
* country
* age
* subscription_plan
* monthly_price
* acquisition_channel
* tenure_months

---

### Customer activity

* monthly_logins
* avg_session_minutes
* support_tickets_last_30d
* payment_failures_last_90d
* days_since_last_activity
* plan_changes_last_6m

---

### Labels

```
churned_30d
```

---

### Generator expectations

Synthetic logic should reflect:

* higher churn with inactivity
* higher churn with billing issues
* higher churn for low tenure
* lower churn with engagement

Generator must support:

* configurable dataset sizes
* reproducible random seeds
* future scoring batches

---

# 8) Time-Based Dataset Logic (MANDATORY)

Explicit windows must exist:

Observation window
Label window
Training period
Validation period
Scoring period

Example:

```
Observation window: last 90 days
Label window: next 30 days
```

Random splits are **not sufficient**.

---

# 9) Data Validation Layer

Validation checks must include:

* required files exist
* required columns exist
* row counts > 0
* churn label ∈ {0,1}
* no duplicate `customer_id + snapshot_date`
* probabilities ∈ [0,1]

Pipelines must **fail loudly** on validation errors.

---

# 10) Feature Engineering

Location:

```
services/training/features/feature_engineering.py
```

Example engineered features:

* engagement_score
* billing_risk_score
* support_burden_score
* inactivity_flag
* tenure_bucket
* price_to_usage_ratio

The same transformations must be used for training and scoring.

---

# 11) Champion / Challenger Framework

Champion:

```
LogisticRegression
```

Challenger:

```
RandomForest
or
GradientBoosting
```

Comparison artifact:

```
model_comparison.json
```

Promotion policy stored in:

```
monitoring/promotion_policy.json
```

Example rule:

* challenger ROC-AUC ≥ champion + 0.01
* recall ≥ minimum threshold

Otherwise champion remains.

---

# 12) Batch Scoring

Output:

```
data/scored/churn_scores.parquet
```

Required columns:

* customer_id
* snapshot_date
* churn_probability
* churn_prediction
* risk_band
* model_name

---

# 13) Business Action Output

Retention teams require actionable outputs.

Create:

```
data/scored/retention_actions.parquet
```

Columns:

* customer_id
* churn_probability
* risk_band
* recommended_action
* priority_rank

---

# 14) Drift Monitoring

Two types:

### Data drift

Compare training baseline vs scoring dataset.

Metrics:

* PSI
* categorical distribution shift

Output:

```
drift_report.json
```

---

### Prediction drift

Monitor:

* churn probability distribution
* risk band distribution
* mean predicted churn

Output:

```
prediction_drift_report.json
```

---

# 15) Backtesting

Backtesting must use rolling windows.

Output:

```
data/backtests/backtest_results.parquet
```

Columns:

* train_start
* train_end
* validation_start
* validation_end
* ROC-AUC
* Precision
* Recall
* F1

---

# 16) Segment Evaluation

Evaluate model by:

* subscription_plan
* country
* tenure_bucket
* acquisition_channel

Output:

```
segment_evaluation.parquet
```

---

# 17) Testing

pytest coverage required for:

* synthetic data
* validation
* feature engineering
* training pipeline
* scoring
* drift monitoring
* backtesting

---

# 18) CI/CD

GitHub Actions must run:

```
uv sync
uv run pytest
```

---

# 19) Definition of Done

Tier 1 complete when:

* synthetic data works
* dataset builds
* champion + challenger train
* MLflow logs runs
* scoring pipeline works
* retention outputs produced
* tests pass

Tier 2 complete when:

* Airflow DAGs run
* drift monitoring works
* backtesting works
* segment evaluation works

Tier 3 complete when:

* calibration analysis exists
* threshold optimization exists
* drift history tracked
* explainability outputs produced

---

# 20) Agent Workflow

The coding agent must:

* work step-by-step
* keep steps runnable in <10 minutes
* avoid unnecessary refactors
* follow uv for all Python operations
* follow the phased implementation plan
* start with **Phase 0: project scaffolding**

---

END

```

---

If you want, I can also give you **the perfect kickoff prompt for Claude Code for this churn project** (like we did for the fraud system) so the first step starts with **clean scaffolding + uv setup + Airflow skeleton**.
```
