"""Synthetic churn dataset generator.

Generates realistic customer data with churn patterns:
- Higher churn with inactivity
- Higher churn with billing issues
- Higher churn for low tenure
- Lower churn with engagement
"""

import numpy as np
import pandas as pd


def generate_churn_dataset(
    n_customers: int = 5000,
    snapshot_date: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic churn dataset for a single snapshot date.

    Args:
        n_customers: Number of customers to generate.
        snapshot_date: The point-in-time date for this snapshot.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with customer demographics, activity, and churn label.
    """
    rng = np.random.default_rng(seed)
    snapshot = pd.Timestamp(snapshot_date)

    # --- Customer demographics ---
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]

    # Signup dates: between 1 and 48 months before snapshot
    days_since_signup = rng.integers(30, 48 * 30, size=n_customers)
    signup_dates = [snapshot - pd.Timedelta(days=int(d)) for d in days_since_signup]

    tenure_months = (days_since_signup / 30).astype(int)

    countries = rng.choice(
        ["US", "UK", "DE", "FR", "CA", "AU", "BR", "IN"],
        size=n_customers,
        p=[0.30, 0.15, 0.10, 0.10, 0.10, 0.08, 0.10, 0.07],
    )

    ages = rng.integers(18, 70, size=n_customers)

    plans = rng.choice(
        ["basic", "standard", "premium"],
        size=n_customers,
        p=[0.40, 0.35, 0.25],
    )

    price_map = {"basic": 9.99, "standard": 19.99, "premium": 39.99}
    monthly_prices = np.array([price_map[p] for p in plans])

    channels = rng.choice(
        ["organic", "paid_search", "social", "referral", "email"],
        size=n_customers,
        p=[0.25, 0.25, 0.20, 0.15, 0.15],
    )

    # --- Customer activity ---
    monthly_logins = rng.poisson(lam=12, size=n_customers).clip(0, 60)
    avg_session_minutes = rng.exponential(scale=15, size=n_customers).clip(0, 120).round(1)
    support_tickets_last_30d = rng.poisson(lam=0.8, size=n_customers).clip(0, 10)
    payment_failures_last_90d = rng.poisson(lam=0.3, size=n_customers).clip(0, 5)
    days_since_last_activity = rng.exponential(scale=7, size=n_customers).clip(0, 90).astype(int)
    plan_changes_last_6m = rng.poisson(lam=0.3, size=n_customers).clip(0, 4)

    # --- Churn probability (latent score) ---
    # Higher score = more likely to churn.
    # Bias term shifts the baseline so that a typical customer has ~5-7%
    # churn probability (sigmoid(-3.0) ≈ 4.7%).  Only customers with
    # genuinely bad signals accumulate enough score to churn.
    churn_score = np.full(n_customers, -3.0)

    # Inactivity drives churn
    churn_score += 0.02 * days_since_last_activity
    churn_score += -0.03 * monthly_logins
    churn_score += -0.01 * avg_session_minutes

    # Billing issues drive churn
    churn_score += 0.4 * payment_failures_last_90d

    # Low tenure drives churn
    churn_score += np.where(tenure_months < 3, 0.8, 0.0)
    churn_score += np.where(tenure_months < 6, 0.3, 0.0)

    # Support tickets signal frustration
    churn_score += 0.15 * support_tickets_last_30d

    # Plan changes signal uncertainty
    churn_score += 0.2 * plan_changes_last_6m

    # Premium customers churn less
    churn_score += np.where(np.array(plans) == "premium", -0.5, 0.0)
    churn_score += np.where(np.array(plans) == "basic", 0.3, 0.0)

    # Add noise
    churn_score += rng.normal(0, 0.3, size=n_customers)

    # Convert to probability via sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_score))

    # Sample binary label
    churned_30d = (rng.random(n_customers) < churn_prob).astype(int)

    # --- Assemble DataFrame ---
    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "snapshot_date": snapshot,
            "signup_date": signup_dates,
            "country": countries,
            "age": ages,
            "subscription_plan": plans,
            "monthly_price": monthly_prices,
            "acquisition_channel": channels,
            "tenure_months": tenure_months,
            "monthly_logins": monthly_logins,
            "avg_session_minutes": avg_session_minutes,
            "support_tickets_last_30d": support_tickets_last_30d,
            "payment_failures_last_90d": payment_failures_last_90d,
            "days_since_last_activity": days_since_last_activity,
            "plan_changes_last_6m": plan_changes_last_6m,
            "churned_30d": churned_30d,
        }
    )

    return df
