"""Central config: paths, seeds, product economics, column groups.

Single source of truth so notebooks/modules/tests agree. Import as:
    from src.config import PATHS, SEED, PRODUCT, set_seeds
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 20260605  # fixed for the whole project


def set_seeds(seed: int = SEED) -> None:
    """Seed every stochastic source we touch (numpy, python, hashing, lgbm)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # lightgbm takes its seed per-call via params (see src/models); nothing global.


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "dataset" / "dataset-compressed"
    dataset: Path = ROOT / "dataset"
    train: Path = ROOT / "dataset" / "dataset-compressed" / "train.csv"
    validation: Path = ROOT / "dataset" / "dataset-compressed" / "validation.csv"
    test: Path = ROOT / "dataset" / "dataset-compressed" / "test.csv"
    data_dictionary: Path = ROOT / "dataset" / "data_dictionary.csv"
    cohort_defs: Path = ROOT / "dataset" / "cohort_week_definitions.csv"
    intervention_queries: Path = ROOT / "dataset" / "intervention_queries.csv"
    submission_b_template: Path = ROOT / "dataset" / "submission_B_template.csv"
    expected_ids: Path = ROOT / "expected_ids"
    reports: Path = ROOT / "reports"
    artifacts: Path = ROOT / "artifacts"
    submissions: Path = ROOT / "submissions"


PATHS = Paths()


# --------------------------------------------------------------------------- #
# Loan product economics (fixed terms, from dataset/README.md)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Product:
    term_days: int = 60            # repaid via daily ACH draws over 60 days
    apr: float = 0.35              # 35% annualized
    origination_fee: float = 0.03  # 3% of amount, collected up front
    default_window_days: int = 90  # balance>0 at day 90 => default
    # default triggers: 3 consecutive missed draws OR 6 cumulative missed OR balance>0 @ day90

    def interest_collected_if_full(self, amount: float) -> float:
        """Simple-interest over the 60-day term at APR (upper-bound: full schedule)."""
        return amount * self.apr * (self.term_days / 365.0)

    def gross_revenue_if_full(self, amount: float) -> float:
        """Fee (up front) + interest over term, assuming full on-time repayment."""
        return amount * self.origination_fee + self.interest_collected_if_full(amount)


PRODUCT = Product()


# --------------------------------------------------------------------------- #
# Column groups (from data_dictionary.csv). Outcome cols are label-only / leaky.
# --------------------------------------------------------------------------- #
ID_COLS = ["business_id", "applicant_id"]

OUTCOME_COLS = [
    "default_flag", "days_to_default", "days_to_full_repayment",
    "repayment_status", "final_recovered_amount", "observation_status",
]

# prior_underwriter cols: usable as features in our re-underwriting framing, but
# flagged for the writeup (argued explicitly in reports/design_decisions.md).
PRIOR_UNDERWRITER_COLS = ["prior_underwriter_score", "prior_decision", "prior_approved_amount"]

# Features the data dictionary marks intervenable=True.
DICT_INTERVENABLE = [
    "stated_annual_revenue", "stated_time_in_business", "requested_amount",
    "observed_monthly_revenue_avg_3mo", "observed_revenue_trend_3mo",
    "observed_revenue_volatility", "observed_cash_balance_p10",
    "observed_overdraft_count_3mo", "payroll_regularity_score",
    "aggregate_credit_utilization", "recent_inquiries_count_6mo",
    "existing_debt_obligations", "owner_personal_credit_band",
    "invoice_payment_delinquency_rate", "application_channel",
    "multi_lender_inquiry_count_30d",
]

# Features actually queried in intervention_queries.csv that the dict marks
# intervenable=False (FLAG: data contradicts dictionary; see reports/eda_findings.md).
QUERIED_NOT_DICT_INTERVENABLE = [
    "account_age_days", "bookkeeping_recency_days", "days_since_last_external_decline",
    "days_since_last_inquiry_elsewhere", "employee_count_bucket", "geography_region",
    "has_linked_bank_feed", "intended_use_of_funds", "platform_active_months",
    "prior_loans_amount_total", "prior_loans_count", "prior_loans_default_count",
    "sector", "vintage_years",
]
