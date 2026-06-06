"""Feature engineering with an explicit causal feature registry.

Design goals (driven by the deliverables):
- `build_features(df, fit_artifacts=None) -> (X, artifacts)` is a PURE transform:
  fit medians / encoders / priors on TRAIN only, freeze them in `artifacts`, and
  apply identically to val/test. No sklearn Pipeline black box — every feature is an
  inspectable function so we can defend each driver to a "regulator" (Deliverable D).
- Every engineered feature declares its `parents`. The REGISTRY is a DAG; intervening
  on a raw feature (Deliverable C) recomputes exactly its descendants and nothing
  else. This registry IS our causal-consistency layer.
- Raw intervenable columns are never mutated in place — interventions operate on a
  copy, so `do(feature=value)` is always well-defined.

See reports/feature_catalog.md for the per-family rationale (name, formula,
hypothesis, expected sign, group).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import PATHS, SEED
from src.economics import INT_TERM
from src.data import load_cohort_defs, assign_cohort_week

# Daily ACH draw as a fraction of principal: principal*(1+interest)/term, fee is upfront.
from src.config import PRODUCT

DAILY_PAY_FRAC = (1 + INT_TERM) / PRODUCT.term_days  # ~0.0176 of principal per day

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class Feature:
    name: str
    parents: list[str]
    group: str
    fn: Callable[[pd.DataFrame, dict], pd.Series]
    expected_sign: str = "?"   # "+" risk-increasing, "-" risk-decreasing, "?" unknown

REGISTRY: dict[str, Feature] = {}


def feature(name: str, parents: list[str], group: str, sign: str = "?"):
    def deco(fn):
        REGISTRY[name] = Feature(name, parents, group, fn, sign)
        return fn
    return deco


def _topo_order() -> list[str]:
    """Topologically order registry features so parents-in-registry compute first."""
    done, order = set(), []
    names = list(REGISTRY)
    while len(order) < len(names):
        progressed = False
        for n in names:
            if n in done:
                continue
            reg_parents = [p for p in REGISTRY[n].parents if p in REGISTRY]
            if all(p in done for p in reg_parents):
                order.append(n); done.add(n); progressed = True
        if not progressed:
            raise RuntimeError("cycle in feature registry")
    return order


def descendants(feat: str) -> set[str]:
    """All registry features whose ancestor set includes `feat` (its raw-column name)."""
    out, changed = set(), True
    while changed:
        changed = False
        for n, f in REGISTRY.items():
            if n in out:
                continue
            if feat in f.parents or out & set(f.parents):
                out.add(n); changed = True
    return out


# --------------------------------------------------------------------------- #
# Small null-safe helpers
# --------------------------------------------------------------------------- #
def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    if np.isscalar(b) or (hasattr(b, "ndim") and getattr(b, "ndim", 1) == 0):
        b = np.nan if b == 0 else b
        return (a / b).replace([np.inf, -np.inf], np.nan)
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _log1p_signed(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.sign(x) * np.log1p(np.abs(x))


def _exp_decay(days, tau):
    d = pd.to_numeric(days, errors="coerce")
    # null (= event never happened) -> 0 recency (good); present -> exp(-days/tau)
    return np.where(d.isna(), 0.0, np.exp(-d.clip(lower=0) / tau))


# =========================================================================== #
# FAMILY 1 — Affordability / cash-flow coverage (buffer-centric)
# =========================================================================== #
@feature("daily_payment", ["requested_amount"], "affordability")
def _daily_payment(w, art):
    return pd.to_numeric(w["requested_amount"], errors="coerce") * DAILY_PAY_FRAC

@feature("buffer_to_payment", ["observed_cash_balance_p10", "requested_amount"], "affordability", "-")
def _buffer_to_payment(w, art):
    return _safe_div(w["observed_cash_balance_p10"], w["daily_payment"]).clip(-50, 200)

@feature("dscr_daily", ["observed_monthly_revenue_avg_3mo", "requested_amount"], "affordability", "-")
def _dscr(w, art):
    daily_rev = _safe_div(w["observed_monthly_revenue_avg_3mo"], 30.0)
    return _safe_div(daily_rev, w["daily_payment"]).clip(0, 5000)

@feature("payment_to_revenue", ["observed_monthly_revenue_avg_3mo", "requested_amount"], "affordability", "+")
def _payment_to_revenue(w, art):
    monthly_pay = w["daily_payment"] * 30
    return _safe_div(monthly_pay, w["observed_monthly_revenue_avg_3mo"]).clip(0, 5)

@feature("dscr_daily_stated", ["stated_annual_revenue", "requested_amount"], "affordability", "-")
def _dscr_stated(w, art):
    daily_rev = _safe_div(w["stated_annual_revenue"], 365.0)
    return _safe_div(daily_rev, w["daily_payment"]).clip(0, 5000)

@feature("leverage_total", ["existing_debt_obligations", "requested_amount",
                            "observed_monthly_revenue_avg_3mo", "stated_annual_revenue"], "affordability", "+")
def _leverage(w, art):
    obs_annual = pd.to_numeric(w["observed_monthly_revenue_avg_3mo"], errors="coerce") * 12
    annual = obs_annual.fillna(pd.to_numeric(w["stated_annual_revenue"], errors="coerce"))
    total_debt = pd.to_numeric(w["existing_debt_obligations"], errors="coerce") + pd.to_numeric(w["requested_amount"], errors="coerce")
    return _safe_div(total_debt, annual).clip(0, 50)


# =========================================================================== #
# FAMILY 2 — Discrepancy / misreporting
# =========================================================================== #
@feature("rev_log_ratio", ["stated_annual_revenue", "observed_monthly_revenue_avg_3mo"], "discrepancy", "+")
def _rev_log_ratio(w, art):
    stated_m = _safe_div(w["stated_annual_revenue"], 12.0)
    r = _safe_div(stated_m, w["observed_monthly_revenue_avg_3mo"])
    return np.log(r.clip(lower=0.01, upper=100))

@feature("revenue_overstated", ["stated_annual_revenue", "observed_monthly_revenue_avg_3mo"], "discrepancy", "+")
def _overstated(w, art):
    stated_m = _safe_div(w["stated_annual_revenue"], 12.0)
    r = _safe_div(stated_m, w["observed_monthly_revenue_avg_3mo"])
    return (r > 1.5).astype(float)

@feature("tib_gap", ["stated_time_in_business", "vintage_years"], "discrepancy", "+")
def _tib_gap(w, art):
    return pd.to_numeric(w["stated_time_in_business"], errors="coerce") - pd.to_numeric(w["vintage_years"], errors="coerce")

@feature("tib_inflated", ["stated_time_in_business", "vintage_years"], "discrepancy", "+")
def _tib_inflated(w, art):
    return (w["tib_gap"] > 0.5).astype(float)

@feature("req_to_obs_annual", ["requested_amount", "observed_monthly_revenue_avg_3mo"], "discrepancy", "+")
def _req_to_obs_annual(w, art):
    return _safe_div(w["requested_amount"], pd.to_numeric(w["observed_monthly_revenue_avg_3mo"], errors="coerce") * 12).clip(0, 5)

@feature("req_to_stated_annual", ["requested_amount", "stated_annual_revenue"], "discrepancy", "+")
def _req_to_stated_annual(w, art):
    return _safe_div(w["requested_amount"], w["stated_annual_revenue"]).clip(0, 5)


# =========================================================================== #
# FAMILY 3 — Informative missingness indicators (first-class)
# =========================================================================== #
@feature("no_bank_feed", ["has_linked_bank_feed"], "missingness", "+")
def _no_feed(w, art):
    return (w["has_linked_bank_feed"] != True).astype(float)

@feature("never_declined_external", ["days_since_last_external_decline"], "missingness", "-")
def _never_declined(w, art):
    return w["days_since_last_external_decline"].isna().astype(float)

@feature("no_inquiry_elsewhere", ["days_since_last_inquiry_elsewhere"], "missingness", "-")
def _no_inquiry(w, art):
    return w["days_since_last_inquiry_elsewhere"].isna().astype(float)

@feature("is_first_time_borrower", ["prior_loans_count"], "missingness", "?")
def _first_time(w, art):
    return (pd.to_numeric(w["prior_loans_count"], errors="coerce").fillna(0) == 0).astype(float)

@feature("prior_declined_elsewhere_flag", ["prior_approved_amount"], "missingness", "+")
def _prior_declined(w, art):
    return w["prior_approved_amount"].isna().astype(float)

@feature("no_feed_x_amount", ["has_linked_bank_feed", "requested_amount"], "missingness", "+")
def _no_feed_x_amount(w, art):
    amt = pd.to_numeric(w["requested_amount"], errors="coerce")
    amt_z = (amt - art["medians"]["requested_amount"]) / (art["scales"]["requested_amount"] + EPS)
    return w["no_bank_feed"] * amt_z


# =========================================================================== #
# FAMILY 4 — Credit-stress composites
# =========================================================================== #
@feature("util_x_inquiries", ["aggregate_credit_utilization", "recent_inquiries_count_6mo"], "credit_stress", "+")
def _util_x_inq(w, art):
    return pd.to_numeric(w["aggregate_credit_utilization"], errors="coerce") * pd.to_numeric(w["recent_inquiries_count_6mo"], errors="coerce")

@feature("debt_to_revenue", ["existing_debt_obligations", "observed_monthly_revenue_avg_3mo", "stated_annual_revenue"], "credit_stress", "+")
def _debt_to_rev(w, art):
    obs_annual = pd.to_numeric(w["observed_monthly_revenue_avg_3mo"], errors="coerce") * 12
    annual = obs_annual.fillna(pd.to_numeric(w["stated_annual_revenue"], errors="coerce"))
    return _safe_div(w["existing_debt_obligations"], annual).clip(0, 50)

@feature("overdrafts_per_month", ["observed_overdraft_count_3mo"], "credit_stress", "+")
def _overdrafts_pm(w, art):
    return pd.to_numeric(w["observed_overdraft_count_3mo"], errors="coerce") / 3.0

@feature("inquiry_velocity", ["multi_lender_inquiry_count_30d", "recent_inquiries_count_6mo"], "credit_stress", "+")
def _inq_velocity(w, art):
    return _safe_div(pd.to_numeric(w["multi_lender_inquiry_count_30d"], errors="coerce"),
                     pd.to_numeric(w["recent_inquiries_count_6mo"], errors="coerce") + 1).clip(0, 30)

@feature("decline_recency", ["days_since_last_external_decline"], "credit_stress", "+")
def _decline_recency(w, art):
    return pd.Series(_exp_decay(w["days_since_last_external_decline"], art["tau"]["days_since_last_external_decline"]), index=w.index)

@feature("inquiry_recency", ["days_since_last_inquiry_elsewhere"], "credit_stress", "+")
def _inquiry_recency(w, art):
    return pd.Series(_exp_decay(w["days_since_last_inquiry_elsewhere"], art["tau"]["days_since_last_inquiry_elsewhere"]), index=w.index)


# =========================================================================== #
# FAMILY 5 — Platform-relationship (empirical-Bayes shrinkage)
# =========================================================================== #
@feature("prior_default_rate_shrunk", ["prior_loans_default_count", "prior_loans_count"], "platform", "+")
def _prior_default_rate(w, art):
    a = art["eb_prior"]["alpha"]; b = art["eb_prior"]["beta"]
    d = pd.to_numeric(w["prior_loans_default_count"], errors="coerce").fillna(0)
    n = pd.to_numeric(w["prior_loans_count"], errors="coerce").fillna(0)
    return (d + a) / (n + a + b)

@feature("avg_prior_loan_size", ["prior_loans_amount_total", "prior_loans_count"], "platform", "?")
def _avg_prior_size(w, art):
    return _safe_div(w["prior_loans_amount_total"], pd.to_numeric(w["prior_loans_count"], errors="coerce")).fillna(0)

@feature("prior_size_vs_request", ["prior_loans_amount_total", "prior_loans_count", "requested_amount"], "platform", "?")
def _prior_size_vs_req(w, art):
    return _safe_div(w["avg_prior_loan_size"], w["requested_amount"]).fillna(0).clip(0, 20)

@feature("engagement_intensity", ["platform_active_months", "account_age_days"], "platform", "-")
def _engagement(w, art):
    months = pd.to_numeric(w["account_age_days"], errors="coerce") / 30.0
    return _safe_div(pd.to_numeric(w["platform_active_months"], errors="coerce"), months).clip(0, 3)

@feature("bookkeeping_recency_decay", ["bookkeeping_recency_days"], "platform", "-")
def _bk_recency(w, art):
    return pd.Series(_exp_decay(w["bookkeeping_recency_days"], art["tau"]["bookkeeping_recency_days"]), index=w.index)


# =========================================================================== #
# FAMILY 6 — Volatility / stability
# =========================================================================== #
@feature("rev_vol_x_negtrend", ["observed_revenue_volatility", "observed_revenue_trend_3mo"], "volatility", "+")
def _vol_x_trend(w, art):
    return pd.to_numeric(w["observed_revenue_volatility"], errors="coerce") * (-pd.to_numeric(w["observed_revenue_trend_3mo"], errors="coerce"))

@feature("payroll_regularity", ["payroll_regularity_score"], "volatility", "-")
def _payroll(w, art):
    return pd.to_numeric(w["payroll_regularity_score"], errors="coerce")

@feature("volatility_level", ["observed_revenue_volatility"], "volatility", "+")
def _vol(w, art):
    return pd.to_numeric(w["observed_revenue_volatility"], errors="coerce")


# =========================================================================== #
# FAMILY 9 — Selection / regression-discontinuity features
# =========================================================================== #
@feature("rd_distance", ["prior_underwriter_score"], "selection", "-")
def _rd_distance(w, art):
    return pd.to_numeric(w["prior_underwriter_score"], errors="coerce") - art["rd_cutoff"]

@feature("above_rd_cutoff", ["prior_underwriter_score"], "selection", "-")
def _above_cutoff(w, art):
    return (pd.to_numeric(w["prior_underwriter_score"], errors="coerce") >= art["rd_cutoff"]).astype(float)

@feature("prior_score", ["prior_underwriter_score"], "selection", "-")
def _prior_score(w, art):
    return pd.to_numeric(w["prior_underwriter_score"], errors="coerce")




# Monetary log transforms (raw passthrough, log1p) — used by linear baseline.
LOG_COLS = ["requested_amount", "stated_annual_revenue", "observed_monthly_revenue_avg_3mo",
            "existing_debt_obligations", "observed_cash_balance_p10", "prior_loans_amount_total"]
for _c in LOG_COLS:
    def _mk(col):
        @feature(f"log_{col}", [col], "transform")
        def _f(w, art, _col=col):
            return _log1p_signed(w[_col])
        return _f
    _mk(_c)


# Numeric raw passthroughs to keep available to the model (already-clean signals).
PASSTHROUGH = [
    "requested_amount", "vintage_years", "stated_time_in_business",
    "aggregate_credit_utilization", "recent_inquiries_count_6mo", "existing_debt_obligations",
    "invoice_payment_delinquency_rate", "multi_lender_inquiry_count_30d",
    "observed_cash_balance_p10", "observed_overdraft_count_3mo",
    "observed_revenue_trend_3mo", "platform_active_months", "account_age_days",
    "prior_loans_count", "repeat_application_count",
]
for _c in PASSTHROUGH:
    def _mkp(col):
        @feature(f"raw_{col}", [col], "passthrough")
        def _f(w, art, _col=col):
            return pd.to_numeric(w[_col], errors="coerce")
        return _f
    _mkp(_c)


# =========================================================================== #
# FAMILY 8 — Categorical encoding (OOF target encoding + ordinal)
# =========================================================================== #
NOMINAL_CATS = ["sector", "geography_region", "intended_use_of_funds", "application_channel"]
ORDINAL_CATS = ["owner_personal_credit_band", "employee_count_bucket"]
TE_SMOOTHING = 20.0


def _fit_target_maps(df, y, cols, smoothing=TE_SMOOTHING):
    prior = float(y.mean())
    maps = {}
    for c in cols:
        g = pd.DataFrame({"c": df[c], "y": y}).dropna(subset=["y"]).groupby("c")["y"]
        cnt, mean = g.count(), g.mean()
        enc = (cnt * mean + smoothing * prior) / (cnt + smoothing)
        maps[c] = enc.to_dict()
    return prior, maps


def _apply_target_maps(df, cols, prior, maps):
    out = {}
    for c in cols:
        out[f"te_{c}"] = df[c].map(maps[c]).fillna(prior).to_numpy()
    return pd.DataFrame(out, index=df.index)


def _oof_target_encode(df, y, groups, cols, seed=SEED, n_splits=5, smoothing=TE_SMOOTHING):
    """Group-aware OOF target encoding so train rows never see their own label/group."""
    from sklearn.model_selection import GroupKFold
    lab = y.notna()
    out = pd.DataFrame(index=df.index, columns=[f"te_{c}" for c in cols], dtype=float)
    prior_full = float(y[lab].mean())
    idx = np.where(lab.to_numpy())[0]
    gkf = GroupKFold(n_splits=n_splits)
    for tr_i, va_i in gkf.split(idx, y[lab].to_numpy(), groups[lab].to_numpy()):
        tr_rows, va_rows = df.index[idx[tr_i]], df.index[idx[va_i]]
        _, maps = _fit_target_maps(df.loc[tr_rows], y.loc[tr_rows], cols, smoothing)
        enc = _apply_target_maps(df.loc[va_rows], cols, prior_full, maps)
        out.loc[va_rows] = enc.to_numpy()
    # unlabeled rows: full-map encoding
    _, full_maps = _fit_target_maps(df.loc[df.index[idx]], y[lab], cols, smoothing)
    enc_all = _apply_target_maps(df.loc[df.index[~lab.to_numpy()]], cols, prior_full, full_maps)
    out.loc[df.index[~lab.to_numpy()]] = enc_all.to_numpy()
    return out.astype(float)


# =========================================================================== #
# Artifacts (fit on train)
# =========================================================================== #
def _fit_artifacts(df: pd.DataFrame, y: pd.Series | None, groups: pd.Series | None) -> dict:
    art: dict = {"medians": {}, "scales": {}, "tau": {}}
    num_cols = df.select_dtypes("number").columns
    for c in num_cols:
        art["medians"][c] = float(df[c].median())
        art["scales"][c] = float(df[c].std(ddof=0)) or 1.0
    for c in ["days_since_last_external_decline", "days_since_last_inquiry_elsewhere", "bookkeeping_recency_days"]:
        present = pd.to_numeric(df[c], errors="coerce").dropna()
        art["tau"][c] = float(present.median()) if len(present) and present.median() > 0 else 30.0
    # RD cutoff: midpoint between max declined score and min approved score
    appr = df.loc[df["prior_decision"] == 1, "prior_underwriter_score"]
    decl = df.loc[df["prior_decision"] == 0, "prior_underwriter_score"]
    art["rd_cutoff"] = float((appr.min() + decl.max()) / 2) if len(decl) else 0.5
    # Empirical-Bayes Beta prior for prior-default-rate shrinkage (method of moments-ish)
    base = float(y.mean()) if y is not None and y.notna().any() else 0.175
    pseudo = 5.0  # pseudo-count strength
    art["eb_prior"] = {"alpha": base * pseudo, "beta": (1 - base) * pseudo, "global_rate": base}
    # Target-encoding maps (full-fit, for transform/intervention)
    if y is not None and y.notna().any():
        prior, maps = _fit_target_maps(df.loc[y.notna()], y[y.notna()], NOMINAL_CATS)
        art["te_prior"], art["te_maps"] = prior, maps
    else:
        art["te_prior"], art["te_maps"] = base, {c: {} for c in NOMINAL_CATS}
    art["seed"] = SEED
    return art


# =========================================================================== #
# Registry compute
# =========================================================================== #
def _compute_registry(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    w = df.copy()
    for name in _topo_order():
        w[name] = REGISTRY[name].fn(w, art)
    return w[[n for n in REGISTRY]]


# =========================================================================== #
# Public API
# =========================================================================== #
def build_features(df: pd.DataFrame, fit_artifacts: dict | None = None,
                   y: pd.Series | None = None, groups: pd.Series | None = None):
    """Pure transform. Fit mode (fit_artifacts=None) needs y+groups for OOF encoding.

    Returns (X, artifacts). X is indexed like df; ids are not included.
    """
    fit = fit_artifacts is None
    if fit:
        if y is None:
            raise ValueError("fit mode requires y (NaN allowed for unlabeled rows)")
        if groups is None:
            groups = df["business_id"]
        art = _fit_artifacts(df, y, groups)
    else:
        art = fit_artifacts

    reg = _compute_registry(df, art)

    # Ordinal categoricals: pass through as ordered integers.
    ordinals = pd.DataFrame({f"ord_{c}": pd.to_numeric(df[c], errors="coerce") for c in ORDINAL_CATS}, index=df.index)

    # Target-encoded nominal categoricals.
    if fit:
        te = _oof_target_encode(df, y, groups, NOMINAL_CATS)
    else:
        te = _apply_target_maps(df, NOMINAL_CATS, art["te_prior"], art["te_maps"])

    X = pd.concat([reg, ordinals, te], axis=1)
    return X, art


def recompute_under_intervention(df: pd.DataFrame, feature_name: str, value, art: dict) -> pd.DataFrame:
    """do(feature_name = value): set the raw column on a COPY and recompute features.

    Returns the full engineered frame (registry + ordinals + target-encoded) under the
    intervention. Non-descendants are byte-identical to the observational features.
    """
    d = df.copy()
    d[feature_name] = value
    reg = _compute_registry(d, art)
    ordinals = pd.DataFrame({f"ord_{c}": pd.to_numeric(d[c], errors="coerce") for c in ORDINAL_CATS}, index=d.index)
    te = _apply_target_maps(d, NOMINAL_CATS, art["te_prior"], art["te_maps"])
    return pd.concat([reg, ordinals, te], axis=1)


def intervenable_descendants(feature_name: str) -> dict:
    """Which engineered features move under do(feature_name=.) — for the writeup/tests."""
    return {"direct_children": [n for n, f in REGISTRY.items() if feature_name in f.parents],
            "all_descendants": sorted(descendants(feature_name))}


# =========================================================================== #
# FAMILY 10 — Survival long-format (Deliverable B)
# =========================================================================== #
def make_survival_long(df: pd.DataFrame, n_weeks: int = 13) -> pd.DataFrame:
    """One row per (labeled loan, loan-age week 1..13) with a discrete-time event
    indicator. No censoring in this data (all matured), so each loan contributes
    weeks until it defaults; default at day d -> event in week ceil(d/7).

    Also flags `day90_open_balance` (the day-90 point mass = open-balance default
    mode), so Deliverable B can be modeled as two components.
    """
    lab = df["default_flag"].notna()
    d = df.loc[lab].copy()
    d["age_week_event"] = np.where(d["default_flag"] == 1,
                                   np.ceil(pd.to_numeric(d["days_to_default"], errors="coerce") / 7).clip(upper=n_weeks),
                                   np.nan)
    d["day90_open_balance"] = ((d["default_flag"] == 1) & (pd.to_numeric(d["days_to_default"], errors="coerce") >= 89)).astype(int)
    rows = []
    aw = d["age_week_event"].to_numpy()
    ids = d["applicant_id"].to_numpy()
    defaulted = (d["default_flag"] == 1).to_numpy()
    for i in range(len(d)):
        last = int(aw[i]) if defaulted[i] and not np.isnan(aw[i]) else n_weeks
        for week in range(1, last + 1):
            event = int(defaulted[i] and not np.isnan(aw[i]) and week == int(aw[i]))
            rows.append((ids[i], week, event))
    long = pd.DataFrame(rows, columns=["applicant_id", "loan_age_week", "event"])
    return long


# =========================================================================== #
# Persistence
# =========================================================================== #
def save_artifacts(art: dict, path: Path | None = None) -> Path:
    path = path or (PATHS.artifacts / "feature_artifacts.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    return path


def load_artifacts(path: Path | None = None) -> dict:
    path = path or (PATHS.artifacts / "feature_artifacts.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# Near-perfect duplicates (Pearson |r| ≥ 0.95) of other features. Kept as registry
# nodes — they are useful semantic parents for `do()` propagation — but dropped
# from the model matrix so SHAP attribution isn't split across collinear twins
# and the regulator-facing driver ranking stays clean. Each comment notes the twin.
REDUNDANT_DROP = [
    "daily_payment",            # ≡ raw_requested_amount (scalar multiple)
    "req_to_obs_annual",        # ≡ payment_to_revenue (same ratio)
    "overdrafts_per_month",     # ≡ raw_observed_overdraft_count_3mo
    "rd_distance",              # ≡ prior_score (affine); keep prior_score + above_rd_cutoff
    "prior_declined_elsewhere_flag",  # ≡ above_rd_cutoff (deterministic gate)
    "log_requested_amount",     # ≡ raw_requested_amount
    "avg_prior_loan_size",      # ≈ log_prior_loans_amount_total / prior_size_vs_request
]


def model_features(X: pd.DataFrame) -> pd.DataFrame:
    """Deduped view of the engineered frame for model training (drops collinear twins).

    The full X (with all registry nodes) is still what `recompute_under_intervention`
    operates on; this is only the matrix we feed the learner.
    """
    return X.drop(columns=[c for c in REDUNDANT_DROP if c in X.columns])


def feature_groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n, f in REGISTRY.items():
        out.setdefault(f.group, []).append(n)
    out["categorical_ordinal"] = [f"ord_{c}" for c in ORDINAL_CATS]
    out["categorical_target_enc"] = [f"te_{c}" for c in NOMINAL_CATS]
    return out
