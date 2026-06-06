"""EDA analysis helpers, organized around the five structural issues.

Reusable, importable functions (KS/SMD, missingness, KM hazard, discrepancy,
leakage) so the notebook is a thin narrative layer. `python -m src.eda`
recomputes every number + figure and dumps reports/eda_stats.json.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import data
from src.config import PATHS, PRODUCT, set_seeds

warnings.filterwarnings("ignore")
FIGDIR = PATHS.reports / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _round(x, n=4):
    try:
        return float(np.round(x, n))
    except Exception:
        return x


# --------------------------------------------------------------------------- #
# 1.1 Selection bias / reject inference
# --------------------------------------------------------------------------- #
def label_coverage(dfs: dict) -> dict:
    out = {}
    for nm, d in dfs.items():
        m = data.labeled_mask(d)
        out[nm] = {
            "n": int(len(d)),
            "labeled": int(m.sum()),
            "labeled_frac": _round(m.mean()),
            "default_rate_labeled": _round(d.loc[m, "default_flag"].mean()) if m.any() else None,
        }
    return out


def prior_decision_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["prior_decision"], df["observation_status"], dropna=False)


def approved_vs_declined_smd(train: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference + KS (numeric) / Cramer-V chi2 (categorical)
    between prior-approved and prior-declined applicants."""
    appr = train["prior_decision"] == 1
    num = data.numeric_feature_columns(train, include_prior_underwriter=True)
    cat = data.categorical_feature_columns(train, include_prior_underwriter=True)
    rows = []
    for c in num:
        a, b = train.loc[appr, c].dropna(), train.loc[~appr, c].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
        ks, kp = stats.ks_2samp(a, b)
        rows.append({"feature": c, "kind": "num", "smd": _round(smd, 3),
                     "ks_stat": _round(ks, 3), "ks_p": _round(kp, 6),
                     "appr_mean": _round(a.mean(), 3), "decl_mean": _round(b.mean(), 3)})
    for c in cat:
        ct = pd.crosstab(train[c], appr)
        if ct.shape[0] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        n = ct.to_numpy().sum()
        cramv = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rows.append({"feature": c, "kind": "cat", "smd": np.nan,
                     "ks_stat": _round(cramv, 3), "ks_p": _round(p, 6),
                     "appr_mean": np.nan, "decl_mean": np.nan})
    df = pd.DataFrame(rows)
    df["abs_effect"] = df["smd"].abs().fillna(df["ks_stat"])
    return df.sort_values("abs_effect", ascending=False).reset_index(drop=True)


def prior_score_threshold(train: pd.DataFrame) -> dict:
    """Characterize the approval rule on prior_underwriter_score: sharp vs fuzzy."""
    s = train["prior_underwriter_score"]
    appr = train["prior_decision"] == 1
    # approval rate by score quantile bin
    q = pd.qcut(s.rank(method="first"), 50, labels=False)
    rate = train.groupby(q)["prior_decision"].mean()
    # find steepest transition; estimate threshold as score where approval rate crosses 0.5
    grid = np.linspace(s.quantile(0.01), s.quantile(0.99), 200)
    appr_rate = np.array([(appr & (s >= g)).sum() / max((s >= g).sum(), 1) for g in grid])
    # sharpness: overlap of score distributions across the decision
    a, b = s[appr].dropna(), s[~appr].dropna()
    ov_lo, ov_hi = max(a.min(), b.min()), min(a.max(), b.max())
    overlap_frac = float(((s >= ov_lo) & (s <= ov_hi)).mean())
    # crossing point of approval-rate==0.5
    cross = grid[np.argmin(np.abs(rate.reindex(range(50)).interpolate().to_numpy().mean() - 0.5))] if False else None
    return {
        "appr_score_mean": _round(a.mean(), 3), "decl_score_mean": _round(b.mean(), 3),
        "appr_score_min": _round(a.min(), 3), "appr_score_max": _round(a.max(), 3),
        "decl_score_min": _round(b.min(), 3), "decl_score_max": _round(b.max(), 3),
        "overlap_frac": _round(overlap_frac, 3),
        "ks": _round(stats.ks_2samp(a, b)[0], 3),
        "auc_score_predicts_approval": _round(_auc(s.fillna(s.median()), appr.astype(int)), 4),
    }


def _auc(score: pd.Series, y: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y, score)
    except Exception:
        return np.nan


# --------------------------------------------------------------------------- #
# 1.2 Censoring & timing (drives Deliverable B)
# --------------------------------------------------------------------------- #
def timing_structure(train: pd.DataFrame) -> dict:
    d = train.loc[data.labeled_mask(train)]
    dd = d.loc[d["default_flag"] == 1, "days_to_default"]
    rep = d.loc[d["default_flag"] == 0, "days_to_full_repayment"]
    return {
        "n_labeled": int(len(d)),
        "n_default": int((d["default_flag"] == 1).sum()),
        "dtd_min": _round(dd.min(), 1), "dtd_max": _round(dd.max(), 1),
        "dtd_mean": _round(dd.mean(), 1), "dtd_median": _round(dd.median(), 1),
        "dtd_pre60_frac": _round((dd <= 60).mean(), 3),
        "dtd_post60_frac": _round((dd > 60).mean(), 3),
        "obs_status_values": d["observation_status"].dropna().unique().tolist(),
        "repay_status_values": d["repayment_status"].dropna().unique().tolist(),
        "repay_min": _round(rep.min(), 1), "repay_max": _round(rep.max(), 1),
        "repay_mean": _round(rep.mean(), 1),
    }


def discrete_hazard(train: pd.DataFrame, n_days: int = 90) -> pd.DataFrame:
    """Discrete-time hazard h(t)=P(default at day t | survived to t-1) over the
    labeled (approved+matured) population. No censoring here, so risk set just
    shrinks by defaults + payoffs each day."""
    d = train.loc[data.labeled_mask(train)].copy()
    n = len(d)
    days = np.arange(1, n_days + 1)
    dtd = d.loc[d["default_flag"] == 1, "days_to_default"].to_numpy()
    payoff = d.loc[d["default_flag"] == 0, "days_to_full_repayment"].to_numpy()
    rows = []
    at_risk = n
    cum_def = 0
    for t in days:
        ev = int((dtd == t).sum())
        po = int((payoff == t).sum())
        haz = ev / at_risk if at_risk > 0 else 0.0
        cum_def += ev
        rows.append({"day": int(t), "at_risk": at_risk, "defaults": ev, "payoffs": po,
                     "hazard": _round(haz, 5), "cum_default_frac": _round(cum_def / n, 5)})
        at_risk -= (ev + po)
    return pd.DataFrame(rows)


def cohort_default_rates(val: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort default rate on the labeled (approved) val applicants."""
    v = val.copy()
    v["cohort_week"] = data.assign_cohort_week(v)
    m = data.labeled_mask(v)
    g = v.loc[m].groupby("cohort_week")["default_flag"]
    return pd.DataFrame({"n_labeled": g.size(), "default_rate": g.mean().round(4)}).reset_index()


def timestamp_ranges(dfs: dict) -> dict:
    out = {}
    for nm, d in dfs.items():
        ts = d["application_timestamp"]
        c = data.assign_cohort_week(d)
        out[nm] = {"min": str(ts.min()), "max": str(ts.max()),
                   "cohort_coverage": _round(c.notna().mean(), 3)}
    return out


# --------------------------------------------------------------------------- #
# 1.3 Informative missingness
# --------------------------------------------------------------------------- #
def missingness_by_split(dfs: dict) -> pd.DataFrame:
    cols = data.feature_columns()
    rows = []
    for c in cols:
        r = {"feature": c}
        for nm, d in dfs.items():
            r[f"null_{nm}"] = _round(d[c].isna().mean(), 4) if c in d else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("null_train", ascending=False).reset_index(drop=True)


def missingness_signal(train: pd.DataFrame) -> pd.DataFrame:
    """Default rate split by null-ness of informative-missing columns."""
    d = train.loc[data.labeled_mask(train)]
    base = d["default_flag"].mean()
    targets = ["has_linked_bank_feed", "days_since_last_external_decline",
               "days_since_last_inquiry_elsewhere", "prior_approved_amount",
               "observed_monthly_revenue_avg_3mo"]
    rows = [{"feature": "<<baseline>>", "group": "all", "n": int(len(d)),
             "default_rate": _round(base, 4)}]
    for c in targets:
        if c not in d:
            continue
        if d[c].dtype == bool or set(d[c].dropna().unique()) <= {True, False}:
            for val_ in [True, False]:
                m = d[c] == val_
                if m.any():
                    rows.append({"feature": c, "group": str(val_), "n": int(m.sum()),
                                 "default_rate": _round(d.loc[m, "default_flag"].mean(), 4)})
        else:
            for label, m in [("null", d[c].isna()), ("present", d[c].notna())]:
                if m.any():
                    rows.append({"feature": c, "group": label, "n": int(m.sum()),
                                 "default_rate": _round(d.loc[m, "default_flag"].mean(), 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1.4 Self-reported vs observed discrepancies
# --------------------------------------------------------------------------- #
def revenue_discrepancy(train: pd.DataFrame) -> dict:
    d = train.loc[data.labeled_mask(train) & (train["has_linked_bank_feed"] == True)].copy()
    stated_m = d["stated_annual_revenue"] / 12.0
    obs_m = d["observed_monthly_revenue_avg_3mo"]
    ratio = (stated_m / obs_m).replace([np.inf, -np.inf], np.nan).dropna()
    d = d.assign(rev_ratio=stated_m / obs_m)
    overst = d["rev_ratio"] > 1.5
    res = {
        "n_with_feed": int(len(d)),
        "ratio_median": _round(ratio.median(), 3),
        "ratio_p10": _round(ratio.quantile(0.1), 3),
        "ratio_p90": _round(ratio.quantile(0.9), 3),
        "frac_overstate_1.5x": _round(overst.mean(), 3),
        "default_rate_overstate": _round(d.loc[overst, "default_flag"].mean(), 4),
        "default_rate_not_overstate": _round(d.loc[~overst, "default_flag"].mean(), 4),
    }
    # time-in-business gap
    tib = train.loc[data.labeled_mask(train)].copy()
    gap = tib["stated_time_in_business"] - tib["vintage_years"]
    res["tib_gap_median"] = _round(gap.median(), 3)
    res["tib_stated_gt_vintage_frac"] = _round((gap > 0.5).mean(), 3)
    return res


def recompute_ratio_check(train: pd.DataFrame) -> dict:
    """Sanity-check provided requested_amount_to_observed_revenue."""
    d = train.copy()
    obs_m = d["observed_monthly_revenue_avg_3mo"]
    recomputed = d["requested_amount"] / obs_m
    prov = d["requested_amount_to_observed_revenue"]
    both = prov.notna() & recomputed.notna()
    diff = (prov[both] - recomputed[both]).abs()
    return {
        "provided_null_frac": _round(prov.isna().mean(), 4),
        "provided_null_when_no_feed_frac": _round(prov[d["has_linked_bank_feed"] == False].isna().mean(), 4),
        "n_comparable": int(both.sum()),
        "max_abs_diff_vs_naive_recompute": _round(diff.max(), 4) if both.any() else None,
        "median_abs_diff": _round(diff.median(), 6) if both.any() else None,
    }


# --------------------------------------------------------------------------- #
# 1.5 Entity & leakage
# --------------------------------------------------------------------------- #
def entity_structure(dfs: dict) -> dict:
    tr, va, te = dfs["train"], dfs["validation"], dfs["test"]
    bt, bv, bte = set(tr.business_id), set(va.business_id), set(te.business_id)
    vc = tr["business_id"].value_counts()
    return {
        "business_overlap_tr_va": len(bt & bv),
        "business_overlap_tr_te": len(bt & bte),
        "business_overlap_va_te": len(bv & bte),
        "train_n_businesses": int(tr["business_id"].nunique()),
        "train_multi_app_businesses": int((vc > 1).sum()),
        "train_max_apps_per_business": int(vc.max()),
        "repeat_app_count_max": int(tr["repeat_application_count"].max()),
        "prior_loans_count_max": int(tr["prior_loans_count"].max()),
        "frac_first_time": _round((tr["prior_loans_count"] == 0).mean(), 3),
    }


def leakage_audit(train: pd.DataFrame) -> pd.DataFrame:
    """Single-feature AUC for default among labeled rows — canary for leakage."""
    from sklearn.metrics import roc_auc_score
    d = train.loc[data.labeled_mask(train)]
    y = d["default_flag"].astype(int)
    cols = data.feature_columns(include_prior_underwriter=True)
    rows = []
    for c in cols:
        x = pd.to_numeric(d[c], errors="coerce")
        if x.notna().sum() < 100 or x.nunique() < 2:
            continue
        xf = x.fillna(x.median())
        try:
            auc = roc_auc_score(y, xf)
        except Exception:
            continue
        rows.append({"feature": c, "auc": _round(max(auc, 1 - auc), 4), "raw_auc": _round(auc, 4)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def class_balance(train: pd.DataFrame, by: list[str]) -> dict:
    d = train.loc[data.labeled_mask(train)]
    out = {"overall": _round(d["default_flag"].mean(), 4)}
    for c in by:
        g = d.groupby(c)["default_flag"].agg(["mean", "size"])
        out[c] = {str(k): {"rate": _round(v["mean"], 4), "n": int(v["size"])}
                  for k, v in g.iterrows()}
    return out


def integrity_checks(dfs: dict) -> dict:
    """Cross-field / logical / domain integrity — the validation that matters here
    instead of classical cleaning (the data is synthetic and internally consistent)."""
    tr, va, te = dfs["train"], dfs["validation"], dfs["test"]
    d = tr.loc[data.labeled_mask(tr)]
    dtd = d["days_to_default"]
    label = {
        "default_flag_xor_status": int(((d.default_flag == 1) != (d.repayment_status == "defaulted")).sum()),
        "default_flag_xor_dtd": int(((d.default_flag == 1) != (d.days_to_default.notna())).sum()),
        "paid_xor_repayday": int(((d.default_flag == 0) != (d.days_to_full_repayment.notna())).sum()),
        "recovery_on_nondefault": int((d.final_recovered_amount.notna() & (d.default_flag == 0)).sum()),
        "dtd_out_of_range": int(((dtd < 1) | (dtd > 90)).fillna(False).sum()),
    }
    logical = {
        "prior_def_gt_count": int((tr.prior_loans_default_count > tr.prior_loans_count).sum()),
        "approved_amt_xor_decision": int((tr.prior_approved_amount.notna() != (tr.prior_decision == 1)).sum()),
        "approved_gt_requested": int((tr.prior_approved_amount > tr.requested_amount).sum()),
        "observed_xor_feed": int((tr.observed_monthly_revenue_avg_3mo.notna() != (tr.has_linked_bank_feed == True)).sum()),
    }
    bounded = ["aggregate_credit_utilization", "invoice_payment_delinquency_rate", "payroll_regularity_score"]
    nonneg = ["stated_annual_revenue", "observed_monthly_revenue_avg_3mo", "vintage_years",
              "stated_time_in_business", "existing_debt_obligations", "requested_amount"]
    domain = {c: int(((tr[c] < 0) | (tr[c] > 1)).sum()) for c in bounded}
    domain.update({c: int((tr[c] < 0).sum()) for c in nonneg})
    cat_new = {}
    for c in ["sector", "geography_region", "employee_count_bucket", "intended_use_of_funds",
              "owner_personal_credit_band", "application_channel"]:
        ts = set(tr[c].dropna().unique())
        cat_new[c] = {"val_unseen": sorted(map(int, set(va[c].dropna().unique()) - ts)),
                      "test_unseen": sorted(map(int, set(te[c].dropna().unique()) - ts))}
    dup = {"exact_rows": int(tr.drop(columns=["applicant_id"]).duplicated().sum()),
           "biz_amount_ts": int(tr.duplicated(subset=["business_id", "requested_amount", "application_timestamp"]).sum())}
    return {"label_consistency": label, "logical_invariants": logical,
            "domain_violations": domain, "unseen_category_codes": cat_new, "duplicates": dup}


def boring_checks(dfs: dict) -> dict:
    tr = dfs["train"]
    dd = data.load_dictionary()
    const_cols = [c for c in tr.columns if tr[c].nunique(dropna=False) <= 1]
    dup_rows = int(tr.drop(columns=["applicant_id"]).duplicated().sum())
    # dtype mismatches vs dict
    type_map = {"int": "integer", "float": "float", "bool": "bool",
                "categorical": "categorical", "timestamp": "datetime", "string": "object"}
    mism = []
    for _, r in dd.iterrows():
        c = r["field"]
        if c not in tr:
            continue
        actual = str(tr[c].dtype)
        rows_ok = (
            (r["dtype"] == "int" and ("int" in actual or "float" in actual))
            or (r["dtype"] == "float" and "float" in actual)
            or (r["dtype"] == "bool" and ("bool" in actual or "object" in actual))
            or (r["dtype"] == "categorical")
            or (r["dtype"] == "timestamp" and "datetime" in actual)
            or (r["dtype"] == "string" and "object" in actual)
        )
        if not rows_ok:
            mism.append({"field": c, "dict": r["dtype"], "actual": actual})
    # outlier scans
    outliers = {
        "cash_balance_p10_negative_frac": _round((tr["observed_cash_balance_p10"] < 0).mean(), 4),
        "requested_amount_min": _round(tr["requested_amount"].min(), 1),
        "requested_amount_max": _round(tr["requested_amount"].max(), 1),
        "requested_amount_out_of_5k_50k_frac": _round(
            ((tr["requested_amount"] < 5000) | (tr["requested_amount"] > 50000)).mean(), 5),
        "existing_debt_p99": _round(tr["existing_debt_obligations"].quantile(0.99), 1),
        "existing_debt_max": _round(tr["existing_debt_obligations"].max(), 1),
    }
    # categorical cardinality
    cats = data.categorical_feature_columns(tr)
    card = {c: int(tr[c].nunique(dropna=True)) for c in cats}
    return {"constant_cols": const_cols, "duplicate_rows_excl_id": dup_rows,
            "dtype_mismatches": mism, "outliers": outliers, "categorical_cardinality": card,
            "prior_decision_values": sorted(tr["prior_decision"].dropna().unique().tolist())}


# --------------------------------------------------------------------------- #
# Intervention-query support analysis (Deliverable C prep)
# --------------------------------------------------------------------------- #
def intervention_support(dfs: dict) -> dict:
    q = data.load_intervention_queries()
    dd = data.load_dictionary()
    interv_dict = set(dd.loc[dd["intervenable"] == True, "field"])
    te = dfs["test"]
    qf = q["feature_name"].value_counts()
    in_support = {}
    for feat in qf.index:
        if feat not in te.columns:
            continue
        vals = pd.to_numeric(q.loc[q["feature_name"] == feat, "intervention_value"], errors="coerce")
        col = pd.to_numeric(te[feat], errors="coerce").astype(float)
        lo, hi = col.quantile(0.01), col.quantile(0.99)
        frac_in = float(((vals >= lo) & (vals <= hi)).mean())
        in_support[feat] = {"n_queries": int(len(vals)), "frac_in_1_99_support": _round(frac_in, 3)}
    return {
        "n_queries": int(len(q)),
        "n_applicants": int(q["applicant_id"].nunique()),
        "all_applicants_in_test": int(set(q["applicant_id"]) <= set(te["applicant_id"])),
        "queried_not_dict_intervenable": sorted(set(qf.index) - interv_dict),
        "feature_query_counts": qf.to_dict(),
        "support": in_support,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(dfs: dict, haz: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, va = dfs["train"], dfs["validation"]
    saved = []

    # Fig 1: prior_underwriter_score by decision (sharp vs fuzzy threshold)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    s = tr["prior_underwriter_score"]
    appr = tr["prior_decision"] == 1
    ax[0].hist(s[appr].dropna(), bins=60, alpha=0.6, label="approved", density=True)
    ax[0].hist(s[~appr].dropna(), bins=60, alpha=0.6, label="declined", density=True)
    ax[0].set_title("prior_underwriter_score by prior decision")
    ax[0].set_xlabel("score"); ax[0].legend()
    q = pd.qcut(s.rank(method="first"), 40, labels=False)
    rate = tr.groupby(q).apply(lambda g: pd.Series(
        {"score": g["prior_underwriter_score"].mean(), "appr": (g["prior_decision"] == 1).mean()}))
    ax[1].plot(rate["score"], rate["appr"], marker="o", ms=3)
    ax[1].set_title("approval rate vs score (fuzzy => smooth S-curve)")
    ax[1].set_xlabel("score bin mean"); ax[1].set_ylabel("P(approved)")
    fig.tight_layout(); p = FIGDIR / "fig01_prior_score_threshold.png"
    fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.name)

    # Fig 2: days_to_default histogram + hazard
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    d = tr.loc[data.labeled_mask(tr)]
    dtd = d.loc[d["default_flag"] == 1, "days_to_default"]
    ax[0].hist(dtd, bins=90, color="firebrick", alpha=0.8)
    ax[0].axvline(60, color="k", ls="--", lw=1, label="term=60d")
    ax[0].set_title("days_to_default (defaulters)"); ax[0].set_xlabel("day"); ax[0].legend()
    ax[1].plot(haz["day"], haz["hazard"], lw=1.2)
    ax[1].axvline(60, color="k", ls="--", lw=1)
    ax[1].set_title("discrete-time hazard h(t)"); ax[1].set_xlabel("loan age (days)")
    fig.tight_layout(); p = FIGDIR / "fig02_default_timing_hazard.png"
    fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.name)

    # Fig 3: cumulative default trajectory (the Deliverable B shape) by week-of-age
    fig, ax = plt.subplots(figsize=(7, 4.2))
    wk = haz.copy()
    wk["age_week"] = np.ceil(wk["day"] / 7).astype(int)
    cdr = wk.groupby("age_week")["cum_default_frac"].max()
    ax.step(cdr.index, cdr.values, where="post", lw=1.6)
    ax.set_title("Pooled cumulative default rate by loan-age week (Deliverable B shape)")
    ax.set_xlabel("loan age (weeks)"); ax.set_ylabel("cumulative default fraction")
    ax.set_xticks(range(1, 14))
    fig.tight_layout(); p = FIGDIR / "fig03_cumulative_trajectory.png"
    fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.name)

    # Fig 4: per-cohort default rate (val) + temporal drift
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cr = cohort_default_rates(va)
    ax.plot(cr["cohort_week"], cr["default_rate"], marker="o")
    ax.axhline(d["default_flag"].mean(), color="gray", ls="--", label="train default rate")
    ax.set_title("Val default rate by cohort week (level calibration target for B)")
    ax.set_xlabel("cohort week"); ax.set_ylabel("default rate"); ax.legend()
    fig.tight_layout(); p = FIGDIR / "fig04_cohort_default_rate.png"
    fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.name)

    # Fig 5: missingness heatmap (sampled)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = data.feature_columns()
    samp = tr[cols].sample(min(2000, len(tr)), random_state=0)
    ax.imshow(samp.isna().T, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=6)
    ax.set_title("Missingness pattern (train, 2000-row sample)"); ax.set_xlabel("rows")
    fig.tight_layout(); p = FIGDIR / "fig05_missingness_heatmap.png"
    fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.name)

    # Fig 6: stated vs observed revenue
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    dd2 = tr.loc[data.labeled_mask(tr) & (tr["has_linked_bank_feed"] == True)]
    sm = dd2["stated_annual_revenue"] / 12
    om = dd2["observed_monthly_revenue_avg_3mo"]
    ax[0].scatter(om, sm, s=3, alpha=0.2)
    lim = np.nanpercentile(np.r_[om.dropna(), sm.dropna()], 99)
    ax[0].plot([0, lim], [0, lim], "r--", lw=1)
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("observed monthly rev"); ax[0].set_ylabel("stated/12")
    ax[0].set_title("stated vs observed monthly revenue")
    ratio = (sm / om).replace([np.inf, -np.inf], np.nan).dropna()
    ax[1].hist(np.log(ratio[ratio > 0]), bins=80, color="seagreen", alpha=0.8)
    ax[1].axvline(0, color="k", ls="--"); ax[1].set_xlabel("log(stated/12 / observed)")
    ax[1].set_title("log revenue ratio (>0 = overstating)")
    fig.tight_layout(); p = FIGDIR / "fig06_revenue_discrepancy.png"
    fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.name)

    return saved


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def compute_all() -> dict:
    set_seeds()
    dfs = data.load_all()
    tr, va = dfs["train"], dfs["validation"]
    haz = discrete_hazard(tr)
    stats_ = {
        "label_coverage": label_coverage(dfs),
        "prior_decision_crosstab_train": prior_decision_crosstab(tr).to_dict(),
        "prior_decision_crosstab_val": prior_decision_crosstab(va).to_dict(),
        "approved_vs_declined_top": approved_vs_declined_smd(tr).head(20).to_dict("records"),
        "prior_score_threshold": prior_score_threshold(tr),
        "timing_structure": timing_structure(tr),
        "hazard_head": haz.head(15).to_dict("records"),
        "cohort_default_rates_val": cohort_default_rates(va).to_dict("records"),
        "timestamp_ranges": timestamp_ranges(dfs),
        "missingness_signal": missingness_signal(tr).to_dict("records"),
        "missingness_by_split_top": missingness_by_split(dfs).head(15).to_dict("records"),
        "revenue_discrepancy": revenue_discrepancy(tr),
        "recompute_ratio_check": recompute_ratio_check(tr),
        "entity_structure": entity_structure(dfs),
        "leakage_audit_top": leakage_audit(tr).head(15).to_dict("records"),
        "class_balance": class_balance(tr, ["sector", "geography_region",
                                            "owner_personal_credit_band", "application_channel"]),
        "boring_checks": boring_checks(dfs),
        "integrity_checks": integrity_checks(dfs),
        "intervention_support": intervention_support(dfs),
    }
    stats_["figures"] = make_figures(dfs, haz)
    # persist hazard + trajectory tables for reuse by Deliverable B work
    haz.to_csv(PATHS.reports / "pooled_hazard.csv", index=False)
    out = PATHS.reports / "eda_stats.json"
    out.write_text(json.dumps(stats_, indent=2, default=str))
    return stats_


if __name__ == "__main__":
    s = compute_all()
    print("EDA complete. Figures:", s["figures"])
    print("Stats written to reports/eda_stats.json")
    lc = s["label_coverage"]
    print("Label coverage:", {k: v["labeled_frac"] for k, v in lc.items()})
