"""Interval calibration on validation (calibration is 20% of the score).

A: scale the PD-interval half-width so binned empirical default rates are covered
   ~90% of the time on val (the finest observable proxy for "containing the truth").
B: val is labeled, matured, and in the 13 cohort weeks, so we observe the TRUE
   cohort trajectories. We conformal-widen B's intervals from val CDR residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data


# --------------------------------------------------------------------------- #
# Deliverable A — PD interval width calibration
# --------------------------------------------------------------------------- #
def binned_coverage(p, lo, hi, y, n_bins=50) -> float:
    """Fraction of PD-bins whose empirical default rate lies in the mean interval."""
    df = pd.DataFrame({"p": p, "lo": lo, "hi": hi, "y": y})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(r=("y", "mean"), lo=("lo", "mean"),
                                             hi=("hi", "mean"))
    return float(((g["lo"] <= g["r"]) & (g["r"] <= g["hi"])).mean())


def fit_pd_interval_scale(p_cal, std, y, z=1.6448536269514722, target=0.90,
                          n_bins=10) -> tuple[float, dict]:
    """Smallest additive width-scale α s.t. decile coverage ≥ target.

    `p_cal` must be OUT-OF-FOLD calibrated PD on val (cross-fit), so the coverage we
    measure reflects genuine calibration error, not in-sample-perfect fit. Interval =
    p_cal ± α·z·std. First α clearing the target wins (not needlessly wide).
    """
    best = None
    for alpha in np.round(np.arange(0.3, 8.01, 0.1), 2):
        lo = np.clip(p_cal - alpha * z * std, 0, 1)
        hi = np.clip(p_cal + alpha * z * std, 0, 1)
        cov = binned_coverage(p_cal, lo, hi, y, n_bins)
        best = (alpha, cov, float(np.mean(hi - lo)))
        if cov >= target:
            break
    return best[0], {"coverage": round(best[1], 3), "mean_width": round(best[2], 4)}


# --------------------------------------------------------------------------- #
# Deliverable B — trajectory interval calibration from val ground truth
# --------------------------------------------------------------------------- #
def true_cohort_trajectory(val: pd.DataFrame, approved_mask: np.ndarray,
                           n_weeks: int = 13) -> dict:
    """Observed cumulative default rate CDR_{w,a} on our approved val set."""
    v = val.loc[approved_mask].copy()
    v["cohort_week"] = data.assign_cohort_week(v)
    out = {}
    for w in range(1, n_weeks + 1):
        cw = v[v["cohort_week"] == w]
        n = len(cw)
        if n == 0:
            out[w] = np.full(n_weeks, np.nan)
            continue
        dtd = pd.to_numeric(cw["days_to_default"], errors="coerce").to_numpy()
        defaulted = (cw["default_flag"] == 1).to_numpy()
        cdr = np.array([np.mean(defaulted & (dtd <= 7 * a)) for a in range(1, n_weeks + 1)])
        out[w] = cdr
    return out


def b_conformal_halfwidth(pred: dict, true: dict, target: float = 0.90,
                          n_weeks: int = 13) -> np.ndarray:
    """Per-age conformal half-width = target-quantile of |true - pred| across cohorts."""
    hw = np.zeros(n_weeks)
    for a in range(n_weeks):
        res = [abs(true[w][a] - pred[w][a]) for w in pred
               if not np.isnan(true[w][a]) and not np.isnan(pred[w][a])]
        hw[a] = float(np.quantile(res, target)) if res else 0.0
    return hw


def b_coverage(pred: dict, true: dict, lower: dict, upper: dict) -> float:
    """Fraction of (cohort,age) cells whose true val CDR is inside [lower, upper]."""
    hit = tot = 0
    for w in pred:
        for a in range(len(pred[w])):
            if np.isnan(true[w][a]):
                continue
            tot += 1
            hit += int(lower[w][a] - 1e-9 <= true[w][a] <= upper[w][a] + 1e-9)
    return hit / tot if tot else float("nan")
