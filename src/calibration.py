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
# Per-cohort PD level shrinkage for A (Iter1B)
# --------------------------------------------------------------------------- #
def per_cohort_pd_scale(va_cohort: np.ndarray, va_pred: np.ndarray, va_y: np.ndarray,
                        K: float = 75.0, n_cohorts: int = 13,
                        gap_threshold: float = 0.0) -> dict:
    """Per-cohort multiplicative scaling factors for A's PD, shrunk toward val rate.

    For each cohort w with n_w labeled approved val loans:
      shrunk_rate_w = (n_w * val_rate_w + K * pred_rate_w) / (n_w + K)
      scale_w = shrunk_rate_w / pred_rate_w
    When |val_rate - pred_rate| < gap_threshold, scale_w := 1.0 (skip small gaps so
    we don't disturb decile calibration where the per-cohort signal is noise).
    Returns {w: scale_w}. Caller applies p_adj = p * scale_w[cohort_w(loan)].
    """
    out = {}
    for w in range(1, n_cohorts + 1):
        m = va_cohort == w
        n_w = int(m.sum())
        if n_w == 0:
            out[w] = 1.0; continue
        pred_w = float(va_pred[m].mean())
        obs_w = float(va_y[m].mean())
        if pred_w <= 1e-6 or abs(obs_w - pred_w) < gap_threshold:
            out[w] = 1.0; continue
        shrunk = (n_w * obs_w + K * pred_w) / (n_w + K)
        out[w] = float(shrunk / pred_w)
    return out


# --------------------------------------------------------------------------- #
# Deliverable A — per-decile binned conformal half-widths (Iter1 replacement)
# --------------------------------------------------------------------------- #
def fit_pd_interval_binned(p_cal, y, n_bins=10, target=0.90,
                           z=1.6448536269514722) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-decile conformal half-widths at the binned-rate scale.

    Half-width for decile k = |bias_k| + z·SE_binomial_k, where
      bias_k = bin_pred − bin_obs   (signed calibration error at the bin)
      SE_k   = sqrt(p_obs(1−p_obs)/n_k)
    Inputs are CROSS-FIT calibrated PD on val. Returns:
      edges      — decile edges (length n_bins+1, with 0/1 sentinels)
      half_widths— per-decile half-width (length n_bins)
      report     — diagnostic dict.
    """
    p_cal = np.asarray(p_cal, float); y = np.asarray(y, int)
    edges = np.quantile(p_cal, np.linspace(0, 1, n_bins + 1))
    edges[0] = 0.0; edges[-1] = 1.0
    deciles = np.clip(np.digitize(p_cal, edges[1:-1]), 0, n_bins - 1)
    hw = np.zeros(n_bins); pred = np.zeros(n_bins); obs = np.zeros(n_bins); ns = np.zeros(n_bins, int)
    for k in range(n_bins):
        m = deciles == k
        n_k = int(m.sum())
        if n_k == 0:
            hw[k] = float("nan"); continue
        pk, ok = float(p_cal[m].mean()), float(y[m].mean())
        bias = abs(pk - ok)
        se = float(np.sqrt(max(ok * (1 - ok), 1e-6) / max(n_k, 1)))
        hw[k] = bias + z * se
        pred[k] = pk; obs[k] = ok; ns[k] = n_k
    # Smooth small-bin nans by neighbor average
    for k in range(n_bins):
        if np.isnan(hw[k]):
            hw[k] = np.nanmean(hw)
    report = {
        "edges": [round(float(e), 4) for e in edges],
        "n_per_bin": ns.tolist(),
        "bin_pred": [round(float(x), 4) for x in pred],
        "bin_obs": [round(float(x), 4) for x in obs],
        "half_widths": [round(float(x), 4) for x in hw],
        "mean_half_width": float(np.mean(hw)),
    }
    return edges, hw, report


def apply_binned_intervals(p, edges, half_widths) -> tuple[np.ndarray, np.ndarray]:
    """Map each scored PD into its val-decile bucket and apply the bucket's half-width."""
    p = np.asarray(p, float)
    deciles = np.clip(np.digitize(p, edges[1:-1]), 0, len(half_widths) - 1)
    hw = half_widths[deciles]
    return np.clip(p - hw, 0, 1), np.clip(p + hw, 0, 1)


# --------------------------------------------------------------------------- #
# Deliverable A — κ-shifted decision rule (uncertainty-aware approval)
# --------------------------------------------------------------------------- #
def _kappa_pnl(kappa, p, sigma, rev, exp_def, realized, idx) -> float:
    """Realized P&L on rows `idx` under approve iff E[NPV](p+κσ)>0."""
    pe = np.clip(p[idx] + kappa * sigma[idx], 0.0, 1.0)
    approve = ((1 - pe) * rev[idx] + pe * exp_def[idx]) > 0
    return float(realized[idx][approve].sum())


def fit_kappa_decision_shift(p, sigma, rev, exp_def, realized, grid=None,
                             n_folds=5, seed=0):
    """Choose κ for `approve iff E[NPV](p_i+κσ_i)>0` by realized-P&L on labeled val.

    σ is the per-loan fold-ensemble disagreement (reused from the interval model).
    Returns (kappa_star, info). kappa_star maximizes realized val P&L over the grid;
    info carries the full κ→P&L curve, the 5-fold CROSS-FIT OOF P&L of the adaptive
    rule (per-fold κ picked on the other 4 folds, scored on the held-out fold) and the
    κ=0 baseline — the cross-fit OOF guards against κ overfit to the full val set.
    """
    from sklearn.model_selection import KFold
    if grid is None:
        grid = np.round(np.arange(0.0, 3.001, 0.25), 2)
    p, sigma = np.asarray(p, float), np.asarray(sigma, float)
    rev, exp_def = np.asarray(rev, float), np.asarray(exp_def, float)
    realized = np.asarray(realized, float)
    allidx = np.arange(len(p))
    curve = {float(k): _kappa_pnl(k, p, sigma, rev, exp_def, realized, allidx) for k in grid}
    kappa_star = float(max(curve, key=curve.get))
    # cross-fit: per fold, pick κ on the train folds, score on the held-out fold
    oof, picks = 0.0, []
    for tri, tei in KFold(n_folds, shuffle=True, random_state=seed).split(allidx):
        kbest = max(grid, key=lambda k: _kappa_pnl(k, p, sigma, rev, exp_def, realized, tri))
        picks.append(float(kbest))
        oof += _kappa_pnl(kbest, p, sigma, rev, exp_def, realized, tei)
    return kappa_star, {"curve": {round(k, 2): round(v) for k, v in curve.items()},
                        "pnl_kappa0": round(curve[0.0]), "pnl_kappa_star": round(curve[kappa_star]),
                        "oof_pnl_adaptive": round(oof), "fold_picks": picks}


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


def _cohort_cdr(dtd, dflag, n_weeks=13) -> np.ndarray:
    """Realized cumulative default rate by loan-age week over a set of loans."""
    dtd = np.asarray(dtd, float); dflag = np.asarray(dflag, float)
    return np.array([np.mean((dflag == 1) & (dtd <= 7 * a)) for a in range(1, n_weeks + 1)])


def fit_shape_shrinkage_c(cohort_loans: dict, model_shape: dict, val_n: dict,
                          grid=(5, 10, 20, 35, 50, 75, 100, 150, 200),
                          n_splits: int = 10, seed: int = 0,
                          n_weeks: int = 13) -> tuple[float, dict]:
    """Pick the Dirichlet concentration c for per-cohort SHAPE shrinkage by honest
    split-half cross-fit within validation (so c is not chosen by self-prediction).

    For each cohort: estimate the empirical default-timing increments on a random half
    (A), blend toward the model band shape with concentration c
    (blended = (n_A·emp + c·model)/(n_A+c)), and score the resulting cumulative curve
    (× half-A level) against the realized CDR of the held-out half (B). c\* minimizes the
    mean held-out |CDR| error. cohort_loans[w] = (days_to_default, default_flag) arrays
    of approved labeled-val loans; model_shape[w] = model cumulative trajectory for w.
    """
    curve = {}
    for c in grid:
        # Reset RNG per c so splits are identical across grid points (otherwise
        # grid order changes the random state seen by each c and the winner shifts).
        rng = np.random.default_rng(seed)
        tot, cells = 0.0, 0
        for w, (dtd, dflag) in cohort_loans.items():
            if val_n.get(w, 0) < 20 or w not in model_shape:
                continue
            ms_cum = np.asarray(model_shape[w], float)
            if ms_cum[-1] <= 1e-6:
                continue
            ms = np.diff(ms_cum / ms_cum[-1], prepend=0.0)
            n = len(dtd)
            for _ in range(n_splits):
                idx = rng.permutation(n); h = n // 2
                A, B = idx[:h], idx[h:]
                cdr_A = _cohort_cdr(dtd[A], dflag[A], n_weeks)
                cdr_B = _cohort_cdr(dtd[B], dflag[B], n_weeks)
                lvl = cdr_A[-1]
                if lvl <= 1e-6:
                    continue
                emp = np.diff(cdr_A / lvl, prepend=0.0)
                bl = (len(A) * emp + c * ms) / (len(A) + c)
                bl = np.clip(bl, 0, None); s = bl.sum()
                if s <= 0:
                    continue
                pred_cum = np.cumsum(bl / s) * lvl
                tot += np.abs(pred_cum - cdr_B).sum(); cells += n_weeks
        curve[float(c)] = tot / cells if cells else float("inf")
    c_star = float(min(curve, key=curve.get))
    return c_star, {"loco_mae": {float(k): round(v, 5) for k, v in curve.items()}}


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
