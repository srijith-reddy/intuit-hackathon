"""Default-timing structure shared by Deliverables A (E[NPV]) and B (trajectory).

Keystone insight (brief p.9): NPV depends on the default *day*, and B is the
cumulative-default curve, so both consume one timing model. We model the per-loan
cumulative curve as F_i(t) = PD_i * S(t), where S(t) is the canonical normalized
shape learned on train (S(90)=1). The shape is ~invariant across risk segments
(validated in notebook 02), so "shape x level" is robust and monotone by design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data, features as F


def daily_shape(train: pd.DataFrame, n_days: int = 90) -> np.ndarray:
    """S(t) for t=1..90, normalized cumulative default fraction (S(90)=1)."""
    haz = F.make_survival_long  # noqa  (keep import warm)
    d = train.loc[data.labeled_mask(train)]
    dtd = d.loc[d["default_flag"] == 1, "days_to_default"].to_numpy()
    cum = np.array([(dtd <= t).sum() for t in range(1, n_days + 1)], float)
    return cum / cum[-1] if cum[-1] > 0 else cum


def weekly_shape(train: pd.DataFrame, n_weeks: int = 13) -> np.ndarray:
    """S(a) for loan-age weeks a=1..13 (day 7a), normalized to S(13)=1."""
    s = daily_shape(train)
    days = np.minimum(7 * np.arange(1, n_weeks + 1), 90)
    return s[days - 1]


def mean_default_day(train: pd.DataFrame) -> float:
    """E[t* | default] — the linear-NPV summary of timing (brief NPV is linear in t*)."""
    d = train.loc[data.labeled_mask(train)]
    return float(d.loc[d["default_flag"] == 1, "days_to_default"].mean())


def mean_recovery_frac(train: pd.DataFrame) -> float:
    """E[recovery / amount | default] — expected recovery fraction for E[NPV]."""
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    return float((d["final_recovered_amount"].fillna(0) / d["requested_amount"]).mean())


# --------------------------------------------------------------------------- #
# E3 — segment-conditional timing (shape varies ~13 days by credit band).
# Risk-segment = owner_personal_credit_band (observed, clean). Worse credit
# defaults earlier; better credit carries more day-90 mass. Pooled shape mis-times
# per-segment, so B and A use a band-conditional shape, falling back to pooled.
# --------------------------------------------------------------------------- #
SEG_COL = "owner_personal_credit_band"


def weekly_shape_by_band(train: pd.DataFrame, n_weeks: int = 13) -> dict:
    """{band: S_band(a) normalized to 1 at week 13}; plus 'pooled' fallback."""
    out = {"pooled": weekly_shape(train, n_weeks)}
    d = train.loc[data.labeled_mask(train)]
    for b, g in d.groupby(SEG_COL):
        dd = g.loc[g["default_flag"] == 1, "days_to_default"].to_numpy()
        if len(dd) < 100:
            continue
        cum = np.array([(dd <= 7 * a).sum() for a in range(1, n_weeks + 1)], float)
        cum = cum / cum[-1] if cum[-1] > 0 else cum
        out[int(b)] = cum
    return out


def mean_default_day_by_band(train: pd.DataFrame) -> dict:
    """{band: E[t*|default, band]}; plus 'pooled' fallback (for A's E[NPV])."""
    out = {"pooled": mean_default_day(train)}
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    for b, g in d.groupby(SEG_COL):
        if len(g) < 100:
            continue
        out[int(b)] = float(g["days_to_default"].mean())
    return out


def daily_dist_by_band(train: pd.DataFrame, n_days: int = 90) -> dict:
    """{band: w_b(t) for t=1..90, P(default day=t | default, band), sums to 1};
    plus 'pooled' fallback. Used for A's EXACT timing integration of E[NPV]:
    E[NPV]=(1-p)*rev + p*Σ_t w_b(t)*NPV_default(t). Since brief NPV is linear in t*,
    Σ_t w_b(t)*NPV_default(t) == NPV_default(E_b[t]) == the daily-mean plug-in; this
    form is the exact expectation and makes the day-90 mass explicit."""
    def _dist(dd: np.ndarray) -> np.ndarray:
        dd = dd[np.isfinite(dd)]
        h = np.array([(dd == t).sum() for t in range(1, n_days + 1)], float)
        s = h.sum()
        return h / s if s > 0 else h
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    out = {"pooled": _dist(pd.to_numeric(d["days_to_default"], errors="coerce").to_numpy(float))}
    for b, g in d.groupby(SEG_COL):
        if len(g) < 100:
            continue
        out[int(b)] = _dist(pd.to_numeric(g["days_to_default"], errors="coerce").to_numpy(float))
    return out


def band_lookup(df: pd.DataFrame, table: dict, default_key: str = "pooled"):
    """Map each row's credit band to its entry in `table` (object array)."""
    bands = pd.to_numeric(df[SEG_COL], errors="coerce")
    return [table.get(int(b), table[default_key]) if pd.notna(b) else table[default_key]
            for b in bands]


# --------------------------------------------------------------------------- #
# Two-mode shape (Iter2): early-default cumulative + day-90 point mass.
# Brief structural observation: defaults are bimodal — days 3-60 (missed-draw,
# 77.5% of defaults) then exactly zero between 61-89 then day-90 sweep (22.5%).
# This is structurally exact in the data; pooled or band-conditional shapes
# smooth it. Per-loan: CDR_i(a) = P_early_i · F_early(a)  for a in [1,12]
#                     CDR_i(13) = P_early_i + P_d90_i = PD_i
# where F_early(a) is cumulative early-default fraction at day 7a (F_early(9)=1).
# --------------------------------------------------------------------------- #
def early_cumulative_shape(train: pd.DataFrame, n_weeks: int = 13) -> np.ndarray:
    """F_early(a) for a=1..13: fraction of EARLY defaults (days 3-60) seen by day 7a,
    normalized to 1 at week 9 (day 63 — past the early window). Weeks 9-13 all == 1."""
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    dtd = pd.to_numeric(d["days_to_default"], errors="coerce").to_numpy()
    early = dtd[dtd <= 60]
    if len(early) == 0:
        return np.ones(n_weeks)
    out = np.zeros(n_weeks)
    for a in range(1, n_weeks + 1):
        cap = min(7 * a, 60)
        out[a - 1] = (early <= cap).sum() / len(early)
    return out  # weeks 9..13 == 1 by construction (cap clipped to 60)


def mean_early_default_day(train: pd.DataFrame) -> float:
    """E[t* | default, t* <= 60] — the early-window mean day (pooled fallback)."""
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    dtd = pd.to_numeric(d["days_to_default"], errors="coerce").to_numpy()
    early = dtd[dtd <= 60]
    return float(early.mean()) if len(early) else 32.0


def mean_early_default_day_by_band(train: pd.DataFrame) -> dict:
    """{band: E[t*|default, t*<=60, band]}; plus 'pooled' fallback."""
    out = {"pooled": mean_early_default_day(train)}
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    for b, g in d.groupby(SEG_COL):
        dd = pd.to_numeric(g["days_to_default"], errors="coerce").to_numpy()
        early = dd[dd <= 60]
        if len(early) < 50:
            continue
        out[int(b)] = float(early.mean())
    return out


def early_cumulative_shape_by_band(train: pd.DataFrame, n_weeks: int = 13) -> dict:
    """Per-band F_early(a) (week-9-and-later == 1). Falls back to 'pooled' for unseen bands."""
    out = {"pooled": early_cumulative_shape(train, n_weeks)}
    d = train.loc[data.labeled_mask(train) & (train["default_flag"] == 1)]
    for b, g in d.groupby(SEG_COL):
        dd = pd.to_numeric(g["days_to_default"], errors="coerce").to_numpy()
        early = dd[dd <= 60]
        if len(early) < 50:
            continue
        cum = np.zeros(n_weeks)
        for a in range(1, n_weeks + 1):
            cap = min(7 * a, 60)
            cum[a - 1] = (early <= cap).sum() / len(early)
        out[int(b)] = cum
    return out
