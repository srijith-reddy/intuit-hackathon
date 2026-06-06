"""Loan-product economics → the profit-optimal decision rule for Deliverable A.

EXACT NPV from the hackathon brief (p.8), which is what the scorer uses for S_P&L:

    NPV_i = F_i + R_i r (T/365)                      if y_i = 0 (repaid)
    NPV_i = F_i + D_i (t*_i - 1) + rec_i - R_i        if y_i = 1 (default at day t*)

    R = requested amount, r = APR = 0.35, T = 60, F = 0.03 R,
    D = daily draw = R (1 + r T/365) / T, rec = recovery, t* = default day.

Decision (brief p.9): d_i = 1[ E[NPV_i | approve] > 0 ] — approve on the NPV sign,
NOT a flat PD threshold. CRUCIAL CONSEQUENCE: NPV depends on the *default day* t*
(a day-5 default loses ~R; a day-55 default nearly breaks even). So E[NPV] needs the
default-*timing* distribution, i.e. A and B share one discrete-time hazard model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PRODUCT

FEE = PRODUCT.origination_fee                      # 0.03
INT_TERM = PRODUCT.apr * PRODUCT.term_days / 365.0  # ~0.0575 simple interest over 60d
REV_RATE = FEE + INT_TERM                          # ~0.0875 of principal if paid in full
DAILY_DRAW_FRAC = (1 + INT_TERM) / PRODUCT.term_days  # D/R, ~0.017625 per day


def revenue_if_full(amount: np.ndarray | pd.Series) -> np.ndarray:
    """Lender revenue (=NPV) on a fully-repaid loan = fee + interest over the term."""
    return np.asarray(amount, float) * REV_RATE


def npv_if_default(amount, t_star, recovery=0.0) -> np.ndarray:
    """EXACT NPV of a defaulted loan: F + D(t*-1) + rec - R (brief p.8).

    t_star is the default day in [1,90]; draws collected up to t*-1 are credited
    (so late defaults are far less costly — even positive for very late t*).
    """
    amount = np.asarray(amount, float)
    t_star = np.asarray(t_star, float)
    recovery = np.asarray(recovery, float)
    D = amount * DAILY_DRAW_FRAC
    return FEE * amount + D * (t_star - 1) + recovery - amount


def expected_npv(amount, pd_hat, default_day_dist=None, mean_default_day=None,
                 recovery_frac=0.0) -> np.ndarray:
    """E[NPV | approve] = (1-PD)·rev + Σ_t Pr(default day=t)·NPV_default(t).

    Provide either `default_day_dist` (array of Pr over days 1..90, per-loan or shared)
    or a scalar/array `mean_default_day` (cruder: plugs the mean into the convex NPV).
    Prefer the full distribution — NPV is nonlinear in t*, and the day-90 mass matters.
    """
    amount = np.asarray(amount, float)
    pd_hat = np.asarray(pd_hat, float)
    rev = revenue_if_full(amount)
    rec = recovery_frac * amount if np.isscalar(recovery_frac) else np.asarray(recovery_frac, float) * amount
    if default_day_dist is not None:
        days = np.arange(1, default_day_dist.shape[-1] + 1)
        npv_def = npv_if_default(amount[:, None], days[None, :], rec[:, None]) if amount.ndim else \
            npv_if_default(amount, days, rec)
        exp_npv_def = (default_day_dist * npv_def).sum(axis=-1)
        return (1 - pd_hat) * rev + exp_npv_def
    t_bar = mean_default_day
    return (1 - pd_hat) * rev + pd_hat * npv_if_default(amount, t_bar, rec)


def lgd_components(df_labeled: pd.DataFrame, mode: str = "draws_aware") -> pd.Series:
    """Per-default loss-given-default fraction in [0,1].

    mode='draws_aware'  : credit daily ACH draws collected before default day
                          (lender actually received this cash) + final recovery.
    mode='conservative' : count only post-hoc final_recovered_amount.
    """
    d = df_labeled.loc[df_labeled["default_flag"] == 1]
    amt = d["requested_amount"].to_numpy(float)
    rec = d["final_recovered_amount"].fillna(0).to_numpy(float)
    if mode == "conservative":
        loss = np.clip(amt - rec, 0, None)
    elif mode == "draws_aware":
        # EXACT brief formula: loss = R - F - D(t*-1) - rec  (clip at 0; late t* can be +NPV)
        daily = amt * DAILY_DRAW_FRAC
        dtd = d["days_to_default"].to_numpy(float)
        draws = np.clip(daily * (dtd - 1), 0, None)
        loss = np.clip(amt - FEE * amt - draws - rec, 0, None)
    else:
        raise ValueError(mode)
    return pd.Series(loss / amt, index=d.index, name=f"lgd_{mode}")


def break_even_pd(lgd: float) -> float:
    """PD above which expected profit per dollar turns negative.

    E[profit]/amount = (1-pd)*REV_RATE - pd*lgd  > 0  ->  pd < REV_RATE/(REV_RATE+lgd).
    """
    return REV_RATE / (REV_RATE + lgd)


def expected_profit(pd_hat, amount, lgd: float) -> np.ndarray:
    """Per-loan expected profit under PD estimate, amount, and an LGD assumption."""
    pd_hat = np.asarray(pd_hat, float)
    amount = np.asarray(amount, float)
    return (1 - pd_hat) * amount * REV_RATE - pd_hat * amount * lgd


def decision_from_pd(pd_hat, amount, lgd: float) -> np.ndarray:
    """Approve (1) iff expected profit > 0 (i.e. pd_hat < break-even PD)."""
    return (expected_profit(pd_hat, amount, lgd) > 0).astype(int)


def profit_curve(pd_true, amount, pd_hat=None, lgd: float = 0.5, n: int = 101) -> pd.DataFrame:
    """Realized portfolio profit as we sweep the PD approval threshold.

    pd_true : realized default outcome (0/1) used to bill profit/loss.
    pd_hat  : score we threshold on (defaults to pd_true for an oracle curve).
    """
    pd_true = np.asarray(pd_true, float)
    amount = np.asarray(amount, float)
    score = pd_true if pd_hat is None else np.asarray(pd_hat, float)
    rev = amount * REV_RATE
    loss = amount * lgd
    per_loan = np.where(pd_true == 0, rev, -loss)  # realized profit if approved
    rows = []
    for thr in np.linspace(0, 1, n):
        approve = score < thr
        rows.append({"threshold": thr, "n_approved": int(approve.sum()),
                     "approve_rate": float(approve.mean()),
                     "total_profit": float(per_loan[approve].sum()),
                     "profit_per_approved": float(per_loan[approve].mean()) if approve.any() else 0.0,
                     "default_rate_approved": float(pd_true[approve].mean()) if approve.any() else 0.0})
    return pd.DataFrame(rows)


def summary(df_labeled: pd.DataFrame) -> dict:
    out = {"rev_rate": REV_RATE, "mean_revenue_if_full": float(revenue_if_full(df_labeled["requested_amount"]).mean())}
    for mode in ("draws_aware", "conservative"):
        lgd = float(lgd_components(df_labeled, mode).mean())
        out[mode] = {"lgd": round(lgd, 3), "break_even_pd": round(break_even_pd(lgd), 3)}
    out["portfolio_default_rate"] = round(float(df_labeled["default_flag"].mean()), 3)
    return out
