"""PD model: GroupKFold LightGBM ensemble + isotonic calibration + intervals.

One calibrated PD model serves all of A (decision via E[NPV]), B (level x shape),
and C (re-predict under do()). Ensemble disagreement across folds is the epistemic
uncertainty we turn into 90% intervals (calibration is 20% of the score).
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from src.config import SEED

LGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.03, num_leaves=31, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=50,
    random_state=SEED, verbose=-1, n_jobs=-1,
)


@dataclass
class PDModel:
    boosters: list
    iso: IsotonicRegression
    z90: float = 1.6448536269514722  # one-sided 95% normal quantile (90% two-sided)
    alpha: float = 1.0               # interval width scale, calibrated on val

    def predict_raw(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean PD + std (epistemic) from the fold boosters."""
        P = np.column_stack([b.predict_proba(X)[:, 1] for b in self.boosters])
        return P.mean(axis=1), P.std(axis=1)

    def predict_calibrated(self, X: pd.DataFrame, width_scale: float | None = None,
                           return_mean_std: bool = False):
        """Calibrated point PD + 90% interval [lo, hi] (monotone, in [0,1]).

        Interval is additive on the calibrated point: p ± α·z·std, where α is fit
        out-of-fold on val (so width reflects real calibration error, not in-sample).
        """
        mean, std = self.predict_raw(X)
        a = self.alpha if width_scale is None else width_scale
        p = np.clip(self.iso.transform(np.clip(mean, 0, 1)), 0, 1)
        half = a * self.z90 * std
        lo = np.clip(p - half, 0, 1); hi = np.clip(p + half, 0, 1)
        out = (p, lo, hi)
        return out + (mean, std) if return_mean_std else out


def train_pd_model(X: pd.DataFrame, y: np.ndarray, groups: pd.Series,
                   n_splits: int = 5) -> tuple[PDModel, np.ndarray]:
    """Train the fold ensemble, return (model, OOF predictions)."""
    boosters, oof = [], np.zeros(len(X))
    gkf = GroupKFold(n_splits=n_splits)
    for tr_i, va_i in gkf.split(X, y, groups):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X.iloc[tr_i], y[tr_i], eval_set=[(X.iloc[va_i], y[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        boosters.append(m)
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof, y)
    return PDModel(boosters=boosters, iso=iso), oof


def calibration_report(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(p, n_bins, duplicates="drop")
    return (pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True)
            .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).round(4))
