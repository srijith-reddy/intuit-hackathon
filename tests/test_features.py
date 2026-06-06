"""Tests for the feature pipeline: leakage-safety, reproducibility, intervention
consistency, monotonicity. These guard the properties the deliverables depend on.
"""
import numpy as np
import pandas as pd
import pytest

from src import data, features as F
from src.config import set_seeds


@pytest.fixture(scope="module")
def fitted():
    set_seeds()
    tr = data.load_train()
    y = tr["default_flag"]
    X, art = F.build_features(tr, y=y, groups=tr["business_id"])
    return tr, y, X, art


# --------------------------------------------------------------------------- #
def test_shapes_and_consistency_across_splits(fitted):
    tr, y, Xtr, art = fitted
    va, te = data.load_validation(), data.load_test()
    Xva, _ = F.build_features(va, fit_artifacts=art)
    Xte, _ = F.build_features(te, fit_artifacts=art)
    assert list(Xtr.columns) == list(Xva.columns) == list(Xte.columns)
    assert len(Xva) == len(va) and len(Xte) == len(te)


def test_no_nan_in_indicator_and_encoded_columns(fitted):
    _, _, X, _ = fitted
    clean_prefixes = ("no_", "never_", "is_", "prior_declined", "above_", "te_", "ord_")
    clean_exact = {"revenue_overstated", "tib_inflated", "daily_payment"}
    cols = [c for c in X.columns if c.startswith(clean_prefixes) or c in clean_exact]
    nan = X[cols].isna().sum()
    assert nan.sum() == 0, f"unexpected NaN: {nan[nan > 0].to_dict()}"


def test_no_infinities(fitted):
    _, _, X, _ = fitted
    fl = X.select_dtypes(float).to_numpy()
    assert not np.isinf(fl).any()


def test_artifacts_reproduce_on_reload(fitted):
    tr, y, X, art = fitted
    p = F.save_artifacts(art)
    art2 = F.load_artifacts(p)
    X2, _ = F.build_features(tr, fit_artifacts=art2)
    # transform-mode (frozen artifacts) must match the fit-mode registry features
    # exactly for every non-OOF column (OOF target-enc differs by construction).
    reg_cols = [c for c in X.columns if not c.startswith("te_")]
    pd.testing.assert_frame_equal(X[reg_cols], X2[reg_cols], check_dtype=False)


def test_oof_target_encoding_has_no_fold_leakage(fitted):
    tr, y, X, art = fitted
    lab = y.notna()
    # full-map encoding (uses every labeled row, incl. self/fold)
    full = F._apply_target_maps(tr.loc[lab], F.NOMINAL_CATS, art["te_prior"], art["te_maps"])
    for c in F.NOMINAL_CATS:
        oof = X.loc[lab, f"te_{c}"].to_numpy()
        fullc = full[f"te_{c}"].to_numpy()
        # OOF must differ from the leaky full-fit encoding (held-out folds)
        assert not np.allclose(oof, fullc), f"{c}: OOF == full-fit (fold leakage!)"
        # but still correlated (same signal, just honestly held out). The floor is
        # modest because category default rates span a narrow band (0.11-0.19), so
        # honest fold variation lowers the correlation — that is the point.
        assert np.corrcoef(oof, fullc)[0, 1] > 0.7


def test_intervention_recompute_touches_only_descendants(fitted):
    tr, y, X, art = fitted
    samp = tr.sample(200, random_state=0)
    X0, _ = F.build_features(samp, fit_artifacts=art)
    feat = "requested_amount"
    Xcf = F.recompute_under_intervention(samp, feat, 30000.0, art)
    desc = F.descendants(feat)  # registry feature names that should move
    changed = {c for c in X0.columns if not np.allclose(
        X0[c].fillna(-999).to_numpy(), Xcf[c].fillna(-999).to_numpy())}
    # everything that changed must be a declared descendant
    assert changed <= desc, f"undeclared changes: {changed - desc}"
    # and the obvious descendants must actually change
    assert {"daily_payment", "buffer_to_payment", "log_requested_amount"} <= changed
    # a clearly-unrelated feature must NOT change
    assert "te_sector" not in changed and "prior_default_rate_shrunk" not in changed


def test_intervention_on_categorical_recomputes_target_encoding(fitted):
    tr, y, X, art = fitted
    samp = tr.sample(100, random_state=1)
    Xcf = F.recompute_under_intervention(samp, "sector", 3, art)
    # do(sector=3) -> te_sector becomes the encoded value for sector 3 on every row
    expected = art["te_maps"]["sector"].get(3, art["te_prior"])
    assert np.allclose(Xcf["te_sector"].to_numpy(), expected)


def test_monotonicity_sanity(fitted):
    _, _, _, art = fitted
    # daily_payment strictly increasing in requested_amount; buffer_to_payment
    # strictly decreasing in requested_amount (fixed positive buffer).
    base = pd.DataFrame({
        "requested_amount": [10000, 20000, 40000],
        "observed_cash_balance_p10": [5000, 5000, 5000],
        "observed_monthly_revenue_avg_3mo": [100000, 100000, 100000],
        "has_linked_bank_feed": [True, True, True],
    })
    dp = F.REGISTRY["daily_payment"].fn(base.assign(daily_payment=np.nan), art)
    assert (np.diff(dp.to_numpy()) > 0).all()
    w = base.copy(); w["daily_payment"] = dp
    b2p = F.REGISTRY["buffer_to_payment"].fn(w, art)
    assert (np.diff(b2p.to_numpy()) < 0).all()


def test_survival_long_matches_default_counts(fitted):
    tr, y, _, _ = fitted
    long = F.make_survival_long(tr)
    n_def = int((tr["default_flag"] == 1).sum())
    assert int(long["event"].sum()) == n_def
    # day-90 mass lands in week 13
    assert int(long.loc[long["loan_age_week"] == 13, "event"].sum()) > 1500
    # cumulative events are monotone non-decreasing across weeks
    cum = long.groupby("loan_age_week")["event"].sum().cumsum()
    assert (np.diff(cum.to_numpy()) >= 0).all()


def test_leakage_canary_no_single_feature_is_the_label(fitted):
    from sklearn.metrics import roc_auc_score
    tr, y, X, _ = fitted
    lab = y.notna()
    yl = y[lab].astype(int).to_numpy()
    suspicious = {}
    for c in X.columns:
        x = pd.to_numeric(X.loc[lab, c], errors="coerce")
        if x.notna().sum() < 100 or x.nunique() < 2:
            continue
        auc = roc_auc_score(yl, x.fillna(x.median()))
        auc = max(auc, 1 - auc)
        if auc > 0.95:
            suspicious[c] = round(auc, 4)
    assert not suspicious, f"possible target leakage (univariate AUC>0.95): {suspicious}"
