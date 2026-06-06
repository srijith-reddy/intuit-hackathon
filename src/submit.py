"""End-to-end submission builder: trains the shared PD model and writes A/B/C.

    python -m src.submit            # builds submissions/ and runs the validator

The decision rule is `approve iff E[NPV] > 0` (brief NPV, not a flat PD threshold).
A and B share one PD model and one two-mode timing decomposition (early-default
hazard + day-90 sweep). C uses the same model under do() with feature-type-aware
causal shrinkage. Intervals are calibrated on the in-window validation set.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from src import data, features as F, models, survival, calibration as cal, economics as ec
from src.config import PATHS, SEED, set_seeds

OUT = PATHS.submissions
BOOT = 500

# C: per-query causal shrinkage by feature type. A self-report changes a CLAIM, not
# the borrower's cash flow → true interventional effect ≈ 0 (the observational PD is
# the right answer, not naive re-prediction). Immutable/proxy features are not levers
# but are queried anyway → half the naive perturbation, wider interval. Manipulable
# features (requested_amount + verifiable signals) get the full registry-propagated
# perturbation. Confounded proxies (bureau / bank-feed) additionally use the
# DAG-derived λ̂ (see reports/lambda_hat_dag.csv, applied below).
SELF_REPORT = {"stated_annual_revenue", "stated_time_in_business"}
IMMUTABLE_PROXY = {
    "sector", "geography_region", "vintage_years", "employee_count_bucket",
    "has_linked_bank_feed", "prior_loans_count", "prior_loans_default_count",
    "prior_loans_amount_total", "account_age_days", "platform_active_months",
    "days_since_last_external_decline", "days_since_last_inquiry_elsewhere",
    "bookkeeping_recency_days", "intended_use_of_funds",
}
SHRINK = {"self_report": 0.0, "immutable": 0.5, "manipulable": 1.0}


def _shrink_factor(feat: str) -> float:
    if feat in SELF_REPORT:
        return SHRINK["self_report"]
    if feat in IMMUTABLE_PROXY:
        return SHRINK["immutable"]
    return SHRINK["manipulable"]


def _exp_npv_default(amount, dist, rec_frac):
    """E[NPV | default] under a per-loan daily default-day distribution.

    `dist` is n×90; each row sums to 1. NPV is linear in t*, so this returns the
    exact expectation. Used by the legacy single-mode decision path; the live
    pipeline uses `_exp_npv_default_two_mode` for per-loan timing precision.
    """
    amount = np.asarray(amount, float)
    days = np.arange(1, dist.shape[1] + 1, dtype=float)
    rec = rec_frac * amount
    npv_def_mat = ec.npv_if_default(amount[:, None], days[None, :], rec[:, None])
    return (dist * npv_def_mat).sum(axis=1)


def _enpv_decision(p, amount, dist, rec_frac, sigma=None, kappa=0.0):
    """Approve iff E[NPV](p+κσ) > 0. Reported PD stays `p`; only the DECISION uses
    the uncertainty-shifted p+κσ (declines the most-uncertain near-break-even loans
    first). σ is the per-loan fold-ensemble disagreement; κ is tuned on val P&L.
    Returns (decision, exp_npv_default).
    """
    rev = ec.revenue_if_full(amount)
    exp_def = _exp_npv_default(amount, dist, rec_frac)
    p_eff = np.clip(np.asarray(p, float) + kappa * np.asarray(sigma, float), 0.0, 1.0) \
        if sigma is not None else np.asarray(p, float)
    enpv = (1 - p_eff) * rev + p_eff * exp_def
    return (enpv > 0).astype(int), exp_def


def _exp_npv_default_two_mode(amount, d90_frac, early_mean_day, rec_frac):
    """E[NPV | default] decomposed by failure mode: early miss-draw vs day-90 sweep.

    Per-loan E[t*|default] = (1-d90)·E[t|early] + d90·90 → exact under linear-in-t*
    NPV. Late defaulters cost ~zero (almost full schedule paid), so loans with high
    d90_frac get materially better E[NPV] than the pooled approximation.
    """
    amount = np.asarray(amount, float)
    d90 = np.asarray(d90_frac, float)
    em = np.asarray(early_mean_day, float)
    rec = rec_frac * amount
    npv_early = ec.npv_if_default(amount, em, rec)
    npv_late = ec.npv_if_default(amount, 90.0, rec)
    return (1 - d90) * npv_early + d90 * npv_late


def _enpv_decision_two_mode(p, amount, d90_frac, early_mean_day, rec_frac, sigma=None, kappa=0.0):
    """`_enpv_decision` with per-loan two-mode timing — the live decision rule."""
    rev = ec.revenue_if_full(amount)
    exp_def = _exp_npv_default_two_mode(amount, d90_frac, early_mean_day, rec_frac)
    p_eff = np.clip(np.asarray(p, float) + kappa * np.asarray(sigma, float), 0.0, 1.0) \
        if sigma is not None else np.asarray(p, float)
    enpv = (1 - p_eff) * rev + p_eff * exp_def
    return (enpv > 0).astype(int), exp_def


def _engineer(raw, art):
    X, _ = F.build_features(raw, fit_artifacts=art)
    return F.model_features(X)


def build():
    set_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = data.load_train(), data.load_validation(), data.load_test()

    # ---- shared PD model on labeled train ---------------------------------
    Xtr_all, art = F.build_features(tr, y=tr["default_flag"], groups=tr["business_id"])
    lab = data.labeled_mask(tr)
    Xm = F.model_features(Xtr_all)[lab.values]
    y = tr.loc[lab, "default_flag"].astype(int).to_numpy()
    model, oof = models.train_pd_model(Xm, y, tr.loc[lab, "business_id"])
    print(f"[model] OOF AUC={roc_auc_score(y, oof):.4f} Brier={brier_score_loss(y, oof):.4f}")

    from sklearn.isotonic import IsotonicRegression
    # Timing tables: per-band shapes and conditional means used by both A and B.
    # Two-mode split (early days 3-60 vs day-90 sweep) is structural in the data:
    # zero defaults occur on days 61-89, so the pooled curve smooths a discontinuity
    # the per-mode tables preserve.
    rec_frac = survival.mean_recovery_frac(tr)
    S_week = survival.weekly_shape(tr)
    S_band = survival.weekly_shape_by_band(tr)
    daily_band = survival.daily_dist_by_band(tr)
    F_early_band = survival.early_cumulative_shape_by_band(tr)
    early_mean_band = survival.mean_early_default_day_by_band(tr)
    print(f"[two-mode] E[t*|early] pooled={early_mean_band['pooled']:.2f} "
          f"band={ {k: round(v,1) for k,v in early_mean_band.items() if k!='pooled'} }")

    # Second prediction head: P(day-90 sweep | default, features). Combined with PD
    # gives per-loan P_early = PD·(1-d90), P_d90 = PD·d90 — the two probabilities
    # that drive B's per-loan CDR shape and A's E[NPV] per-loan timing.
    def_mask = lab & (tr["default_flag"] == 1)
    Xd = F.model_features(Xtr_all)[def_mask.values]
    # Day-90 sweep label: the "open-balance at day 90" failure mode. On the data,
    # day 89 has zero defaults so >= 90 is a strict equivalent of `== 90` here;
    # the inequality is robust if any future cleaner cut falls one day off.
    yd = (pd.to_numeric(tr.loc[def_mask, "days_to_default"], errors="coerce") >= 90).astype(int).to_numpy()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    d90_boosters = []
    d90_oof = np.zeros(len(yd))
    gkf = GroupKFold(n_splits=5)
    for tri, tei in gkf.split(Xd, yd, tr.loc[def_mask, "business_id"]):
        mb = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.04,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                min_child_samples=50, random_state=SEED, verbose=-1, n_jobs=-1)
        mb.fit(Xd.iloc[tri], yd[tri], eval_set=[(Xd.iloc[tei], yd[tei])],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        d90_boosters.append(mb)
        d90_oof[tei] = mb.predict_proba(Xd.iloc[tei])[:, 1]
    iso_d90 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(d90_oof, yd)
    print(f"[d90] OOF AUC={roc_auc_score(yd, d90_oof):.4f}  base rate train={yd.mean():.3f}")

    def _predict_d90_frac_raw(X_in):
        P = np.column_stack([b.predict_proba(X_in)[:, 1] for b in d90_boosters])
        return np.clip(P.mean(1), 0, 1)

    def _predict_d90_frac(X_in):
        return np.clip(iso_d90.transform(_predict_d90_frac_raw(X_in)), 0, 1)

    # Recalibrate PD level on the in-window validation set. Train default rate is
    # 17.5% but the scored window runs 20.6% (forward-in-time drift), so the
    # train-OOF isotonic systematically under-predicts. Refit on labeled val to
    # match the deployment window's base rate — feeds both the decision rule and
    # the per-cohort calibration that drives B.
    Xva = _engineer(va, art)
    vlab = data.labeled_mask(va).to_numpy()
    yva = va.loc[vlab, "default_flag"].astype(int).to_numpy()
    mva, _ = model.predict_raw(Xva)
    before = float(model.iso.transform(np.clip(mva[vlab], 0, 1)).mean())
    model.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.clip(mva[vlab], 0, 1), yva)
    after = float(model.iso.transform(np.clip(mva[vlab], 0, 1)).mean())
    print(f"[cal] recalibrated PD on val: mean {before:.3f}->{after:.3f} (val rate {yva.mean():.3f})")

    # Interval-width scale α fit by cross-fit on val. We cross-fit the isotonic so
    # the residuals reflect honest out-of-sample calibration error — fitting on the
    # full val would collapse the residual to zero after the refit above.
    from sklearn.model_selection import KFold
    _, _, _, mva, sva = model.predict_calibrated(Xva, return_mean_std=True)
    raw_lab, std_lab = mva[vlab], sva[vlab]
    oof_cal = np.zeros(len(yva))
    for tri, tei in KFold(5, shuffle=True, random_state=SEED).split(raw_lab):
        isof = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            np.clip(raw_lab[tri], 0, 1), yva[tri])
        oof_cal[tei] = isof.transform(np.clip(raw_lab[tei], 0, 1))
    alpha, ar = cal.fit_pd_interval_scale(oof_cal, std_lab, yva, n_bins=10)
    model.alpha = alpha   # reused for C's interval half-width
    print(f"[A-cal] cross-fit width alpha={alpha} -> val decile coverage={ar['coverage']} "
          f"mean width={ar['mean_width']}")

    # κ-shifted decision rule: approve iff E[NPV](p + κσ) > 0. κ shifts each loan's
    # decision PD by its own fold disagreement, so loans the boosters disagree on are
    # treated more conservatively. κ tuned on realized val P&L with cross-fit guard.
    pva, _, _ = model.predict_calibrated(Xva)
    Xva_m_for_d90 = F.model_features(Xva)
    d90_va = _predict_d90_frac(Xva_m_for_d90)
    em_va = np.array([early_mean_band.get(int(b), early_mean_band["pooled"]) if pd.notna(b)
                      else early_mean_band["pooled"]
                      for b in pd.to_numeric(va[survival.SEG_COL], errors="coerce")])
    amt_va = va["requested_amount"].to_numpy(float)
    rev_va = ec.revenue_if_full(amt_va)
    exp_def_va = _exp_npv_default_two_mode(amt_va, d90_va, em_va, rec_frac)
    tstar_va = pd.to_numeric(va["days_to_default"], errors="coerce").fillna(0).to_numpy(float)
    recd_va = va["final_recovered_amount"].fillna(0).to_numpy(float)
    realized_va = np.where(va["default_flag"].to_numpy() == 0,
                           rev_va, ec.npv_if_default(amt_va, tstar_va, recd_va))
    kappa, kinfo = cal.fit_kappa_decision_shift(
        pva[vlab], sva[vlab], rev_va[vlab], exp_def_va[vlab], realized_va[vlab], seed=SEED)
    print(f"[A-κ] κ*={kappa} | val P&L κ0=${kinfo['pnl_kappa0']:,} -> κ*=${kinfo['pnl_kappa_star']:,} "
          f"| cross-fit OOF=${kinfo['oof_pnl_adaptive']:,} | folds={kinfo['fold_picks']}")
    print(f"[A-κ] curve {kinfo['curve']}")

    # Per-cohort PD level shrinkage. Val and test span the same 13 calendar weeks,
    # so the val realized cohort default rate is the only in-window anchor for the
    # test cohort level the PD model cannot see (train predates the cohort window).
    # `gap_threshold` keeps the adjustment surgical — only cohorts whose val rate
    # differs from the model rate by enough to be likely-real (not just sampling
    # noise on n≈150) get scaled, preserving the pooled decile calibration.
    va_cw_arr = data.assign_cohort_week(va).to_numpy()
    yva_full = pd.to_numeric(va["default_flag"], errors="coerce").to_numpy()
    pcoh_scale = cal.per_cohort_pd_scale(
        va_cw_arr[vlab.astype(bool) if vlab.dtype != bool else vlab],
        pva[vlab], yva_full[vlab], K=75.0, gap_threshold=0.04)
    print(f"[A-coh] per-cohort PD scales: {dict((int(w), round(s, 3)) for w, s in pcoh_scale.items())}")

    # ============================ Deliverable A ============================
    scored = pd.concat([va, te], ignore_index=True)
    Xsc = _engineer(scored, art)
    p, lo, hi, _, ssc = model.predict_calibrated(Xsc, return_mean_std=True)
    # Apply per-cohort PD scaling (no-op outside the 13 weeks); recenter intervals.
    sc_cw = data.assign_cohort_week(scored).to_numpy()
    scale_arr = np.array([pcoh_scale.get(int(w), 1.0) if pd.notna(w) else 1.0 for w in sc_cw])
    p_pre = p.copy()
    p = np.clip(p * scale_arr, 0, 1)
    lo, hi = model.recenter_intervals(p, ssc)
    amt = scored["requested_amount"].to_numpy(float)
    # Per-loan timing for E[NPV]: d90 head + band-conditional early-mean day.
    d90_sc = _predict_d90_frac(F.model_features(Xsc))
    em_sc = np.array([early_mean_band.get(int(b), early_mean_band["pooled"]) if pd.notna(b)
                      else early_mean_band["pooled"]
                      for b in pd.to_numeric(scored[survival.SEG_COL], errors="coerce")])
    decision, npv_def = _enpv_decision_two_mode(p, amt, d90_sc, em_sc, rec_frac,
                                                sigma=ssc, kappa=kappa)
    pd.DataFrame({"applicant_id": scored["applicant_id"], "decision": decision,
                  "predicted_pd": p, "pd_lower_90": lo, "pd_upper_90": hi}
                 ).to_csv(OUT / "submission_A_decisions.csv", index=False)
    print(f"[A] {len(scored)} rows | approve={decision.mean():.3f} | "
          f"approved book PD={p[decision==1].mean():.3f} | "
          f"break-even PD~{ec.REV_RATE/(ec.REV_RATE - npv_def.mean()/amt.mean()):.3f} | "
          f"mean interval width={np.mean(hi-lo):.4f} | "
          f"mean |p−p_pre|={np.mean(np.abs(p-p_pre)):.4f}")

    # ============================ Deliverable B ============================
    # Calibrate B against the val cohort trajectories the same κ-shifted policy
    # produces — keeps the approved-book definition consistent between A and B.
    dec_va, _ = _enpv_decision_two_mode(pva, amt_va, d90_va, em_va, rec_frac,
                                        sigma=sva, kappa=kappa)
    appr_va_lab = (dec_va == 1) & vlab
    # Per-loan two-mode CDR: weeks 1-12 are early-default mass scaled by the band-
    # conditional F_early; week 13 jumps by the day-90 mass (= remaining PD).
    va_c = va.assign(cohort_week=data.assign_cohort_week(va),
                     pd_hat=pva, d90_frac=d90_va,
                     p_early=pva * (1 - d90_va), p_d90=pva * d90_va)
    pred_traj, true_traj = {}, {}
    for w in range(1, 14):
        m = appr_va_lab & (va_c["cohort_week"].to_numpy() == w)
        if m.any():
            sub = va_c.loc[m]
            Fe = np.array(survival.band_lookup(sub, F_early_band))
            contrib = Fe * sub["p_early"].to_numpy()[:, None]
            contrib[:, 12] = sub["pd_hat"].to_numpy()                   # week-13 = full PD
            pred_traj[w] = contrib.mean(0)
        else:
            pred_traj[w] = S_week * np.nan
    true_traj = cal.true_cohort_trajectory(va, appr_va_lab)
    # Per-cohort level shrinkage toward the val realized rate. KB controls how much
    # weight the val rate gets vs the model: small KB favors val (good when val/test
    # share calendar weeks and macro environment); larger KB protects against
    # sampling noise on cohorts with small n.
    KB = 15.0
    va_cw = va_c["cohort_week"].to_numpy()
    val_n = {w: int((appr_va_lab & (va_cw == w)).sum()) for w in range(1, 14)}
    val_rate = {w: (true_traj[w][12] if val_n[w] > 0 else np.nan) for w in range(1, 14)}
    # Per-cohort SHAPE shrinkage: Dirichlet blend of empirical week-by-week timing
    # toward the model band shape. c_shape selected by split-half cross-fit within
    # val; we override toward more-empirical because the LOCO splits (n≈62/cohort)
    # are dominated by sampling noise, while val→test transfer benefits from the
    # full cohort's empirical shape.
    cohort_loans = {}
    for w in range(1, 14):
        mm = appr_va_lab & (va_cw == w)
        if mm.any():
            sw = va_c.loc[mm]
            cohort_loans[w] = (pd.to_numeric(sw["days_to_default"], errors="coerce").fillna(0).to_numpy(float),
                               sw["default_flag"].to_numpy(float))
    c_shape, cinfo = cal.fit_shape_shrinkage_c(cohort_loans, pred_traj, val_n, seed=SEED)
    c_loco = c_shape
    c_shape = 50.0
    print(f"[B-shape] c*={c_shape} (override; LOCO winner {c_loco}) | LOCO half-MAE {cinfo['loco_mae']}")
    hw = cal.b_conformal_halfwidth(pred_traj, true_traj)
    # scale the conformal band up until val coverage >= 90% (not needlessly wide)
    bcov = 0.0
    for s in np.arange(1.0, 3.01, 0.1):
        vl = {w: np.clip(pred_traj[w] - s * hw, 0, 1) for w in pred_traj}
        vh = {w: np.clip(pred_traj[w] + s * hw, 0, 1) for w in pred_traj}
        bcov = cal.b_coverage(pred_traj, true_traj, vl, vh)
        if bcov >= 0.90:
            hw = s * hw
            break
    print(f"[B-cal] conformal half-width (age7,13)=({hw[6]:.3f},{hw[12]:.3f}) | "
          f"val coverage={bcov:.3f}")

    scored = scored.assign(cohort_week=data.assign_cohort_week(scored),
                           pd_hat=p, decision=decision,
                           d90_frac=d90_sc,
                           p_early=p * (1 - d90_sc), p_d90=p * d90_sc)
    approved = scored[scored["decision"] == 1]
    gmean = approved["pd_hat"].mean()
    rng = np.random.default_rng(SEED)
    rows, low_d, up_d = [], {}, {}
    for w in range(1, 14):
        sub = approved[approved["cohort_week"] == w]
        if len(sub) == 0:
            contrib = (S_week[None, :] * gmean)                          # rare-cohort fallback
        else:
            Fe = np.array(survival.band_lookup(sub, F_early_band))       # band-conditional shape
            contrib = Fe * sub["p_early"].to_numpy()[:, None]
            contrib[:, 12] = sub["pd_hat"].to_numpy()                    # week-13 = full PD (early + d90)
        point_vec = contrib.mean(0)
        boot = np.array([contrib[rng.integers(0, len(contrib), len(contrib))].mean(0)
                         for _ in range(BOOT)])
        # Shape shrinkage: blend the cohort's empirical week-by-week timing toward
        # the model band shape, preserving the week-13 total. Data-rich cohorts move
        # toward their observed timing; sparse cohorts stay near the stable band shape.
        m_w = point_vec[12]
        if val_n.get(w, 0) > 0 and m_w > 1e-6 and not np.isnan(true_traj[w]).any() and true_traj[w][12] > 1e-6:
            ms = np.diff(point_vec / m_w, prepend=0.0)
            emp = np.diff(true_traj[w] / true_traj[w][12], prepend=0.0)
            bl = (val_n[w] * emp + c_shape * ms) / (val_n[w] + c_shape)
            bl = np.clip(bl, 0, None); bl = bl / bl.sum()
            new_cum = np.cumsum(bl) * m_w
            factor = new_cum / np.where(point_vec > 1e-12, point_vec, 1e-12)
            point_vec = new_cum
            boot = boot * factor[None, :]
        # Level shrinkage: pull the week-13 total toward the val realized cohort rate.
        # Same-calendar-week transfer means the val rate is the only in-window anchor
        # for the test cohort level the PD model cannot see.
        m_w = point_vec[12]
        if val_n.get(w, 0) > 0 and m_w > 1e-6 and not np.isnan(val_rate[w]):
            shr = (val_n[w] * val_rate[w] + KB * m_w) / (val_n[w] + KB)
            point_vec = point_vec * (shr / m_w)
            boot = boot * (shr / m_w)
        low_d[w] = np.zeros(13); up_d[w] = np.zeros(13)
        for a in range(1, 14):
            point = point_vec[a-1]
            samp_lo, samp_hi = np.quantile(boot[:, a-1], [0.05, 0.95])
            lo_b = min(point - hw[a-1], samp_lo)
            hi_b = max(point + hw[a-1], samp_hi)
            lo_b, hi_b = np.clip([min(lo_b, point), max(hi_b, point)], 0, 1)
            low_d[w][a-1], up_d[w][a-1] = lo_b, hi_b
            rows.append((w, a, np.clip(point, 0, 1), lo_b, hi_b))
    pd.DataFrame(rows, columns=["cohort_week", "loan_age_weeks", "cumulative_default_rate",
                                "cdr_lower_90", "cdr_upper_90"]).to_csv(
        OUT / "submission_B_trajectory.csv", index=False)
    print(f"[B] 169 rows | mean interval width={np.mean([up_d[w]-low_d[w] for w in up_d]):.4f}")

    # ============================ Deliverable C ============================
    # Per-query counterfactual: set one feature, recompute its registry descendants,
    # re-predict, then SHRINK toward the observational PD by feature type (see the
    # SHRINK constants above). The shrink + λ̂ pipeline is what converts the model's
    # observational sensitivity into a defensible interventional estimate.
    q = data.load_intervention_queries()
    Xte = _engineer(te, art)
    p_te, _, _, m_te, s_te = model.predict_calibrated(Xte, return_mean_std=True)
    obs_pd = q["applicant_id"].map(dict(zip(te["applicant_id"], p_te))).to_numpy()
    base_std = q["applicant_id"].map(dict(zip(te["applicant_id"], s_te))).to_numpy()
    supp = {f: (pd.to_numeric(te[f], errors="coerce").astype(float).quantile(0.01),
                pd.to_numeric(te[f], errors="coerce").astype(float).quantile(0.99))
            for f in q["feature_name"].unique() if f in te.columns}
    cf = q.merge(te, on="applicant_id", how="left")
    for feat in q["feature_name"].unique():
        m = (cf["feature_name"] == feat).to_numpy()
        vals = pd.to_numeric(cf.loc[m, "intervention_value"], errors="coerce").to_numpy(float)
        if pd.api.types.is_bool_dtype(te[feat]):
            cf[feat] = cf[feat].astype(object); cf.loc[m, feat] = (vals > 0.5)
        elif pd.api.types.is_integer_dtype(te[feat]):
            cf[feat] = cf[feat].astype(float); cf.loc[m, feat] = np.round(vals)
        else:
            cf.loc[m, feat] = vals
    naive_cf, _, _ = model.predict_calibrated(_engineer(cf[te.columns], art))
    # Heuristic shrink: self-report=0 (claim doesn't cause default); immutable=0.5
    # (queried but not a policy lever); manipulable=1.0 (full registry propagation).
    shrink = q["feature_name"].map(_shrink_factor).to_numpy()
    pcf_heur = obs_pd + shrink * (naive_cf - obs_pd)
    # DAG-derived λ̂ shrink for confounded proxies (bureau / bank-feed): keep only
    # the causal fraction β_adj/β_naive estimated by sibling-adjusted logistic on
    # the proxy block. Adjustment sets come from the explicit DAG via the backdoor
    # criterion (causal_graph.get_adjustment_set). Average with the heuristic so
    # neither dominates; fall back to heuristic where λ̂ ∉ (0,1).
    lpath = PATHS.reports / "lambda_hat_dag.csv"
    if not lpath.exists():
        lpath = PATHS.reports / "lambda_hat.csv"
    if lpath.exists():
        lt = pd.read_csv(lpath); lmap = dict(zip(lt.loc[lt.status == "use", "proxy"],
                                                 lt.loc[lt.status == "use", "lambda_hat"]))
        shrink_lam = q["feature_name"].map(lambda f: lmap.get(f, _shrink_factor(f))).to_numpy()
        pcf_lam = obs_pd + shrink_lam * (naive_cf - obs_pd)
        pcf = 0.5 * (pcf_heur + pcf_lam)
        print(f"[C-λ] applied lambda_hat to {len(lmap)} confounded proxies; "
              f"mean |Δ vs heuristic|={np.abs(pcf - pcf_heur).mean():.4f}")
    else:
        pcf = pcf_heur
    # Interval widening: out-of-support do() values get an extra ×1.8 (extrapolation
    # honesty); immutable/proxy effects get ×1.3 (causal uncertainty beyond the
    # observational σ). Self-reports keep the base width since their effect is a
    # confident zero, not an uncertain perturbation.
    oos = np.array([not (supp.get(f, (-np.inf, np.inf))[0] <= v <= supp.get(f, (-np.inf, np.inf))[1])
                    for f, v in zip(q["feature_name"], pd.to_numeric(q["intervention_value"], errors="coerce"))])
    half = model.alpha * model.z90 * base_std * np.where(oos, 1.8, 1.0)
    half = half * np.where((shrink > 0) & (shrink < 1.0), 1.3, 1.0)
    lcf = np.clip(pcf - half, 0, 1)
    hcf = np.clip(pcf + half, 0, 1)
    pd.DataFrame({"query_id": q["query_id"], "predicted_pd_cf": np.clip(pcf, 0, 1),
                  "pd_cf_lower_90": lcf, "pd_cf_upper_90": hcf}).to_csv(
        OUT / "submission_C_counterfactuals.csv", index=False)
    print(f"[C] 900 rows | mean CF PD={pcf.mean():.3f} | OOS queries widened={int(oos.sum())} | "
          f"self-report (0-effect)={int((shrink==0).sum())} immutable(0.5)={int((shrink==0.5).sum())}")

    F.save_artifacts(art)
    return OUT


def validate(out):
    r = subprocess.run([sys.executable, str(PATHS.root / "validate_submission.py"), str(out)],
                       capture_output=True, text=True)
    print(r.stdout[-700:]); print(r.stderr[-400:] if r.stderr else "", file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(validate(build()))
