"""End-to-end submission builder: trains the shared PD model and writes A/B/C.

    python -m src.submit            # builds submissions/ and runs the validator

Deliverables (brief weights): A P&L 30%, B trajectory 25%, calibration 20%,
C counterfactual 10%, writeup 15%. A decides via E[NPV] sign; B = canonical weekly
shape x per-cohort approved mean PD; C = registry do() with causal treatment by
feature type. Intervals are calibrated on validation (see src/calibration.py).
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

# C — causal treatment of the intervention target (brief: match TRUE interventional
# effect, not naive re-prediction). Shrink the naive perturbation toward the
# observational PD by feature type.
SELF_REPORT = {"stated_annual_revenue", "stated_time_in_business"}  # do() = a claim, ~0 causal effect
IMMUTABLE_PROXY = {  # forced do() on non-manipulable features -> partial effect, wider interval
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
    """Exact E[NPV_default | default] = Σ_t w_b(t)·NPV_default(amount, t, rec) over the
    band-conditional daily default-day distribution (dist: n×90, rows sum to 1).
    NPV is linear in t*, so this equals NPV_default at the daily-mean default day."""
    amount = np.asarray(amount, float)
    days = np.arange(1, dist.shape[1] + 1, dtype=float)
    rec = rec_frac * amount
    npv_def_mat = ec.npv_if_default(amount[:, None], days[None, :], rec[:, None])  # n×90
    return (dist * npv_def_mat).sum(axis=1)


def _enpv_decision(p, amount, dist, rec_frac, sigma=None, kappa=0.0):
    """Approve iff E[NPV](p+κσ) > 0 with exact timing integration.

    Reported PD stays `p`; only the DECISION uses the uncertainty-shifted p+κσ
    (κ≥0 → approve the marginal high-disagreement loans less; tuned on val P&L).
    Returns (decision, exp_npv_default) — the latter is p-independent, for reporting.
    """
    rev = ec.revenue_if_full(amount)
    exp_def = _exp_npv_default(amount, dist, rec_frac)
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
    t_bar = survival.mean_default_day(tr)
    rec_frac = survival.mean_recovery_frac(tr)
    S_week = survival.weekly_shape(tr)
    S_band = survival.weekly_shape_by_band(tr)         # E3: band-conditional shape
    tbar_band = survival.mean_default_day_by_band(tr)  # E3: band-conditional E[t*]
    daily_band = survival.daily_dist_by_band(tr)       # E5: band daily default-day dist for exact E[NPV]

    # ---- E1: recalibrate the PD level on in-window validation -------------
    # Train OOF base rate is 17.5% but the scored window runs ~20.6% -> refit the
    # isotonic calibrator on labeled val (truly held out from the full ensemble) so
    # the level matches the deployment window. Fixes the under-prediction the audit
    # found leaking into both S_cal and S_P&L.
    Xva = _engineer(va, art)
    vlab = data.labeled_mask(va).to_numpy()
    yva = va.loc[vlab, "default_flag"].astype(int).to_numpy()
    mva, _ = model.predict_raw(Xva)
    before = float(model.iso.transform(np.clip(mva[vlab], 0, 1)).mean())
    model.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.clip(mva[vlab], 0, 1), yva)
    after = float(model.iso.transform(np.clip(mva[vlab], 0, 1)).mean())
    print(f"[E1] recalibrated on val: mean PD {before:.3f}->{after:.3f} (val rate {yva.mean():.3f})")

    # ---- E2: calibrate A interval width via CROSS-FIT on val --------------
    # Fit width on out-of-fold calibrated PD so coverage reflects real calibration
    # error (in-sample would collapse the width to ~0 after E1's val refit).
    from sklearn.model_selection import KFold
    _, _, _, mva, sva = model.predict_calibrated(Xva, return_mean_std=True)
    raw_lab, std_lab = mva[vlab], sva[vlab]
    oof_cal = np.zeros(len(yva))
    for tri, tei in KFold(5, shuffle=True, random_state=SEED).split(raw_lab):
        isof = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            np.clip(raw_lab[tri], 0, 1), yva[tri])
        oof_cal[tei] = isof.transform(np.clip(raw_lab[tei], 0, 1))
    alpha, ar = cal.fit_pd_interval_scale(oof_cal, std_lab, yva, n_bins=10)
    model.alpha = alpha
    print(f"[A-cal] cross-fit width alpha={alpha} -> val decile coverage={ar['coverage']} "
          f"mean width={ar['mean_width']}")

    # ---- E5: tune the κ-shifted decision rule on labeled-val realized P&L --
    # approve iff E[NPV](p+κσ)>0. The audit shows a +PD shift RAISES realized P&L
    # (we over-approve marginal loans); κ shifts by the loan's own fold-disagreement
    # σ, so the most uncertain near-break-even loans are declined first. κ chosen by
    # realized P&L over a grid, with a 5-fold cross-fit OOF check (no test rows).
    pva, _, _ = model.predict_calibrated(Xva)
    dist_va = np.array(survival.band_lookup(va, daily_band), dtype=float)
    amt_va = va["requested_amount"].to_numpy(float)
    rev_va = ec.revenue_if_full(amt_va)
    exp_def_va = _exp_npv_default(amt_va, dist_va, rec_frac)
    tstar_va = pd.to_numeric(va["days_to_default"], errors="coerce").fillna(0).to_numpy(float)
    recd_va = va["final_recovered_amount"].fillna(0).to_numpy(float)
    realized_va = np.where(va["default_flag"].to_numpy() == 0,
                           rev_va, ec.npv_if_default(amt_va, tstar_va, recd_va))
    kappa, kinfo = cal.fit_kappa_decision_shift(
        pva[vlab], sva[vlab], rev_va[vlab], exp_def_va[vlab], realized_va[vlab], seed=SEED)
    print(f"[A-κ] κ*={kappa} | val P&L κ0=${kinfo['pnl_kappa0']:,} -> κ*=${kinfo['pnl_kappa_star']:,} "
          f"| cross-fit OOF=${kinfo['oof_pnl_adaptive']:,} | folds={kinfo['fold_picks']}")
    print(f"[A-κ] curve {kinfo['curve']}")

    # ============================ Deliverable A ============================
    scored = pd.concat([va, te], ignore_index=True)
    Xsc = _engineer(scored, art)
    p, lo, hi, _, ssc = model.predict_calibrated(Xsc, return_mean_std=True)
    amt = scored["requested_amount"].to_numpy(float)
    dist_sc = np.array(survival.band_lookup(scored, daily_band), dtype=float)  # E5 per-loan daily dist
    decision, npv_def = _enpv_decision(p, amt, dist_sc, rec_frac, sigma=ssc, kappa=kappa)
    pd.DataFrame({"applicant_id": scored["applicant_id"], "decision": decision,
                  "predicted_pd": p, "pd_lower_90": lo, "pd_upper_90": hi}
                 ).to_csv(OUT / "submission_A_decisions.csv", index=False)
    print(f"[A] {len(scored)} rows | approve={decision.mean():.3f} | "
          f"approved book PD={p[decision==1].mean():.3f} | "
          f"break-even PD~{ec.REV_RATE/(ec.REV_RATE - npv_def.mean()/amt.mean()):.3f} | "
          f"mean interval width={np.mean(hi-lo):.4f}")

    # ============================ Deliverable B ============================
    # Calibrate B intervals against TRUE val cohort trajectories. The approved-val
    # cohort uses the SAME κ-shifted rule as A (consistency of the approved book).
    dec_va, _ = _enpv_decision(pva, amt_va, dist_va, rec_frac, sigma=sva, kappa=kappa)
    appr_va_lab = (dec_va == 1) & vlab
    va_c = va.assign(cohort_week=data.assign_cohort_week(va), pd_hat=pva)
    pred_traj, true_traj = {}, {}
    for w in range(1, 14):
        m = appr_va_lab & (va_c["cohort_week"].to_numpy() == w)
        if m.any():  # E3: per-loan band shape, weighted by PD
            sub = va_c.loc[m]
            Smat = np.array(survival.band_lookup(sub, S_band))
            pred_traj[w] = (Smat * sub["pd_hat"].to_numpy()[:, None]).mean(0)
        else:
            pred_traj[w] = S_week * np.nan
    true_traj = cal.true_cohort_trajectory(va, appr_va_lab)
    # E4: per-cohort level signal. Val & test span the SAME 13 weeks, so the val
    # realized approved cohort rate informs the test cohort level the PD model can't
    # see (train predates the cohorts). Shrink val rate toward the model level.
    KB = 75.0
    va_cw = va_c["cohort_week"].to_numpy()
    val_n = {w: int((appr_va_lab & (va_cw == w)).sum()) for w in range(1, 14)}
    val_rate = {w: (true_traj[w][12] if val_n[w] > 0 else np.nan) for w in range(1, 14)}
    # E6: hierarchical SHAPE shrinkage (mirrors E4's level shrinkage). Blend each
    # cohort's empirical val timing increments toward the model band shape, concentration
    # c_shape chosen by split-half cross-fit within val (honest LOCO, no self-prediction).
    cohort_loans = {}
    for w in range(1, 14):
        mm = appr_va_lab & (va_cw == w)
        if mm.any():
            sw = va_c.loc[mm]
            cohort_loans[w] = (pd.to_numeric(sw["days_to_default"], errors="coerce").fillna(0).to_numpy(float),
                               sw["default_flag"].to_numpy(float))
    c_shape, cinfo = cal.fit_shape_shrinkage_c(cohort_loans, pred_traj, val_n, seed=SEED)
    print(f"[B-shape] c*={c_shape} | LOCO half-MAE {cinfo['loco_mae']}")
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

    scored = scored.assign(cohort_week=data.assign_cohort_week(scored), pd_hat=p, decision=decision)
    approved = scored[scored["decision"] == 1]
    gmean = approved["pd_hat"].mean()
    rng = np.random.default_rng(SEED)
    rows, low_d, up_d = [], {}, {}
    for w in range(1, 14):
        sub = approved[approved["cohort_week"] == w]
        if len(sub) == 0:  # fallback: pooled shape x global mean PD
            contrib = (S_week[None, :] * gmean)
        else:  # E3: per-loan CDR = PD_i * S_band(i)(a); cohort CDR = mean over loans
            Smat = np.array(survival.band_lookup(sub, S_band))      # n x 13
            contrib = Smat * sub["pd_hat"].to_numpy()[:, None]      # n x 13
        point_vec = contrib.mean(0)
        boot = np.array([contrib[rng.integers(0, len(contrib), len(contrib))].mean(0)
                         for _ in range(BOOT)])                     # BOOT x 13
        # E6: shrink the cohort SHAPE toward the model band shape (concentration c_shape),
        # using the val empirical timing where it exists. Tail cohorts (sparse, noisy
        # timing) fall back to the stable band shape; data-rich cohorts follow the data.
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
        # E4: shrink the cohort LEVEL toward the val realized rate (same calendar week)
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
    q = data.load_intervention_queries()
    # observational PD per queried applicant
    Xte = _engineer(te, art)
    p_te, _, _, m_te, s_te = model.predict_calibrated(Xte, return_mean_std=True)
    obs_pd = q["applicant_id"].map(dict(zip(te["applicant_id"], p_te))).to_numpy()
    base_std = q["applicant_id"].map(dict(zip(te["applicant_id"], s_te))).to_numpy()
    # support bounds per feature (test 1-99 pctile)
    supp = {f: (pd.to_numeric(te[f], errors="coerce").astype(float).quantile(0.01),
                pd.to_numeric(te[f], errors="coerce").astype(float).quantile(0.99))
            for f in q["feature_name"].unique() if f in te.columns}
    # naive counterfactual via registry recompute
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
    # causal shrink toward observational PD + OOS interval widening
    shrink = q["feature_name"].map(_shrink_factor).to_numpy()
    pcf_heur = obs_pd + shrink * (naive_cf - obs_pd)
    # Part 4: for CONFOUNDED proxies (bureau/bank-feed symptoms of latent health),
    # replace the heuristic full-strength perturbation with an estimated causal
    # fraction lambda_hat = beta_adj/beta_naive (sibling-adjusted logistic; only proxies
    # with lambda_hat in (0,1), else fall back to heuristic). Final = mean(heuristic, lambda).
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
    oos = np.array([not (supp.get(f, (-np.inf, np.inf))[0] <= v <= supp.get(f, (-np.inf, np.inf))[1])
                    for f, v in zip(q["feature_name"], pd.to_numeric(q["intervention_value"], errors="coerce"))])
    half = model.alpha * model.z90 * base_std * np.where(oos, 1.8, 1.0)
    # immutable/proxy effects are causally uncertain -> extra width; self-report
    # (shrink=0) is a confident ~0 effect, so it keeps the base (obs) uncertainty.
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
