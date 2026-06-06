"""Step 1 — temporal stability audit (Home Credit playbook). READ-ONLY.

Evaluates OUR final predictions (submission_A) on labeled validation, our B grid
vs realized val CDR, an independent adversarial-validation drift check, and whether
our timing model distinguishes early vs late defaults. Writes nothing outside audit/.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_predict
from src import data, features as F
from src.config import SEED

D = "dataset/dataset-compressed/"
A = pd.read_csv("submissions/submission_A_decisions.csv")
B = pd.read_csv("submissions/submission_B_trajectory.csv")
va = data.load_validation(); tr = data.load_train()
va = va.assign(cohort_week=data.assign_cohort_week(va))

# our PD on labeled val
v = A.merge(va, on="applicant_id", how="inner")
vl = v[v["default_flag"].notna()].copy()
vl["y"] = vl["default_flag"].astype(int)
print(f"labeled val: {len(vl)}  overall AUC={roc_auc_score(vl.y, vl.predicted_pd):.4f} "
      f"Brier={brier_score_loss(vl.y, vl.predicted_pd):.4f}")

# ---- 1.1 per-cohort metrics --------------------------------------------------
def ece(p, y, b=5):
    df = pd.DataFrame({"p": p, "y": y}); df["bin"] = pd.qcut(df.p, b, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pm=("p","mean"), ym=("y","mean"), n=("y","size"))
    return float((g.n*(g.pm-g.ym).abs()).sum()/g.n.sum())

print("\n=== 1.1 per-cohort-week metrics (our PD on val) ===")
rows = []
for w, g in vl.groupby("cohort_week"):
    try: auc = roc_auc_score(g.y, g.predicted_pd)
    except Exception: auc = np.nan
    rows.append((int(w), len(g), g.y.mean(), auc, brier_score_loss(g.y, g.predicted_pd),
                 g.predicted_pd.mean()-g.y.mean(), ece(g.predicted_pd.values, g.y.values, 4)))
pc = pd.DataFrame(rows, columns=["cohort","n","def_rate","AUC","Brier","cal_gap","ECE"])
print(pc.round(4).to_string(index=False))
print(f"AUC: mean={pc.AUC.mean():.4f} min={pc.AUC.min():.4f} max={pc.AUC.max():.4f} "
      f"spread={pc.AUC.max()-pc.AUC.min():.4f}")
print(f"Brier: mean={pc.Brier.mean():.4f} worst={pc.Brier.max():.4f}")
print(f"|cal_gap|: mean={pc.cal_gap.abs().mean():.4f} worst={pc.cal_gap.abs().max():.4f} "
      f"(cohort {int(pc.loc[pc.cal_gap.abs().idxmax(),'cohort'])})")

# ---- 1.2 adversarial validation (STANDALONE, does not touch our model) -------
print("\n=== 1.2 adversarial drift: features distinguishing train-early/late vs val ===")
tr2 = tr.assign(ts=tr["application_timestamp"])
median_ts = tr2.ts.median()
Xtr_all, art = F.build_features(tr, y=tr["default_flag"], groups=tr["business_id"])
Xva_all, _ = F.build_features(va, fit_artifacts=art)
Xtr_m = F.model_features(Xtr_all); Xva_m = F.model_features(Xva_all)
def adv(Xa, Xb, name):
    X = pd.concat([Xa, Xb], ignore_index=True)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, verbose=-1)
    oof = cross_val_predict(m, X, y, cv=3, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    m.fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print(f"  {name}: adversarial AUC={auc:.3f}  (0.5=no drift, 1.0=total shift)")
    print("    top drifting features:", list(imp.head(8).index))
    return auc, imp
late = tr2.ts > median_ts
adv(Xtr_m[~late.values], Xtr_m[late.values], "train-early vs train-late")
_, imp_val = adv(Xtr_m, Xva_m, "train vs val")
# cross-ref against our model importances (preview proxy)
fi = pd.read_csv("reports/feature_importance_preview.csv", index_col=0).iloc[:,0]
top_model = set(fi.sort_values(ascending=False).head(12).index)
top_drift = set(imp_val.head(12).index)
print("    OUR top-12 model features that ALSO drift (train vs val):", sorted(top_model & top_drift))

# ---- 1.3 B grid monotonicity + vs realized val CDR --------------------------
print("\n=== 1.3 B grid: monotonicity + error vs realized val CDR ===")
mono = all(B.sort_values(["cohort_week","loan_age_weeks"]).groupby("cohort_week")
           .cumulative_default_rate.apply(lambda s: (s.diff().fillna(0) >= -1e-9).all()))
print("  monotone in age for every cohort:", mono)
# realized CDR on OUR approved val set
vad = A.merge(va, on="applicant_id").query("decision==1 and default_flag==default_flag")
vad = vad.assign(y=vad.default_flag.astype(int))
errs = []
for w in range(1, 14):
    cw = vad[vad.cohort_week == w]
    if len(cw) < 10: continue
    for a in range(1, 14):
        real = float(((cw.y == 1) & (cw.days_to_default <= 7*a)).mean())
        pred = float(B[(B.cohort_week == w) & (B.loan_age_weeks == a)].cumulative_default_rate.iloc[0])
        errs.append((w, a, real, pred, pred-real, len(cw)))
e = pd.DataFrame(errs, columns=["w","a","real","pred","err","n"])
print(f"  cells compared={len(e)}  MAE={e.err.abs().mean():.4f}  RMSE={np.sqrt((e.err**2).mean()):.4f}  "
      f"mean signed err(pred-real)={e.err.mean():+.4f}")
worst = e.reindex(e.err.abs().sort_values(ascending=False).index).head(6)
print("  worst cells:\n", worst.round(4).to_string(index=False))

# ---- 1.4 does our timing distinguish early vs late? -------------------------
print("\n=== 1.4 timing model: early vs late default distinction ===")
print("  Our F_i(t)=PD_i*S(t): S(t) is POOLED -> every loan shares ONE shape, scaled by PD.")
print("  => loans differ in LEVEL of default, NOT in TIMING. A day-5 and day-55 default")
print("     have identical per-loan shape; A's E[NPV] uses pooled t_bar=43 for all.")
# would a risk-segment-specific shape differ materially? compare normalized shape by PD tertile
trl = tr[data.labeled_mask(tr)].copy()
# proxy risk by credit band (no model needed)
from src import survival
import numpy as np
for band in sorted(trl.owner_personal_credit_band.dropna().unique()):
    g = trl[trl.owner_personal_credit_band == band]
    dd = g.loc[g.default_flag==1, "days_to_default"]
    if len(dd) < 50: continue
    med = dd.median(); late_share = (dd > 60).mean()
    print(f"   credit_band {int(band)}: n_def={len(dd):5d} median_dtd={med:4.0f} day>60 share={late_share:.3f}")
print("  -> if median_dtd / late-share vary across bands, pooled shape mis-times per-segment.")
