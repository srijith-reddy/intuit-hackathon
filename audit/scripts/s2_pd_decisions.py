"""Step 2 — missingness, reject inference, calibration, decision economics (AmEx).
READ-ONLY: evaluates our final predictions/decisions on labeled val. The only model
trained is an independent propensity classifier (selection diagnostic).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from src import data, features as F, economics as ec
from src.config import SEED

A = pd.read_csv("submissions/submission_A_decisions.csv")
va = data.load_validation().assign(cohort_week=lambda d: data.assign_cohort_week(d))
tr = data.load_train()
v = A.merge(va, on="applicant_id", how="inner")
vl = v[v["default_flag"].notna()].copy(); vl["y"] = vl["default_flag"].astype(int)
print(f"labeled val n={len(vl)}")

# ---- 2.1 missingness signal vs our handling ---------------------------------
print("\n=== 2.1 missingness: default rate by pattern (labeled val) ===")
for col, lab in [("has_linked_bank_feed", "feed"), ("days_since_last_external_decline", "ext_decline"),
                 ("days_since_last_inquiry_elsewhere", "inquiry"), ("prior_loans_count", "prior_loans")]:
    if col == "has_linked_bank_feed":
        for val_ in [True, False]:
            m = vl[col] == val_
            if m.any(): print(f"  {lab}={val_}: n={m.sum():4d} default={vl.loc[m,'y'].mean():.3f}")
    elif col == "prior_loans_count":
        for label, m in [("first_time(0)", vl[col] == 0), ("repeat(>0)", vl[col] > 0)]:
            print(f"  {label}: n={m.sum():4d} default={vl.loc[m,'y'].mean():.3f}")
    else:
        for label, m in [(f"{lab}=null", vl[col].isna()), (f"{lab}=present", vl[col].notna())]:
            print(f"  {label}: n={m.sum():4d} default={vl.loc[m,'y'].mean():.3f}")
print("  our handling: NaN passed natively to LGBM + explicit indicators "
      "(no_bank_feed, never_declined_external, no_inquiry_elsewhere, is_first_time_borrower).")

# ---- 2.2 reject inference: calibration vs approval propensity ----------------
print("\n=== 2.2 reject inference: PD calibration by approval propensity ===")
Xtr, art = F.build_features(tr, y=tr["default_flag"], groups=tr["business_id"])
Xva, _ = F.build_features(va, fit_artifacts=art)
drop_sel = ["prior_score", "rd_distance", "above_rd_cutoff", "prior_declined_elsewhere_flag"]
Xtr_p = F.model_features(Xtr).drop(columns=drop_sel, errors="ignore")
Xva_p = F.model_features(Xva).drop(columns=drop_sel, errors="ignore")
yA = (tr["prior_decision"] == 1).astype(int).to_numpy()
prop = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          random_state=SEED, verbose=-1)
oof = cross_val_predict(prop, Xtr_p, yA, cv=3, method="predict_proba")[:, 1]
print(f"  propensity P(prior-approve | X, no selection feats): OOF AUC={roc_auc_score(yA, oof):.3f}")
prop.fit(Xtr_p, yA)
vl_prop = prop.predict_proba(Xva_p.loc[vl.index])[:, 1]
vl = vl.assign(prop=vl_prop)
vl["ptile"] = pd.qcut(vl["prop"], 3, labels=["low(decline-like)", "mid", "high(approve-like)"])
print("  calibration of OUR PD within approval-propensity tertiles (labeled val):")
g = vl.groupby("ptile", observed=True).agg(n=("y", "size"), pred=("predicted_pd", "mean"),
                                          obs=("y", "mean"))
g["gap"] = g.pred - g.obs
print(g.round(4).to_string())
print("  NOTE: true prior-declines have NO labels anywhere -> calibration on the scored")
print("        decline population is unverifiable; 'low' tertile = approved-but-decline-looking proxy.")

# ---- 2.3 calibration: ECE + interval coverage -------------------------------
def ece(p, y, b=10):
    df = pd.DataFrame({"p": p, "y": y}); df["bin"] = pd.qcut(df.p, b, duplicates="drop")
    gg = df.groupby("bin", observed=True).agg(pm=("p", "mean"), ym=("y", "mean"), n=("y", "size"))
    return float((gg.n * (gg.pm - gg.ym).abs()).sum() / gg.n.sum())
print("\n=== 2.3 calibration & interval coverage (labeled val) ===")
print(f"  pooled ECE(10-bin)={ece(vl.predicted_pd.values, vl.y.values):.4f}")
vl["dec"] = pd.qcut(vl.predicted_pd, 10, duplicates="drop")
dd = vl.groupby("dec", observed=True).agg(n=("y","size"), pred=("predicted_pd","mean"),
        obs=("y","mean"), lo=("pd_lower_90","mean"), hi=("pd_upper_90","mean"))
dd["covered"] = (dd.lo <= dd.obs) & (dd.obs <= dd.hi)
dd["width"] = dd.hi - dd.lo
print("  per-decile reliability + interval coverage of empirical rate:")
print(dd.round(4).to_string())
print(f"  decile-coverage={dd.covered.mean():.3f}  mean width={dd.width.mean():.4f}")
# per-cohort coverage
cc = vl.groupby("cohort_week").agg(obs=("y", "mean"), lo=("pd_lower_90", "mean"),
                                   hi=("pd_upper_90", "mean"))
cov = ((cc.lo <= cc.obs) & (cc.obs <= cc.hi))
print(f"  per-cohort coverage (mean interval contains cohort default rate): "
      f"{cov.mean():.3f} ({int(cov.sum())}/13)")

# ---- 2.4 decision rule + realized P&L + stress ------------------------------
print("\n=== 2.4 decision economics (realized NPV on labeled val) ===")
amt = vl.requested_amount.to_numpy(float); y = vl.y.to_numpy()
tstar = vl.days_to_default.fillna(0).to_numpy(float)
rec = vl.final_recovered_amount.fillna(0).to_numpy(float)
realized = np.where(y == 0, ec.revenue_if_full(amt), ec.npv_if_default(amt, tstar, rec))
def pnl(mask): return float(realized[mask].sum())
ours = vl.decision.to_numpy().astype(bool)
oracle = realized > 0
print(f"  total labeled-val loans={len(vl)}  (all are prior-approved)")
print(f"  OUR policy:    approve {ours.mean():.3f}  realized P&L=${pnl(ours):,.0f}")
print(f"  approve-all:   approve 1.000  realized P&L=${pnl(np.ones(len(vl),bool)):,.0f}")
print(f"  prior-underwr: approve 1.000 (all labeled were prior-approved) =${pnl(np.ones(len(vl),bool)):,.0f}")
print(f"  oracle NPV>0:  approve {oracle.mean():.3f}  realized P&L=${pnl(oracle):,.0f}")
print(f"  -> our P&L capture vs oracle: {pnl(ours)/pnl(oracle):.3f}; vs approve-all: "
      f"{pnl(ours)/pnl(np.ones(len(vl),bool)):.3f}")
# stress: re-decide under PD +/-20% and recovery +/-50%, measure flips + realized P&L
t_bar = 43.1; rec_frac = 0.091
def decide(pd_hat, rfrac):
    nd = ec.npv_if_default(amt, t_bar, rfrac*amt)
    return ((1-pd_hat)*ec.revenue_if_full(amt) + pd_hat*nd) > 0
base = decide(vl.predicted_pd.to_numpy(), rec_frac)
print("\n  stress test (decision flips & realized P&L vs our actual):")
for dpd, lab in [(1.2,"PD+20%"),(0.8,"PD-20%")]:
    d = decide(np.clip(vl.predicted_pd.to_numpy()*dpd,0,1), rec_frac)
    print(f"   {lab}: flips={int((d!=base).sum())}/{len(vl)} ({(d!=base).mean():.3%})  realized P&L=${pnl(d):,.0f}")
for dr, lab in [(1.5,"recov+50%"),(0.5,"recov-50%")]:
    d = decide(vl.predicted_pd.to_numpy(), rec_frac*dr)
    print(f"   {lab}: flips={int((d!=base).sum())}/{len(vl)} ({(d!=base).mean():.3%})  realized P&L=${pnl(d):,.0f}")
