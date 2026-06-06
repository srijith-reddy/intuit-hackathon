"""Step 3 — causal correctness (C) + writeup defensibility (D). READ-ONLY.
Sanity-batteries OUR submitted p_cf against OUR submitted baseline PD (submission_A
on test). No model is trained or reloaded.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
from src import data, features as F

A = pd.read_csv("submissions/submission_A_decisions.csv")
C = pd.read_csv("submissions/submission_C_counterfactuals.csv")
q = pd.read_csv("dataset/intervention_queries.csv")
te = data.load_test()
obs_pd = dict(zip(A.applicant_id, A.predicted_pd))   # baseline PD = our A prediction on test
m = q.merge(C, on="query_id").assign(obs=lambda d: d.applicant_id.map(obs_pd))
m["delta"] = m.predicted_pd_cf - m.obs

from src.submit import SELF_REPORT, IMMUTABLE_PROXY
def grp(f): return "self_report" if f in SELF_REPORT else ("immutable" if f in IMMUTABLE_PROXY else "manipulable")
m["group"] = m.feature_name.map(grp)

print("=== 3.1 C sanity battery ===")
print(f"  all p_cf in [0,1]: {bool(((C.predicted_pd_cf>=0)&(C.predicted_pd_cf<=1)).all())}")
print(f"  interval order l<=p<=u holds: {bool(((C.pd_cf_lower_90<=C.predicted_pd_cf+1e-9)&(C.predicted_pd_cf<=C.pd_cf_upper_90+1e-9)).all())}")
print(f"  rows={len(C)}  mean p_cf={C.predicted_pd_cf.mean():.3f}")
print("\n  per-group |delta from baseline| (delta = p_cf - our baseline PD):")
print(m.groupby("group").agg(n=("delta","size"), mean_delta=("delta","mean"),
      mean_abs=("delta",lambda s:s.abs().mean()), frac_zero=("delta",lambda s:(s.abs()<1e-6).mean())).round(4).to_string())
print("  -> self_report should be ~0 (we set p_cf=baseline); manipulable should move; immutable partial.")

print("\n  do(x=observed) check — queries whose value ~= the applicant's observed value:")
# attach observed feature value per query
ov = []
te_idx = te.set_index("applicant_id")
for _,r in q.iterrows():
    f=r.feature_name
    try: ov.append(float(pd.to_numeric(te_idx.loc[r.applicant_id, f], errors="coerce")))
    except Exception: ov.append(np.nan)
m["obs_val"] = ov; m["val"] = pd.to_numeric(m.intervention_value, errors="coerce")
near = m[(m.obs_val.notna()) & ((m.val-m.obs_val).abs() <= 0.01*m.obs_val.abs().clip(lower=1))]
print(f"   queries with value≈observed: n={len(near)}  mean|p_cf-baseline|={near.delta.abs().mean():.4f} (want ~0)")

print("\n  monotonicity — do(requested_amount): higher amount => weakly higher PD?")
ra = m[m.feature_name=="requested_amount"].dropna(subset=["obs_val","val"])
ra_dv = (ra.val - ra.obs_val).to_numpy(); ra_dp = ra.delta.to_numpy()
corr = np.corrcoef(ra_dv, ra_dp)[0,1] if len(ra)>2 else np.nan
concord = float(np.mean(np.sign(ra_dv)*np.sign(ra_dp) >= 0))  # same sign or a zero
print(f"   n={len(ra)}  corr(Δamount, Δpd)={corr:+.3f}  concordant(sign agrees/zero)={concord:.2f}")

print("\n=== 3.2 channel-awareness: stale descendants? ===")
print("  do() recompute path: submit.py builds intervened RAW frame -> F.build_features(transform)")
print("  -> ALL descendants recompute from raw (registry). Unit test test_intervention_recompute")
print("     asserts only descendants change. requested_amount monotonicity above is the live check.")
# demonstrate (read-only) that descendants of requested_amount are non-empty
print("  registry descendants of requested_amount:", sorted(F.descendants("requested_amount"))[:8], "...")

print("\n=== 3.3 writeup D checks ===")
w = open("submissions/submission_D_writeup.md").read()
secs = re.findall(r"^##\s*\d\.\s*(.+)$", w, re.M)
print("  sections found:", secs)
words = len(re.findall(r"\w+", w))
print(f"  word count={words} (~{words/550:.1f} pages at ~550 wpp; 4-page limit) — PDF NOT yet rendered")
for claim, present in [("SHAP", "shap" in w.lower()), ("isotonic", "isotonic" in w.lower()),
                       ("GroupKFold", "groupkfold" in w.lower()), ("E[NPV]/NPV", "npv" in w.lower())]:
    print(f"   mentions {claim}: {present}")
# does code produce SHAP?
code = "".join(open(f"src/{f}").read() for f in os.listdir("src") if f.endswith(".py"))
print(f"  CODE actually computes SHAP? {'shap' in code.lower()}  <-- writeup claims SHAP; code does not")

print("\n=== 3.4 interpretability / top-feature legitimacy ===")
fi = pd.read_csv("reports/feature_importance_preview.csv", index_col=0).iloc[:,0].sort_values(ascending=False)
print("  top-10 features by gain (preview LGBM):")
print(fi.head(10).round(0).to_string())
print("  SHAP values persisted as artifact?", os.path.exists("reports/shap_values.csv") or os.path.exists("artifacts/shap.pkl"))
