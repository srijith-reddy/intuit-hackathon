"""Surgical C patch: apply the descendant-corrected lambda_hat to shipped C answers.

Exact algebra on shipped artifacts (no model retrain needed):
  shipped  pcf = obs + 0.5*(0.5 + lam_old) * (naive - obs)
  corrected pcf = obs + 0.5*(0.5 + lam_new) * (naive - obs)
  where obs = submission_A predicted_pd (same calibrated PD used as C baseline).
Intervals recentered with shipped half-widths, re-clipped to [0,1].
Run AFTER p4_lambda.py regenerates reports/lambda_hat.csv with DESCENDANTS exclusion.
"""
import pandas as pd, numpy as np
q = pd.read_csv("dataset/intervention_queries.csv")
A = pd.read_csv("submissions/submission_A_decisions.csv")
C = pd.read_csv("submissions/submission_C_counterfactuals.csv")
assert (C.query_id.values == q.query_id.values).all()
base = q["applicant_id"].map(dict(zip(A["applicant_id"], A["predicted_pd"]))).to_numpy()
FIX = {"observed_cash_balance_p10": (0.032, 0.437),
       "observed_monthly_revenue_avg_3mo": (0.588, 0.690)}
pcf, lo, hi = (C[c].to_numpy().copy() for c in
               ["predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90"])
for feat, (l_old, l_new) in FIX.items():
    m = (q["feature_name"] == feat).to_numpy()
    s_old, s_new = 0.5*(0.5+l_old), 0.5*(0.5+l_new)
    new = np.clip(base[m] + (s_new/s_old)*(pcf[m]-base[m]), 0, 1)
    lo[m] = np.clip(new-(pcf[m]-lo[m]), 0, 1); hi[m] = np.clip(new+(hi[m]-pcf[m]), 0, 1)
    print(f"{feat}: {m.sum()} queries | mean|d|={np.abs(new-pcf[m]).mean():.4f} "
          f"max={np.abs(new-pcf[m]).max():.4f}")
    pcf[m] = new
C["predicted_pd_cf"], C["pd_cf_lower_90"], C["pd_cf_upper_90"] = pcf, lo, hi
assert (C.pd_cf_lower_90 <= C.predicted_pd_cf).all() and (C.predicted_pd_cf <= C.pd_cf_upper_90).all()
C.to_csv("submissions/submission_C_counterfactuals.csv", index=False)
print("patched submissions/submission_C_counterfactuals.csv")
