"""Step 5 — literal compliance of our submitted outputs with the brief's formulas.
READ-ONLY. Checks A decision/NPV, B CDR definition, C do() against the screenshots.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
from src import data, features as F, survival, economics as ec

A = pd.read_csv("submissions/submission_A_decisions.csv")
B = pd.read_csv("submissions/submission_B_trajectory.csv")
C = pd.read_csv("submissions/submission_C_counterfactuals.csv")
tr, va, te = data.load_train(), data.load_validation(), data.load_test()
scored = pd.concat([va, te], ignore_index=True)

print("=== NPV formula (screenshot 'More precisely') penny-exact ===")
R, r, T = 17000.0, 0.35, 60
F_, D = 0.03*R, R*(1+r*T/365)/T
brief_repaid = F_ + R*r*(T/365)
brief_def = lambda t, rec: F_ + D*(t-1) + rec - R
print(f"  repaid: brief={brief_repaid:.4f}  ours={ec.revenue_if_full(R):.4f}  "
      f"match={abs(brief_repaid-ec.revenue_if_full(R))<1e-6}")
ok = all(abs(brief_def(t,0)-float(ec.npv_if_default(R,t,0)))<1e-6 for t in [1,5,43,55,90])
print(f"  default@t* for t in [1,5,43,55,90]: all match={ok}")
print(f"  D=R(1+rT/365)/T: brief={D:.5f} ours={R*ec.DAILY_DRAW_FRAC:.5f}  F=0.03R: {F_:.1f}/{ec.FEE*R:.1f}")

print("\n=== A: d_i == 1[E[NPV_i | approve] > 0] (reconstruct from formula) ===")
tbar_band = survival.mean_default_day_by_band(tr)
rec_frac = survival.mean_recovery_frac(tr)
amt = scored["requested_amount"].to_numpy(float)
p = A.set_index("applicant_id").loc[scored["applicant_id"], "predicted_pd"].to_numpy()
tb = np.array(survival.band_lookup(scored, tbar_band), float)
enpv = (1-p)*ec.revenue_if_full(amt) + p*ec.npv_if_default(amt, tb, rec_frac*amt)
recon = (enpv > 0).astype(int)
sub_dec = A.set_index("applicant_id").loc[scored["applicant_id"], "decision"].to_numpy()
print(f"  decisions matching 1[E[NPV]>0]: {(recon==sub_dec).mean():.4f} "
      f"({int((recon==sub_dec).sum())}/{len(recon)})")
print(f"  naive-threshold check: would 1[p<0.5] differ? flips vs ours={int((sub_dec!=(p<0.5).astype(int)).sum())}")
print(f"  interval order l<=p<=u: {bool(((A.pd_lower_90<=A.predicted_pd+1e-9)&(A.predicted_pd<=A.pd_upper_90+1e-9)).all())}; "
      f"all in [0,1]: {bool(((A[['predicted_pd','pd_lower_90','pd_upper_90']]>=0).all().all()) and (A[['predicted_pd','pd_lower_90','pd_upper_90']]<=1).all().all())}")

print("\n=== B: CDR_{w,a}=|{i in A_w: t_i<=7a}|/|A_w| ===")
print(f"  169 cells, cohort/age grid complete: {len(B)==169 and set(zip(B.cohort_week,B.loan_age_weeks))=={(w,a) for w in range(1,14) for a in range(1,14)}}")
print(f"  all CDR in [0,1]: {bool(((B.cumulative_default_rate>=0)&(B.cumulative_default_rate<=1)).all())}")
mono = all(B.sort_values(['cohort_week','loan_age_weeks']).groupby('cohort_week')
           .cumulative_default_rate.apply(lambda s:(s.diff().fillna(0)>=-1e-9).all()))
print(f"  monotone non-decreasing in age (a) per cohort: {mono}")
# a=13 (day 91 >= 90) must equal full default fraction of A_w  -> sanity vs approved mean PD
appr = scored.assign(cohort_week=data.assign_cohort_week(scored),
                     pd_hat=p, dec=sub_dec).query("dec==1")
cdr13 = B[B.loan_age_weeks==13].set_index("cohort_week").cumulative_default_rate
mean_pd_w = appr.groupby("cohort_week").pd_hat.mean()
cmp = pd.DataFrame({"CDR_a13": cdr13, "approved_mean_PD": mean_pd_w}).round(3)
print("  CDR_{w,13} vs approved mean PD (should be close; differs by E4 per-cohort shrinkage):")
print(cmp.to_string())

print("\n=== C: p_cf = Pr(y|do(f=v), X_{-f} held); not naive ===")
print(f"  all p_cf in [0,1]: {bool(((C.predicted_pd_cf>=0)&(C.predicted_pd_cf<=1)).all())}; "
      f"interval order: {bool(((C.pd_cf_lower_90<=C.predicted_pd_cf+1e-9)&(C.predicted_pd_cf<=C.pd_cf_upper_90+1e-9)).all())}")
# provided requested_amount_to_observed_revenue must NOT be a stale model feature
mf = list(F.model_features(F.build_features(te, fit_artifacts=__import__('pickle').load(open('artifacts/feature_artifacts.pkl','rb')))[0]).columns)
print(f"  provided 'requested_amount_to_observed_revenue' used as a model feature? "
      f"{'requested_amount_to_observed_revenue' in mf}  (False => no stale-descendant bug)")
print(f"  our recomputed ratios in model: {[c for c in mf if 'req_to' in c or c in ('buffer_to_payment','daily_payment','leverage_total')]}")
