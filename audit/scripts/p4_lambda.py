"""Part 4: estimate causal fraction lambda-hat for confounded-proxy C channels.
READ-ONLY measurement. Eigenvalue check + sibling-adjusted logistic contrast.
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src import data

tr = data.load_train(); d = tr[data.labeled_mask(tr)].copy()
y = d["default_flag"].astype(int).to_numpy()

# Confounded-proxy block = bureau + bank-feed + behavioral symptoms of latent health.
# (requested_amount excluded = real mechanical cause; stated_* excluded = self-reports.)
BUREAU = ["aggregate_credit_utilization","recent_inquiries_count_6mo","existing_debt_obligations",
          "owner_personal_credit_band"]
FEED   = ["observed_monthly_revenue_avg_3mo","observed_revenue_trend_3mo","observed_revenue_volatility",
          "observed_cash_balance_p10","observed_overdraft_count_3mo","payroll_regularity_score"]
BEHAV  = ["invoice_payment_delinquency_rate","multi_lender_inquiry_count_30d"]
FAM = {"bureau":BUREAU, "bank_feed":FEED, "behavioral":BEHAV}
block = BUREAU+FEED+BEHAV

Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(d[block]))
Z = pd.DataFrame(Z, columns=block)

print("=== eigenvalue check (corr of proxy block) ===")
ev = np.linalg.eigvalsh(np.corrcoef(Z.values, rowvar=False))[::-1]
print("  eigenvalues:", np.round(ev,2))
print(f"  top eigenvalue {ev[0]:.2f} ({ev[0]/ev.sum():.0%} of variance); ratio ev1/ev2 = {ev[0]/ev[1]:.2f}")
print("  -> one dominant factor" if ev[0]/ev[1] > 2.5 else "  -> several comparable factors (use sibling-proxy ratio)")

def beta(cols, target_col):
    X = sm.add_constant(Z[cols]); m = sm.Logit(y, X).fit(disp=0); return m.params[target_col]

print("\n=== lambda_hat = beta_adj(X | siblings) / beta_naive(X) ===")
# Conditioning-set correction (from audit/scripts/proxy_structure_check.py):
# valid adjustment conditions on CO-PROXIES of latent health, not on DOWNSTREAM
# CONSEQUENCES of the feature itself (conditioning on a descendant blocks the
# real causal path -> over-shrinks lambda). The FDR-controlled partial-correlation
# graph flagged: invoice delinquency & overdrafts are consequences of low cash
# (cash_p10 -- invoice_delinq partial corr -0.78), and payroll regularity is a
# consequence of revenue stability (+0.50). Exclude them from those features' sets.
DESCENDANTS = {
    "observed_cash_balance_p10": ["invoice_payment_delinquency_rate",
                                  "observed_overdraft_count_3mo"],
    "observed_monthly_revenue_avg_3mo": ["payroll_regularity_score"],
}
rows=[]
for fam, cols in FAM.items():
    for X in cols:
        b_naive = beta([X], X)
        cond = [c for c in block if c not in DESCENDANTS.get(X, [])]
        b_adj   = beta(cond, X)           # adjust for co-proxies, minus X's descendants
        lam = b_adj/b_naive if abs(b_naive)>1e-6 else np.nan
        ok = (0 < lam < 1)
        rows.append((fam, X, round(b_naive,3), round(b_adj,3), round(lam,3),
                     "use" if ok else "FALLBACK"))
t = pd.DataFrame(rows, columns=["family","proxy","beta_naive","beta_adj","lambda_hat","status"])
print(t.to_string(index=False))
print("\n  per-family mean lambda_hat (where in (0,1)):")
print(t[t.status=="use"].groupby("family").lambda_hat.mean().round(3).to_string())
print(f"  proxies in valid (0,1): {(t.status=='use').sum()}/{len(t)}; "
      f"fallbacks: {(t.status=='FALLBACK').sum()}")
t.to_csv("reports/lambda_hat.csv", index=False)
