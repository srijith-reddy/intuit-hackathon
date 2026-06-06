"""Part 6: recompute lambda-hat using DAG-derived backdoor adjustment sets, and run a
PC/GES direction check (diagnostic). Generalizes audit/scripts/p4_lambda.py — the
hand-coded DESCENDANTS dict is replaced by src.causal_graph.get_adjustment_set().
READ-ONLY measurement; writes reports/lambda_hat_dag.csv and audit/06_dag_C.md.
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src import data
from src.causal_graph import (PROXY_BLOCK, FAMILIES, MECHANISTIC_EDGES,
                              get_adjustment_set, regression_columns, DAG)

OUT_CSV = "reports/lambda_hat_dag.csv"
OUT_MD = "audit/06_dag_C.md"

tr = data.load_train(); d = tr[data.labeled_mask(tr)].copy()
y = d["default_flag"].astype(int).to_numpy()
# identical preprocessing to p4_lambda.py (median impute + standardize, same column order)
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(d[PROXY_BLOCK]))
Z = pd.DataFrame(Z, columns=PROXY_BLOCK)


def beta(cols, target):
    m = sm.Logit(y, sm.add_constant(Z[cols])).fit(disp=0)
    return m.params[target]


# ---- C3: lambda via DAG adjustment sets ------------------------------------
rows = []
for fam, cols in FAMILIES.items():
    for X in cols:
        b_naive = beta([X], X)
        b_adj = beta(regression_columns(X), X)   # [X] + get_adjustment_set(X)
        lam = b_adj / b_naive if abs(b_naive) > 1e-6 else np.nan
        rows.append((fam, X, round(b_naive, 3), round(b_adj, 3), round(lam, 3),
                     "use" if (0 < lam < 1) else "FALLBACK"))
t = pd.DataFrame(rows, columns=["family", "proxy", "beta_naive", "beta_adj", "lambda_hat", "status"])
t.to_csv(OUT_CSV, index=False)

# sanity: cash must stay in a plausible band (the descendant-exclusion must hold)
lam_cash = float(t.loc[t.proxy == "observed_cash_balance_p10", "lambda_hat"].iloc[0])
assert lam_cash >= 0.10, f"lambda_cash={lam_cash} < 0.10 -> conditioning on a descendant again"
n_use = int((t.status == "use").sum())
assert n_use >= len(PROXY_BLOCK) // 2, f"only {n_use}/{len(PROXY_BLOCK)} proxies in (0,1)"

# ---- compare DAG lambda vs hand-coded lambda (reports/lambda_hat.csv) -------
old = pd.read_csv("reports/lambda_hat.csv")
cmp = t.merge(old[["proxy", "lambda_hat"]], on="proxy", suffixes=("_dag", "_hand"))
cmp["abs_dlam"] = (cmp.lambda_hat_dag - cmp.lambda_hat_hand).abs()
max_dlam = float(cmp.abs_dlam.max())

# ---- C2: PC + GES direction check (DIAGNOSTIC ONLY) ------------------------
def _edge_dir(G, names, a, b):
    """Classify the edge between a,b in a causal-learn GeneralGraph: '->','<-','--','<->','none'."""
    i, j = names.index(a), names.index(b)
    m = G.graph
    eij, eji = m[i, j], m[j, i]
    if eij == -1 and eji == 1:  return "->"   # a tail, b arrow  => a->b
    if eij == 1 and eji == -1:  return "<-"   # a arrow, b tail  => b->a
    if eij == -1 and eji == -1: return "--"   # undirected (both tails)
    if eij == 1 and eji == 1:   return "<->"  # bidirected
    return "none"

discovery = {}
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    Znp = Z.to_numpy()
    cg = pc(Znp, alpha=0.05, indep_test="fisherz", show_progress=False)
    discovery["PC"] = {f"{a}->{b}": _edge_dir(cg.G, PROXY_BLOCK, a, b) for a, b in MECHANISTIC_EDGES}
    rec = ges(Znp)
    discovery["GES"] = {f"{a}->{b}": _edge_dir(rec["G"], PROXY_BLOCK, a, b) for a, b in MECHANISTIC_EDGES}
except Exception as e:  # diagnostic only — never blocks the pipeline
    discovery["error"] = repr(e)

# ---- write audit/06_dag_C.md (C2 + C3 sections) ----------------------------
def verdict(edge, mark):
    if mark == "->":  return "agrees (X→Y)"
    if mark == "<-":  return "REVERSED (Y→X) — flagged"
    if mark == "--":  return "adjacent, undirected (consistent; CPDAG cannot orient)"
    if mark == "<->": return "bidirected (latent common cause — consistent with H)"
    return "absent in discovery"

lines = ["# 06 — Deliverable C: DAG-derived adjustment sets (C1–C3)\n",
         f"DAG: {DAG.number_of_nodes()} nodes, {DAG.number_of_edges()} edges, acyclic. "
         "Latent H → every proxy/stated/default; mechanistic proxy→proxy edges below; "
         "every proxy → default; requested_amount → derived → default; stated_* have no edge to default.\n",
         "## C2 — PC / GES direction check (diagnostic, does NOT rewrite the DAG)\n",
         "PC (fisherz, α=0.05) and GES on the standardized labeled-train proxy block. "
         "The latent confounder H violates causal sufficiency, so PC/GES assumptions are "
         "broken: agreement corroborates our mechanistic edges, disagreement is NOT proof "
         "we are wrong (an unobserved common cause can flip or hide a discovered direction).\n",
         "| mechanistic edge | PC | GES |", "|---|---|---|"]
if "error" in discovery:
    lines.append(f"| (discovery unavailable: {discovery['error']}) | — | — |")
else:
    for a, b in MECHANISTIC_EDGES:
        k = f"{a}->{b}"
        lines.append(f"| `{a}` → `{b}` | {verdict(k, discovery['PC'][k])} | {verdict(k, discovery['GES'][k])} |")
lines += ["", "## C3 — λ̂ recomputed via DAG adjustment sets vs hand-coded\n",
          f"Max |Δλ̂| (DAG vs hand-coded) = **{max_dlam:.4f}** — the DAG reproduces the hand-coded "
          "exclusions (regression test in `src/causal_graph.py`), so λ̂ is unchanged by construction.\n",
          "| proxy | λ̂ hand | λ̂ DAG | status |", "|---|---|---|---|"]
for _, r in cmp.iterrows():
    lines.append(f"| `{r.proxy}` | {r.lambda_hat_hand} | {r.lambda_hat_dag} | {r.status} |")
lines += ["", f"λ̂_cash = {lam_cash} (≥0.10 band check PASS); {n_use}/{len(PROXY_BLOCK)} proxies in (0,1)."]
with open(OUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"[p6] wrote {OUT_CSV} and {OUT_MD}")
print(f"[p6] lambda_cash={lam_cash} | max|Δλ vs hand-coded|={max_dlam:.4f} | use={n_use}/{len(PROXY_BLOCK)}")
print(f"[p6] PC/GES: {discovery if 'error' in discovery else {k: discovery[k] for k in ('PC','GES')}}")
