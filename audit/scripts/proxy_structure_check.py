"""Proxy-block dependence diagnostic.

Tests the 'siblings' assumption behind the Deliverable C lambda-hat shrinkage:
that bureau + bank-feed + behavioral proxies are co-symptoms of one latent
business-health factor rather than causes/mediators of each other.

Outputs: reports/proxy_structure.md, reports/figures/proxy_structure.png
Touches no shipped code. Read-only on dataset and lambda_hat.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLassoCV
from sklearn.decomposition import FactorAnalysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(42)

PROXIES = [
    # bureau
    "aggregate_credit_utilization", "recent_inquiries_count_6mo",
    "existing_debt_obligations", "owner_personal_credit_band",
    "days_since_last_external_decline",
    # bank feed
    "observed_monthly_revenue_avg_3mo", "observed_revenue_trend_3mo",
    "observed_revenue_volatility", "observed_cash_balance_p10",
    "observed_overdraft_count_3mo", "payroll_regularity_score",
    # behavioral / context
    "invoice_payment_delinquency_rate", "multi_lender_inquiry_count_30d",
]
SHORT = {
    "aggregate_credit_utilization": "credit_util",
    "recent_inquiries_count_6mo": "inq_6mo",
    "existing_debt_obligations": "debt_oblig",
    "owner_personal_credit_band": "credit_band",
    "days_since_last_external_decline": "decline_recency",
    "observed_monthly_revenue_avg_3mo": "obs_revenue",
    "observed_revenue_trend_3mo": "rev_trend",
    "observed_revenue_volatility": "rev_volatility",
    "observed_cash_balance_p10": "cash_p10",
    "observed_overdraft_count_3mo": "overdrafts",
    "payroll_regularity_score": "payroll_reg",
    "invoice_payment_delinquency_rate": "invoice_delinq",
    "multi_lender_inquiry_count_30d": "inq_30d",
}

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # ordinal credit band -> numeric (dictionary says ordering is meaningful)
    if x["owner_personal_credit_band"].dtype == object:
        cats = sorted(x["owner_personal_credit_band"].dropna().unique())
        x["owner_personal_credit_band"] = x["owner_personal_credit_band"].map(
            {c: i for i, c in enumerate(cats)})
    x["owner_personal_credit_band"] = pd.to_numeric(
        x["owner_personal_credit_band"], errors="coerce")
    # null decline recency = never declined -> exp-decay recency transform, null -> 0
    x["days_since_last_external_decline"] = np.exp(
        -pd.to_numeric(x["days_since_last_external_decline"], errors="coerce") / 180.0
    ).fillna(0.0)
    # light tail-taming on skewed levels
    for c in ["observed_monthly_revenue_avg_3mo", "existing_debt_obligations"]:
        x[c] = np.sign(x[c]) * np.log1p(np.abs(pd.to_numeric(x[c], errors="coerce")))
    return x[PROXIES]

def fisher_z_pvals(pcorr: np.ndarray, n: int, k: int) -> np.ndarray:
    """Two-sided p-values for partial correlations via Fisher z.
    k = number of conditioned variables (p - 2)."""
    z = np.arctanh(np.clip(pcorr, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - k - 3)
    return 2 * stats.norm.sf(np.abs(z) / se)

def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = pvals[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        keep[order[: kmax + 1]] = True
    return keep

def main():
    train = pd.read_csv(ROOT / "dataset" / "train.csv")
    feed = train[train["has_linked_bank_feed"] == True]  # noqa: E712
    X = prepare(feed).dropna()
    n, p = X.shape
    Xs = (X - X.mean()) / X.std(ddof=0)

    # ---- sparse partial-correlation graph ----
    gl = GraphicalLassoCV(cv=5).fit(Xs.values)
    Theta = gl.precision_
    d = np.sqrt(np.diag(Theta))
    pcorr = -Theta / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)

    iu = np.triu_indices(p, 1)
    pc_flat = pcorr[iu]
    nz = np.abs(pc_flat) > 1e-4  # glasso support
    pvals = fisher_z_pvals(pc_flat, n=n, k=p - 2)
    keep = np.zeros_like(nz)
    keep[nz] = bh_fdr(pvals[nz], q=0.05)

    edges = []
    for idx in np.where(keep)[0]:
        i, j = iu[0][idx], iu[1][idx]
        edges.append((SHORT[PROXIES[i]], SHORT[PROXIES[j]],
                      float(pcorr[i, j]), float(pvals[idx])))
    edges.sort(key=lambda e: -abs(e[2]))

    # ---- factor / eigen check ----
    corr = np.corrcoef(Xs.values, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    shares = eigvals / eigvals.sum()
    fa = FactorAnalysis(n_components=3, random_state=42).fit(Xs.values)
    load1 = fa.components_[0]
    if np.median(load1) < 0:  # sign convention: healthy = positive
        load1 = -load1
    loadings = sorted(zip([SHORT[c] for c in PROXIES], load1),
                      key=lambda t: -abs(t[1]))

    # ---- degree / hubness ----
    deg = np.zeros(p, int)
    for idx in np.where(keep)[0]:
        deg[iu[0][idx]] += 1
        deg[iu[1][idx]] += 1
    degrees = sorted(zip([SHORT[c] for c in PROXIES], deg), key=lambda t: -t[1])

    # ---- intervenable-pair mediator scan ----
    intervenables = {"credit_util", "inq_6mo", "debt_oblig", "credit_band",
                     "obs_revenue", "rev_trend", "rev_volatility", "cash_p10",
                     "overdrafts", "payroll_reg", "invoice_delinq", "inq_30d"}
    strong_iv_pairs = [e for e in edges
                       if e[0] in intervenables and e[1] in intervenables
                       and abs(e[2]) >= 0.20]

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, p, endpoint=False)
    pos = {SHORT[PROXIES[i]]: (np.cos(theta[i]), np.sin(theta[i])) for i in range(p)}
    for a, b, w, _ in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], lw=abs(w) * 12,
                color="tab:red" if w < 0 else "tab:blue", alpha=0.6, zorder=1)
    for name, (xx, yy) in pos.items():
        ax.scatter([xx], [yy], s=350, c="white", edgecolors="black", zorder=2)
        ax.annotate(name, (xx, yy), ha="center", va="center", fontsize=7, zorder=3)
    ax.set_title(f"FDR-surviving partial correlations (q=0.05), n={n:,}")
    ax.axis("off")
    ax2 = axes[1]
    ax2.bar(range(1, p + 1), shares * 100, color="tab:blue")
    ax2.set_xlabel("eigenvalue rank"); ax2.set_ylabel("% variance")
    ax2.set_title(f"Scree: F1={shares[0]*100:.1f}%, F2={shares[1]*100:.1f}%, F3={shares[2]*100:.1f}%")
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "proxy_structure.png", dpi=150)

    # ---- verdict ----
    f1, f2 = shares[0], shares[1]
    dominant = (f1 >= 0.30) and (f1 / max(f2, 1e-9) >= 1.8)
    if dominant and not strong_iv_pairs:
        verdict = "GREEN — single dominant factor, no strong direct intervenable-pair edges; siblings assumption holds; lambda-hat valid for all 'use' families."
    elif dominant or len(strong_iv_pairs) <= 3:
        verdict = ("PARTIAL — factor structure broadly supports a latent-health confounder, but specific "
                   "intervenable pairs show strong direct edges (mediator risk): "
                   + "; ".join(f"{a}—{b} ({w:+.2f})" for a, b, w, _ in strong_iv_pairs)
                   + ". For these features, prefer the heuristic / exclude the partner from conditioning sets.")
    else:
        verdict = "RED — no dominant factor and dense cross-edges; keep uniform heuristic shrinkage everywhere."

    # ---- report ----
    lam = pd.read_csv(ROOT / "reports" / "lambda_hat.csv") if (ROOT / "reports" / "lambda_hat.csv").exists() else None
    lines = ["# Proxy-Block Dependence Diagnostic", "",
             f"Population: train, linked-feed complete cases (n={n:,} of {len(train):,}; feed rate {train['has_linked_bank_feed'].mean():.1%}).",
             f"Features: {p} proxies (bureau + bank-feed + behavioral). GraphicalLassoCV alpha={gl.alpha_:.4f}.", "",
             "## Eigen / factor structure",
             f"- Variance shares: F1 {f1:.1%}, F2 {f2:.1%}, F3 {shares[2]:.1%} (F1/F2 ratio {f1/max(f2,1e-9):.2f})",
             "- Factor-1 loadings (|.| sorted): " + ", ".join(f"{k} {v:+.2f}" for k, v in loadings), "",
             "## FDR-surviving edges (q=0.05), |partial corr| sorted",
             "| edge | partial corr | p |", "|---|---|---|"]
    lines += [f"| {a} — {b} | {w:+.3f} | {pv:.1e} |" for a, b, w, pv in edges]
    lines += ["", "## Node degrees", ", ".join(f"{k}:{v}" for k, v in degrees), "",
              "## Strong direct intervenable-pair edges (|pc| >= 0.20)",
              ("none" if not strong_iv_pairs else
               "; ".join(f"{a}—{b} ({w:+.3f})" for a, b, w, _ in strong_iv_pairs)), "",
              "## VERDICT", verdict, ""]
    if lam is not None:
        lines += ["## Cross-check vs shipped lambda_hat.csv",
                  "Features already on FALLBACK (heuristic) per the sanity gate: "
                  + ", ".join(lam.loc[lam.status == "FALLBACK", "proxy"]) + ".",
                  "Audit below flags any 'use'-status feature contradicted by this diagnostic.", ""]
    (ROOT / "reports" / "proxy_structure.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
