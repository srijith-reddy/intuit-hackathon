"""Render the explicit causal DAG (src/causal_graph.py) to reports/figures/causal_dag.png
for Deliverable D §3. Latent H confounds the proxy block; the 3 defended mechanistic
proxy->proxy edges are highlighted; every proxy -> default; requested_amount runs a
mechanical path; stated_* attach to H only (no edge to default).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from src.causal_graph import PROXY_BLOCK, MECHANISTIC_EDGES

ABBR = {
    "aggregate_credit_utilization": "util", "recent_inquiries_count_6mo": "inq6mo",
    "existing_debt_obligations": "debt", "owner_personal_credit_band": "cr.band",
    "observed_monthly_revenue_avg_3mo": "rev_avg", "observed_revenue_trend_3mo": "rev_trend",
    "observed_revenue_volatility": "rev_vol", "observed_cash_balance_p10": "cash",
    "observed_overdraft_count_3mo": "overdraft", "payroll_regularity_score": "payroll",
    "invoice_payment_delinquency_rate": "delinq", "multi_lender_inquiry_count_30d": "inq30d",
}
# order proxies so mechanistic-edge endpoints sit adjacent (short, readable arrows)
ORDER = ["util", "debt", "cr.band", "inq6mo", "inq30d", "rev_trend", "rev_vol",
         "rev_avg", "payroll", "cash", "delinq", "overdraft"]
MECH = {(ABBR[a], ABBR[b]) for a, b in MECHANISTIC_EDGES}

fig, ax = plt.subplots(figsize=(12, 4.2))
n = len(ORDER)
xs = {name: 0.04 + 0.92 * i / (n - 1) for i, name in enumerate(ORDER)}
y_prox, y_H, y_def = 0.50, 0.92, 0.06
H_x, def_x = 0.5, 0.5

def node(x, y, label, fc, ec="#333", fs=9, style="round,pad=0.25"):
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=5,
            bbox=dict(boxstyle=style, fc=fc, ec=ec, lw=1.2))

def arrow(x1, y1, x2, y2, color, lw, alpha, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=11, lw=lw, color=color, alpha=alpha, zorder=3))

# H -> proxies (confounding, faint), proxies -> default (faint)
for name, x in xs.items():
    arrow(H_x, y_H - 0.05, x, y_prox + 0.05, "#9aa", 0.7, 0.45)
    arrow(x, y_prox - 0.05, def_x, y_def + 0.05, "#bbb", 0.6, 0.35)
# H -> default
arrow(H_x, y_H - 0.05, def_x + 0.06, y_def + 0.05, "#9aa", 0.9, 0.55, rad=-0.25)
# mechanistic proxy -> proxy (highlighted)
for a, b in MECH:
    arrow(xs[a], y_prox, xs[b], y_prox, "#c0392b", 2.2, 0.95, rad=-0.45)

# nodes
node(H_x, y_H, "H  (latent business health)", "#fdf2d0", ec="#b8860b", fs=10, style="round,pad=0.4")
node(def_x, y_def, "default", "#d6eaf8", ec="#2471a3", fs=10, style="round,pad=0.4")
for name, x in xs.items():
    node(x, y_prox, name, "#eef6ee", fs=8)

# legend / annotations
ax.plot([], [], color="#c0392b", lw=2.2, label="defended mechanistic edge (descendant)")
ax.plot([], [], color="#9aa", lw=0.9, label="H confounds every proxy (proxy adjustment target)")
ax.plot([], [], color="#bbb", lw=0.8, label="proxy → default (direct effect to isolate)")
ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=1, bbox_to_anchor=(0.0, 1.02))
ax.text(0.5, -0.02, "stated_* attach to H only (no edge to default); requested_amount → derived "
        "affordability → default (mechanical, full path)", ha="center", va="top", fontsize=7.5,
        style="italic", color="#555", transform=ax.transAxes)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.08, 1.02); ax.axis("off")
plt.tight_layout()
out = "reports/figures/causal_dag.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
