"""Explicit causal DAG over the proxy block + outcome, and algorithmic backdoor
adjustment-set derivation for Deliverable C.

The structure is encoded ONCE as a `networkx.DiGraph`; per-treatment adjustment
sets are derived by graph reachability (`networkx.descendants`) rather than
hand-listed. This makes the causal assumptions inspectable, regression-testable,
and easy to amend when new evidence about an edge arrives.

Structure encoded
-----------------
* Latent `H` ("business health", unobserved) → every proxy, every stated_* field,
  and default. `H` is the unobservable confounder; conditioning on the OTHER proxies
  (co-children of `H`) is the proxy-adjustment workaround.
* Mechanistic proxy→proxy edges defended by both the mechanism and the FDR
  partial-correlation graph in reports/proxy_structure.md:
    observed_cash_balance_p10 → invoice_payment_delinquency_rate    (FDR r=-0.78)
    observed_cash_balance_p10 → observed_overdraft_count_3mo        (cash funds draws)
    observed_monthly_revenue_avg_3mo → payroll_regularity_score     (FDR r=+0.50)
  The candidate edge observed_revenue_volatility → observed_cash_balance_p10 is
  NOT asserted: FDR partial-correlation is not significant.
* Every proxy → default (the direct effects C tries to isolate); H → default.
* requested_amount → {derived affordability features} → default (mechanical path).
* stated_*: H → stated_*, but NO stated_* → default (a claim does not cause default).

Adjustment rule
---------------
For a proxy treatment `X`, condition on its co-proxies of `H`, EXCLUDING graph
descendants of `X` — conditioning on a descendant/mediator blocks part of the
true `X → default` path and over-shrinks the causal fraction `λ̂`.
"""
from __future__ import annotations

import networkx as nx

# Proxy block: the bureau / bank-feed / behavioral signals that act as
# observable proxies for the latent business-health confounder H.
BUREAU = ["aggregate_credit_utilization", "recent_inquiries_count_6mo",
          "existing_debt_obligations", "owner_personal_credit_band"]
FEED = ["observed_monthly_revenue_avg_3mo", "observed_revenue_trend_3mo",
        "observed_revenue_volatility", "observed_cash_balance_p10",
        "observed_overdraft_count_3mo", "payroll_regularity_score"]
BEHAV = ["invoice_payment_delinquency_rate", "multi_lender_inquiry_count_30d"]
FAMILIES = {"bureau": BUREAU, "bank_feed": FEED, "behavioral": BEHAV}
PROXY_BLOCK = BUREAU + FEED + BEHAV

STATED = ["stated_annual_revenue", "stated_time_in_business"]
# Derived affordability features on the requested_amount mechanical path (writeup §3).
DERIVED = ["daily_payment", "buffer_to_payment", "debt_to_revenue",
           "requested_amount_to_observed_revenue"]

# Mechanistic proxy->proxy edges (defended + FDR-supported). EXACTLY the relationships
# the hand-coded DESCENDANTS dict encoded; this list is the single source of truth.
MECHANISTIC_EDGES = [
    ("observed_cash_balance_p10", "invoice_payment_delinquency_rate"),
    ("observed_cash_balance_p10", "observed_overdraft_count_3mo"),
    ("observed_monthly_revenue_avg_3mo", "payroll_regularity_score"),
]


def build_dag() -> nx.DiGraph:
    """Construct the explicit DAG described in the module docstring."""
    G = nx.DiGraph()
    G.add_nodes_from(PROXY_BLOCK + STATED + DERIVED + ["H", "default", "requested_amount"])
    # latent health confounds every proxy, every self-report, and default
    for p in PROXY_BLOCK + STATED:
        G.add_edge("H", p)
    G.add_edge("H", "default")
    # mechanistic proxy->proxy edges
    G.add_edges_from(MECHANISTIC_EDGES)
    # every proxy directly affects default (the effect we isolate)
    for p in PROXY_BLOCK:
        G.add_edge(p, "default")
    # requested_amount -> derived affordability -> default (mechanical, full path)
    for f in DERIVED:
        G.add_edge("requested_amount", f)
        G.add_edge(f, "default")
    # stated_* have NO edge to default (a claim does not cause default)
    assert nx.is_directed_acyclic_graph(G), "causal graph must be acyclic"
    return G


DAG = build_dag()


def get_adjustment_set(treatment: str, graph: nx.DiGraph | None = None) -> list[str]:
    """Backdoor adjustment set for a proxy `treatment` under latent-H confounding.

    Returns the co-proxies of H to condition on, EXCLUDING descendants of the treatment
    (graph reachability). The estimating regression conditions on ``[treatment] + set``.
    """
    G = graph or DAG
    if treatment not in PROXY_BLOCK:
        raise ValueError(f"{treatment} is not in the proxy block")
    desc = nx.descendants(G, treatment)
    return [p for p in PROXY_BLOCK if p != treatment and p not in desc]


def regression_columns(treatment: str, graph: nx.DiGraph | None = None) -> list[str]:
    """Columns for the sibling-adjusted logistic: treatment + its adjustment set."""
    return [treatment] + get_adjustment_set(treatment, graph)


# Expected exclusions per treatment, derived directly from the MECHANISTIC_EDGES
# above (every named edge X → Y excludes Y from X's adjustment set). Asserted by
# the self-test so any future edge addition without updating this dict fails loudly.
_EXPECTED_DESCENDANTS = {
    "observed_cash_balance_p10": ["invoice_payment_delinquency_rate",
                                  "observed_overdraft_count_3mo"],
    "observed_monthly_revenue_avg_3mo": ["payroll_regularity_score"],
}


def _self_test() -> None:
    """Regression-test: graph reachability reproduces the expected exclusions."""
    for X in PROXY_BLOCK:
        adj = set(get_adjustment_set(X))
        excluded = set(PROXY_BLOCK) - {X} - adj
        expected = set(_EXPECTED_DESCENDANTS.get(X, []))
        assert excluded == expected, (
            f"{X}: DAG excludes {excluded}, expected {expected}")
    cash = get_adjustment_set("observed_cash_balance_p10")
    assert "invoice_payment_delinquency_rate" not in cash
    assert "observed_overdraft_count_3mo" not in cash
    rev = get_adjustment_set("observed_monthly_revenue_avg_3mo")
    assert "payroll_regularity_score" not in rev
    assert not DAG.has_edge("stated_annual_revenue", "default")
    print("causal_graph self-test PASS")


if __name__ == "__main__":
    G = DAG
    print(f"DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"acyclic={nx.is_directed_acyclic_graph(G)}")
    print("mechanistic proxy->proxy edges:", MECHANISTIC_EDGES)
    for X in PROXY_BLOCK:
        d = sorted(nx.descendants(G, X) & set(PROXY_BLOCK))
        if d:
            print(f"  descendants(in-block) of {X}: {d}  -> excluded from its adjustment set")
    _self_test()
