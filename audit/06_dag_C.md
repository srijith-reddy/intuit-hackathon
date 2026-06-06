# 06 — Deliverable C: DAG-derived adjustment sets (C1–C3)

DAG: 21 nodes, 38 edges, acyclic. Latent H → every proxy/stated/default; mechanistic proxy→proxy edges below; every proxy → default; requested_amount → derived → default; stated_* have no edge to default.

## C2 — PC / GES direction check (diagnostic, does NOT rewrite the DAG)

PC (fisherz, α=0.05) and GES on the standardized labeled-train proxy block. The latent confounder H violates causal sufficiency, so PC/GES assumptions are broken: agreement corroborates our mechanistic edges, disagreement is NOT proof we are wrong (an unobserved common cause can flip or hide a discovered direction).

| mechanistic edge | PC | GES |
|---|---|---|
| `observed_cash_balance_p10` → `invoice_payment_delinquency_rate` | agrees (X→Y) | REVERSED (Y→X) — flagged |
| `observed_cash_balance_p10` → `observed_overdraft_count_3mo` | REVERSED (Y→X) — flagged | agrees (X→Y) |
| `observed_monthly_revenue_avg_3mo` → `payroll_regularity_score` | REVERSED (Y→X) — flagged | agrees (X→Y) |

## C3 — λ̂ recomputed via DAG adjustment sets vs hand-coded

Max |Δλ̂| (DAG vs hand-coded) = **0.0000** — the DAG reproduces the hand-coded exclusions (regression test in `src/causal_graph.py`), so λ̂ is unchanged by construction.

| proxy | λ̂ hand | λ̂ DAG | status |
|---|---|---|---|
| `aggregate_credit_utilization` | 0.521 | 0.521 | use |
| `recent_inquiries_count_6mo` | -0.02 | -0.02 | FALLBACK |
| `existing_debt_obligations` | -0.06 | -0.06 | FALLBACK |
| `owner_personal_credit_band` | -0.223 | -0.223 | FALLBACK |
| `observed_monthly_revenue_avg_3mo` | 0.69 | 0.69 | use |
| `observed_revenue_trend_3mo` | 0.666 | 0.666 | use |
| `observed_revenue_volatility` | 0.728 | 0.728 | use |
| `observed_cash_balance_p10` | 0.437 | 0.437 | use |
| `observed_overdraft_count_3mo` | 0.418 | 0.418 | use |
| `payroll_regularity_score` | 0.134 | 0.134 | use |
| `invoice_payment_delinquency_rate` | 0.638 | 0.638 | use |
| `multi_lender_inquiry_count_30d` | -0.177 | -0.177 | FALLBACK |

λ̂_cash = 0.437 (≥0.10 band check PASS); 8/12 proxies in (0,1).
