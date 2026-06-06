# Audit 03 — Causal Correctness (C) + Writeup Defensibility (D) (Illinois/WiDS)

Lesson: winners finish every deliverable and defend every choice simply. Evaluated
OUR submitted `p_cf` against OUR baseline PD (`submission_A` on test). Reproduce:
`python audit/scripts/s3_causal_writeup.py`.

## 3.1 How did we compute C, and does it pass sanity? — **PASS (strong)**
C is **channel-aware**, not naive re-prediction: `submit.py` builds the intervened
**raw** frame and calls `build_features` (registry), then applies a **per-feature
causal treatment** (self-report shrink 0, immutable/proxy 0.5, manipulable 1.0) +
out-of-support widening. Sanity battery on the 900 submitted values:

| check | result | verdict |
|---|---|---|
| all `p_cf ∈ [0,1]` | True | PASS |
| interval order `l ≤ p_cf ≤ u` | True | PASS |
| `do(x = observed)` ≈ baseline (71 near-observed queries) | **mean \|Δ\| = 0.0000** | PASS |
| `do(requested_amount)` monotone: ↑amount ⇒ ↑PD | **corr(Δamt,ΔPD)=+0.714, concordant 1.00** | PASS |
| self-report `do()` returns baseline | **frac_zero = 1.000** (104 queries) | PASS |

The +0.714 amount→PD correlation also **proves descendants are recomputed live** (a
stale-descendant bug would give ~0 or wrong-sign response). Registry descendants of
`requested_amount` (buffer_to_payment, daily_payment, dscr, leverage, ratios…) are
non-empty and recompute on intervention; `tests/test_features.py::
test_intervention_recompute` asserts *only* descendants change. **No stale-descendant
bug.**

## 3.2 Proxy/confounded channels — did we shrink? — **PASS**
Per-group |Δ from baseline| over the 900 queries:
| group | n | mean \|Δ\| | frac exactly 0 |
|---|---|---|---|
| self_report | 104 | **0.0000** | 1.000 |
| immutable/proxy | 174 | 0.0021 | 0.707 |
| manipulable | 622 | 0.0276 | 0.447 |

- Self-reports are **fully shrunk to baseline** (the brief's "naive re-prediction"
  trap is explicitly avoided — `do(stated_revenue)` ⇒ 0 effect, which is the
  causally-correct answer for a self-report).
- Manipulable channels move materially; immutable channels move ~10× less (half
  shrink + low model dependence). **Not identical to naive** → avoids the FAIL case.
- **Caveat:** the shrink factors {0, 0.5, 1} are **hand-set, not learned** — defensible
  heuristic, but not a fitted SCM. (C is only 10% of score; see Step 4 cost/benefit.)

## 3.3 Deliverable D writeup — **PASS, with one unsupported claim**
- Exists; **all 5 required sections, correct order**: Problem framing / Methodology /
  Causal reasoning & counterfactual / Calibration & uncertainty / Limitations.
- **~1,170 words ≈ 2.1 pages** → within the 4-page limit, but **still markdown — the
  PDF (and ≥11pt / ≥0.75in / ≤4pp format gate) is unverified.**
- Causal section **does** distinguish observational vs interventional (explicitly).
- Claims-vs-code drift check: isotonic ✓, GroupKFold ✓, NPV decision ✓, self-report
  shrink ✓ — **all real**, EXCEPT:
  - **⚠️ The writeup claims "Drivers are explained with SHAP over an inspectable,
    de-correlated feature set." The code computes NO SHAP** (`grep` finds only
    "shape"/one comment; no `import shap`, no persisted SHAP artifact). We *did*
    de-correlate (the dedup view) citing SHAP as motivation, but never produced SHAP
    values. **Unsupported claim — fix the writeup or produce SHAP.**
- **Verdict: PASS** on structure/length/causal-distinction; **one unsupported claim**
  (SHAP) and PDF-render still pending.

## 3.4 Interpretability / top-feature legitimacy — **PASS** (SHAP gap noted)
- We have **gain importances** (`reports/feature_importance_preview.csv`), **not
  SHAP**, and no per-example rationales persisted.
- Top-10 (preview): invoice_delinquency, credit_utilization, volatility, vol×−trend,
  dscr, revenue_trend, daily_payment, req_to_stated_annual, buffer_to_payment,
  cash_balance_p10. **All are legitimate credit/affordability signals — none smell of
  leakage** (consistent with our leakage canary: no univariate AUC>0.90). Regulator-
  defensible.
- **Verdict: PASS** on legitimacy; **WEAK** on the SHAP gap (claimed, not delivered) —
  matters because the writeup (15%) leans on it as the driver-explanation method.

## Headline verdicts
| Item | Verdict |
|---|---|
| 3.1 C method + sanity battery | **PASS (strong)** — channel-aware, all sanity checks pass |
| 3.1 stale descendants | **PASS** — recompute is live (corr +0.714) + unit-tested |
| 3.2 proxy shrink vs naive | **PASS** — self-report=0 effect, not naive; shrink heuristic (note) |
| 3.3 writeup structure/length/causal | **PASS** — 5 sections, ~2.1pp, obs vs interventional clear |
| 3.3 writeup-vs-code drift | **FLAG** — SHAP claimed but not computed; PDF unrendered |
| 3.4 top features legitimacy | **PASS** — all defensible, no leakage; SHAP absent |
