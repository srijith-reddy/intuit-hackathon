# Polish Log — evidence-backed, gated, revertable changes

Baseline = shipped submission on `main` (validator PASS, tests 10/10, E1–E4 banked).
Branch `polish/*`. Each change: gate defined first, measured before/after; shipped
state wins ties. Any new val fitting is cross-fit (we already spent val on E1/E4).

Probe that motivates this pass:
```
OOF (GroupKFold)   AUC      Brier
  lgb (shipped)   0.7697   0.1181
  xgb             0.7708   0.1176
  logit           0.7754   0.1169   <- best single
  blend           0.7746   0.1169
corr(cross-family std, |error|) = +0.54; std 3.2x wider in top vs bottom error decile
```

---

## Part 1 — Cross-family σ for A intervals (S_cal, 20%)
Gate (all on cross-fit val): honest coverage ∈ [0.88,0.92]; mean width ≤ shipped 0.064;
per-cohort coverage ≥ 0.85; per-decile coverage ≥ 0.85; width ADAPTIVE across error
deciles; report α.

**RESULT: REJECTED** (no pipeline change). Measured gate-first via
`audit/scripts/p1_measure.py` (read-only), honest OOF-point + cross-fit-α protocol.

| metric (honest, val) | SHIPPED fold-σ | NEW cross-family-σ |
|---|---|---|
| cross-fit α | 1.10 | 1.20 |
| decile coverage (cross-fit α, full val) | 0.90 (9/10) | 0.90 (9/10) |
| mean width | **0.0531** | 0.0549 (wider) |
| per-cohort coverage | 0.77 (10/13) | 0.85 (11/13) |
| width adaptivity (hi/lo err decile) | **2.58×** | 2.21× |
| corr(σ, \|error\|) on val | **+0.62** | +0.50 |

**Why rejected:** the motivating signal (probe's +0.54 family-σ↔error on *train* OOF)
**did not transfer to val** — fold-σ actually correlates with realized error *more*
(0.62 vs 0.50). Cross-family-σ is slightly **wider** and **less adaptive**; its only
edge is per-cohort coverage, a **single cohort** (10/13→11/13) within sampling noise at
~150 loans/cohort. The change adds real complexity (training xgb+logit families in the
ship path) for no honest improvement. Shipped fold-σ wins the tie. Did not iterate.

*Note:* the in-sample-point variant of the gate floored α and collapsed width to ~0.014
(the documented E1/E2 failure mode); the honest protocol uses OOF-calibrated val points.

*Side finding:* shipped per-cohort interval coverage is ~0.77 (3 of 13 cohort rates fall
outside the mean interval) — a known consequence of calibrating A intervals at
per-applicant/decile granularity; not fixed by cross-family-σ.

---

## Part 2 — Logistic parity (S_write, 15%)  — **SHIPPED** (writeup/reports only)
No pipeline change (point preds, decisions, intervals, B, C all unchanged). Added
`reports/model_family_comparison.md` and a §3 "Model-class check" paragraph.
- logit best single model: OOF AUC 0.7754 / Brier 0.1169; val AUC **0.759 vs LGB 0.751**
  (more drift-robust); 8/10 coefficient signs agree, 2 collinearity sign-flips
  (`buffer_to_payment`, `debt_to_revenue`) — motivates SHAP+registry over raw coefficients.
- Writeup back to 4pp (shrank fig1 0.82→0.70, fig3 0.6→0.5; trimmed §2/§4/§5 prose). PASS.

---

## Part 3 — Explicitly REJECTED changes (on record, with evidence)
- **Optuna / hyperparameter search — REJECTED.** Model-family probe: all learners within a
  0.006 AUC / 0.001 Brier band (lgb 0.7697, xgb 0.7708, logit 0.7754). Tuning chases noise
  inside that band and risks the val set that certifies our calibration. No score path.
- **XGB / stacking for accuracy — REJECTED.** Blend (OOF AUC 0.7746) < best single (logit
  0.7754); ensembling does not beat the best base. Not accuracy-bound (audit-confirmed).
- **Cross-family σ for intervals — REJECTED** (Part 1): hypothesis falsified on val.
- **HMM / regime meta-learning — REJECTED.** E3 (band-conditional timing) + E4 (per-cohort
  EB level) already are the context-conditional adjustment at the right scale for 13 cohorts
  × ~150 loans; a fuller regime stack (project4 §6.3) overfits here.
- **Model swap to logit — REJECTED.** Despite the ≤0.006 AUC edge, swapping the point model
  cascades through every deliverable — E1 level calibration, the A intervals, the E3 hazard
  interaction, and all Deliverable-C baselines/registry re-predictions run through the
  shipped LGB — and re-spends val honesty re-validating the chain, for a noise-band gain.
  The logit's value is evidentiary (Part 2), not as the shipped predictor.

---

## Part 4 — Deliverable C λ̂ for confounded proxies (S_C, 10%)  — **SHIPPED**
Measured via `audit/scripts/p4_lambda.py` → `reports/lambda_hat.csv`.
- **Eigenvalue check:** proxy block eigenvalues [3.51, 1.65, 1.11, …], ev1/ev2 = 2.13 (<2.5)
  → several comparable factors → use **sibling-proxy ratio** λ̂ = β_adj/β_naive (plain logistic).
- **λ̂ (8/12 valid in (0,1)):** utilization 0.52, invoice-delinq 0.64, obs_monthly_rev 0.59,
  trend 0.67, volatility 0.73, cash_balance 0.03, overdraft 0.42, payroll 0.13. Per-family
  mean: bureau 0.52, bank-feed 0.43, behavioral 0.64. **4 fall back** (sign-flip under
  sibling adjustment → mediator/collinearity): recent_inquiries, existing_debt, credit_band,
  multi_lender_inquiry → keep shipped heuristic.
- **Change:** confounded proxies (currently full-strength = the penalized "naive
  re-prediction") shrunk by λ̂; final p_cf = mean(heuristic, λ̂). requested_amount stays full
  mechanical; self-reports stay ≈0; immutables/fallbacks unchanged.
- **Gate (sanity, no labels) — ALL PASS:** p_cf∈[0,1] ✓; l≤p≤u ✓; λ̂∈(0,1) ✓;
  do(observed)≈base max|Δ|=0.000<0.005 ✓; self-report still 0 ✓; do(requested_amount)
  monotone corr 0.64 ✓. Confounded-proxy effect now 0.033 mean |Δ| from baseline (was naive
  full strength); mean |Δ vs heuristic| = 0.0036. Tests 10/10, B monotone, validator PASS.
- Writeup §3 updated (heuristic→estimated λ̂ + per-family table) and §5 hedge; PDF 4pp.
