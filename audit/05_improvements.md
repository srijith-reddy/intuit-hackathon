# 05 — Post-submission improvements (cross-fit, no test tuning)

All metrics are realized on **labeled validation** (n=2,551, prior-approved). Realized
P&L uses the exact brief NPV with actual `days_to_default` and `final_recovered_amount`
(same logic as `audit/scripts/s2_pd_decisions.py`). Baseline = the shipped rule before
this round: realized val P&L **$3,726,634**, capture vs oracle 0.569, vs approve-all 1.742.

Guardrail on C: `python -m src.submit` regenerates a C that is **numerically identical**
to the committed one but differs in float-string formatting (16 vs 17 digits). The
committed, sha-locked C (`6c113d08…`, produced by `patch_C_lambda_fix.py`) is preserved
byte-for-byte via `git checkout` after each rebuild. C is not touched.

---

## Task 1 — κ-shifted decision rule (target S_P&L, 30%) — ACCEPTED

`approve iff E[NPV](p_i + κ·σ_i) > 0`, σ_i = per-loan fold-ensemble disagreement
(reused from the interval model, no retrain). κ chosen on a grid {0,0.25,…,3.0} by
realized val P&L, with a 5-fold cross-fit OOF check (per-fold κ picked on the other 4
folds, scored on the held-out fold). Implemented in `calibration.fit_kappa_decision_shift`,
wired in `submit.py` (E5). Reported PD/intervals are unchanged — only the decision uses p+κσ.

- **κ\* = 1.25.**
- κ→P&L curve (full labeled val): {0.0: 3,726,634; 0.25: 3,762,318; 0.5: 3,762,318;
  0.75: 3,762,318; 1.0: 3,760,827; **1.25: 3,772,684**; 1.5: 3,719,413; 1.75: 3,719,429;
  2.0: 3,693,164; 2.25: 3,677,014; 2.5: 3,713,455; 2.75: 3,716,405; 3.0: 3,675,773}.
- **Cross-fit OOF P&L (adaptive κ) = $3,760,827 > baseline $3,726,634** (+$34,193, +0.92%).
  Fold picks = [1.25, 1.25, 1.25, 1.25, 0.25] (4/5 agree on 1.25).
- Shipped (κ\*=1.25) on the regenerated submission_A: realized val P&L **$3,772,684**
  (+$46,050, +1.24%), capture vs oracle **0.569 → 0.576**, vs approve-all **1.742 → 1.763**,
  labeled-val approve 0.845 → 0.842 (full scored set approve 0.73 → 0.724).
- **Acceptance: OOF realized val P&L strictly > current rule's. PASS.**

## Task 2 — exact timing integration in NPV (target S_P&L) — ACCEPTED (neutral, exactness)

Replaced the plug-in mean default day with the explicit expectation
`E[NPV]=(1-p)·rev + p·Σ_t w_b(t)·NPV_default(t)`, w_b = band-conditional **daily** default-day
distribution (`survival.daily_dist_by_band`, rows sum to 1), integrated in
`submit._exp_npv_default`. Combined with Task 1 (integrate timing, then shift PD).

- **Neutral by construction:** the brief's NPV_default is linear in t\* (no kink/clip), so
  `Σ_t w_b(t)·NPV(t) = NPV(Σ_t w_b(t)·t) = NPV(E_b[t])` — exactly the daily-mean plug-in.
  Confirmed empirically: κ=0 P&L = $3,726,634, identical to the pre-change baseline.
- Kept anyway per the task — it is the exact expectation, zero risk, makes the day-90 mass
  explicit, and is robust if NPV ever becomes nonlinear in t (e.g. day-dependent recovery).
- Note: `economics.expected_npv`'s `default_day_dist` branch omits the `pd_hat` factor
  (returns `(1-pd)·rev + Σdist·npv` instead of `+ pd·Σdist·npv`); it is dead code in the
  scored path, so left untouched — integration is done directly in `submit.py`.

## Task 3 — hierarchical SHAPE shrinkage for B (target S_traj, 25%) — ACCEPTED

Extends E4's per-cohort LEVEL shrinkage to the SHAPE. For each cohort, the empirical
val timing increments are blended toward the model band shape, Dirichlet-style:
`blended = (n_w·empirical + c·band_shape)/(n_w + c)`, renormalized, cumulative kept
monotone (`submit.py` E6). The concentration c is chosen by **split-half cross-fit
within validation** (`calibration.fit_shape_shrinkage_c`): estimate the shape on a random
half of each cohort, score the blended curve against the held-out half's realized CDR —
this avoids the c→0 self-prediction trap (a cohort's own realized shape predicting itself).

- Grid LOCO (split-half) half-MAE: {10: 0.0290, **25: 0.0285**, 50: 0.0326, 100: 0.0292,
  200: 0.0304} → **c\* = 25**.
- B grid MAE vs realized val CDR (s1 §1.3 metric): **0.0125 → 0.0044**. Monotone: **True**.
- Tail cohorts max abs error: cohort 5 **0.0326 → 0.0098**, cohort 13 **0.0406 → 0.0240**.
- B interval coverage vs realized val CDR: **1.000** (≥ 0.88 guard; now over-covered →
  tightened in Task 4b), mean width 0.0571 → 0.0565.
- **Acceptance: realized-val MAE < 0.0125, tails shrink, monotone. PASS.**
- Honesty: the 0.0044 grid MAE is partly in-sample (the point now incorporates val timing,
  borrowed-strength as in E4); the split-half LOCO half-MAE (0.0285) is the honest
  c-selection metric and confirms c>0 generalizes. Test benefit relies on the val→test
  per-cohort shape transfer (same 13 calendar weeks), exactly as for the level shrinkage.

## Task 4a — normalized asymmetric conformal A intervals (target S_cal, 20%) — REVERTED

Tried interval = [p+q_lo·σ, p+q_hi·σ] with q_lo,q_hi = 0.05/0.95 quantiles of the signed
normalized residual (y−p)/σ (cross-fit on val).

- **Result: mean width 0.064 → 0.83 (13× WORSE), decile coverage 1.0.** Reverted.
- **Why it fails (principled):** the conformal score |y−p|/σ divides a *binary-outcome*
  residual (|y−p| ≈ 0.2–0.8) by the *epistemic* ensemble σ (≈0.005–0.02), so normalized
  residuals are O(20–80) and the 0.95 quantile (≈76) yields ~0.8-wide intervals. Conformal
  that targets individual 0/1 outcomes inherently needs ~unit width; the scored object here
  is the *binned default rate*, which the existing α-additive interval already covers at
  0.90 decile coverage / 0.053–0.064 width. Acceptance (width < 0.064) impossible. Kept the
  α-additive A interval unchanged (also what C's half-width depends on).

## Task 4b — multiplier-bootstrap simultaneous B bands (target S_cal) — REVERTED

Tried: iid N(1,1) per-loan multipliers → recompute cohort CDR per draw (500); band =
point ± q_sup·SE(a), q_sup = 0.90 quantile of sup_a |dev|/SE. Replaces resample+conformal.

- **Result: mean width 0.0565 → 0.0114 (narrower) but coverage COLLAPSES** — cell 1.000 →
  0.675, simultaneous (whole-curve) 0.308. Acceptance (simultaneous ≥ 0.88) fails. Reverted.
- **Why it fails:** the wild bootstrap captures only *within-cohort sampling* variance
  (SE ≈ 0.002–0.005 for ~150-loan cohorts); the realized curve deviates from the point by
  the *model/shape residual* (up to ~0.024), which the dropped conformal half-width was
  covering. The pointwise-0.90 fallback is even narrower → worse coverage, so it does not
  rescue it (the task's fallback is for the over-wide case, not under-coverage). Kept the
  resample-bootstrap + conformal band (coverage 1.000, width 0.0565).

---

## Task 5 — final regenerate + audit (old vs new)

`python -m src.submit` → validator **PASS**; `submission_C` restored byte-identical
(`6c113d08…`); A & B regenerated. Accepted changes: Tasks 1, 2, 3. Reverted: 4a, 4b.

| Metric (labeled-val realized) | Baseline | Final | Δ |
|---|---|---|---|
| A realized val P&L | $3,726,634 | **$3,772,684** | +$46,050 (+1.24%) |
| A P&L cross-fit OOF (κ adaptive) | $3,726,634 | **$3,760,827** | +$34,193 (+0.92%) |
| A capture vs oracle | 0.569 | **0.576** | +0.007 |
| A P&L vs approve-all | 1.742 | **1.763** | +0.021 |
| A approve rate (labeled val / full set) | 0.845 / 0.730 | 0.842 / 0.724 | slightly tighter |
| A decile coverage / width | 0.90 / 0.064 | 0.90 / 0.064 | unchanged (4a reverted) |
| B MAE vs realized val CDR | 0.0125 | **0.0044** | −0.0081 |
| B tail max-abs: cohort 5 / 13 | 0.0326 / 0.0406 | **0.0098 / 0.0240** | both shrink |
| B monotone / coverage / width | True / 1.00 / 0.0571 | True / 1.00 / 0.0565 | unchanged (4b reverted) |
| C (counterfactuals) | 6c113d08 | 6c113d08 | untouched |

Net: S_P&L (30%) and S_traj (25%) improved; S_cal (20%) unchanged (both interval
redesigns failed acceptance and were reverted); S_C (10%) untouched. All cross-fit /
within-val; no test tuning.
