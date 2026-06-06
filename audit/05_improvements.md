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
