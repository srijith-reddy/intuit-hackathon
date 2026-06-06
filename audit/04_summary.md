# Audit 04 — Final Gap Report & Scorecard

Read-only audit of our existing solution against three playbooks (Home Credit
stability, AmEx missingness/calibration/decisions, Illinois/WiDS causal/finish-
everything). Nothing was modified. Score model: **S = 0.30·P&L + 0.25·traj +
0.20·cal + 0.10·C + 0.15·write**. All numbers reproduce via `audit/scripts/s1..s3`.

## Scorecard

| # | Audit item | Verdict | Leaks from | Est. impact |
|---|---|---|---|---|
| 0 | Submission validity (A/B/C) | **PASS** | — | gate met |
| 0 | Deliverable D as **PDF** | **FAIL** (markdown only) | write 0.15 | **HIGH** (D unscoreable w/o PDF) |
| 1.1 | Per-cohort PD stability (AUC spread 0.116) | WEAK | traj 0.25, cal 0.20 | MED |
| 1.2 | Leaning on drifting features (#1 feat drifts) | WEAK | traj 0.25 | MED |
| 1.3a | B monotone in age | PASS | — | — |
| 1.3b | B accuracy vs realized (MAE 0.016; ±0.06 tails) | WEAK | traj 0.25 | MED |
| 1.4 | Timing distinguishes early/late (pooled shape) | WEAK | traj 0.25, P&L 0.30 | MED |
| 2.1 | Missingness (NaN native + indicators) | **PASS** | — | (strength) |
| 2.2 | Reject inference (uncorrected, bias −0.032) | WEAK | P&L 0.30, cal 0.20 | LOW* |
| 2.3 | Calibration: ECE 0.025 but coverage ~100% (over-wide) | WEAK | cal 0.20 | **MED-HIGH** |
| 2.3 | PD level bias (under-predicts ~−0.025) | WEAK | cal 0.20, P&L 0.30 | **HIGH** |
| 2.4 | Decision rule = exact NPV, 1.73× prior underwriter | **PASS** | — | (strength) |
| 2.4 | P&L capture 0.564 of oracle (over-approves) | WEAK | P&L 0.30 | **HIGH** |
| 2.4 | Decision robustness (5% flips under ±20%/±50%) | **PASS** | — | (strength) |
| 3.1 | C channel-aware + sanity battery (all pass) | **PASS** | — | (strength) |
| 3.1 | Stale descendants in `do()` | **PASS** (live, tested) | — | (no bug) |
| 3.2 | Proxy shrink vs naive (self-report=0, not naive) | **PASS** | — | (strength) |
| 3.3 | Writeup 5 sections / ≤4pp / obs-vs-interventional | **PASS** | — | — |
| 3.3 | **SHAP claimed but not computed** (writeup-vs-code drift) | **FLAG** | write 0.15 | LOW-MED |
| 3.4 | Top-feature legitimacy (no leakage) | **PASS** | — | (strength) |

\* 2.2 is WEAK but largely *unfixable* — see "data outranks playbook" below.

## Ranked recommendations (impact-per-effort; specs are self-contained)

### BUGS / completeness — fix regardless
**B1. Export Deliverable D to PDF (and verify ≤4pp, ≥11pt, ≥0.75in).**
Without the PDF the 15% writeup cannot be scored — this is the single biggest at-risk
chunk and it's nearly free.
- *File/where:* new `make pdf` or `quarto render submissions/submission_D_writeup.md
  --to pdf` → `submission_D_writeup.pdf`. quarto 1.8.26 is installed.
- *Change:* render; confirm page count ≤4 and font/margins.
- *Verify:* `validate_submission.py` warning disappears; visual ≤4 pages.
- *Effort:* trivial. *Impact:* protects 15%.

**B2. Reconcile the SHAP claim (writeup-vs-code drift).**
Writeup §3 says drivers are explained with SHAP; code computes none.
- *Option A (cheap):* edit `submissions/submission_D_writeup.md` to say "gain-based
  importance over a de-correlated feature set" (what we actually do).
- *Option B (better, +S_write):* add `audit`-style SHAP on the trained model, persist
  a summary plot, cite it. (Note: model isn't persisted — would require exposing it.)
- *Verify:* every writeup claim maps to code/artifact.
- *Effort:* A trivial / B low. *Impact:* integrity of 15% section.

### ENHANCEMENTS — by score-impact-per-effort
**E1. Recalibrate PD level on in-window val (fixes the −0.025 under-prediction).** ⭐ top
Single root cause leaking from **two** components (cal 0.20 + P&L 0.30). Isotonic is fit
on train OOF (17.5% base) but the scored window runs 20.6% → systematic under-prediction
→ over-approval (capture 0.564, and PD+20% *raised* realized P&L).
- *File/function:* `src/models.py::train_pd_model` / `src/submit.py::build` — fit (or
  refit/blend) the isotonic calibrator on **labeled validation** PD↔outcome instead of
  (or in addition to) train OOF; or add a logit-shift chosen to match the val base rate.
- *Expected movement:* pooled ECE 0.025→~0.01; per-cohort \|cal_gap\| mean 0.026→<0.015;
  P&L capture 0.564→~0.65–0.75; approve rate drops toward oracle's 0.853.
- *Verify:* re-run `audit/scripts/s2_pd_decisions.py` — cal_gap→0, oracle capture up,
  approve rate ↓.
- *Effort:* LOW. *Impact:* **HIGH** (touches 50% of the score).

**E2. Re-target interval width to 90% (stop over-covering).**
Decile/cohort coverage is ~100% vs the 90% target → "needlessly wide" penalty (mean
width 0.152). leaks from cal 0.20.
- *File/function:* `src/calibration.py::fit_pd_interval_scale` — finer `alpha` grid and
  target *realized* decile coverage = 0.90 (not binned-at-20 ≥0.90 which overshoots);
  same idea for `b_conformal_halfwidth` (currently scaled to 0.935).
- *Expected:* A width 0.152→~0.09–0.11 at ~0.90 coverage; B width similar; S_cal up.
- *Verify:* `s2` decile & cohort coverage ≈0.90; widths down.
- *Effort:* LOW. *Impact:* **MED-HIGH** (direct on 20%).

**E3. PD-/band-conditional timing shape (B + A).**
Pooled `S(t)` mis-times per-segment (shape varies ~13 days by credit band; bad-credit
default earlier, good-credit carry the day-90 mass). leaks from traj 0.25 (+ small P&L).
- *File/function:* `src/survival.py` — add `weekly_shape_by_segment` (segment = PD
  tertile or `owner_personal_credit_band`); use the segment shape in `submit.py` B grid
  (`CDR = mean_i S_seg(i)(a)·PD_i`) and in A's `expected_npv` (per-loan E[t*] from the
  segment shape instead of pooled `t_bar=43`).
- *Expected:* B tail-cohort errors (cohort 5/13, currently ±0.06) shrink; B MAE
  0.016→~0.012; A E[NPV] timing more accurate (modest P&L).
- *Verify:* `s1` B per-cell error heatmap; tail cells improve.
- *Effort:* MED. *Impact:* **MED-HIGH** (on 25%).

**E4. Per-cohort calibration / drift dampening (stability).**
Per-cohort AUC spread 0.116 and reliance on drifting features. Mostly *symptom* of E1;
remaining gap is genuine cohort heterogeneity. leaks from traj 0.25 + cal 0.20.
- *File/function:* `src/submit.py` B path — apply a light per-cohort level adjustment to
  CDR using val cohort rates (shrunk toward global to avoid overf_n≈200_); optionally
  add time-aware CV in `models.py` for model selection.
- *Expected:* per-cohort cal_gap spread ↓; B tail accuracy ↑.
- *Verify:* `s1` per-cohort metrics, `s2` per-cohort coverage.
- *Effort:* MED. *Impact:* MED (overlaps E1/E3 — do after them).

**E5. Learned C shrink (only if time allows).**
Shrink {0,0.5,1} is heuristic. C is **10%** and already passes every sanity check, so
this is low priority.
- *File/function:* `src/submit.py` C section — replace fixed factors with effects from a
  small structural/mediation model on the intervenable subgraph.
- *Effort:* HIGH. *Impact:* LOW (10% component, already PASS).

**Suggested order:** B1 → E1 → E2 → B2 → E3 → E4 → (E5). B1+E1+E2 are low-effort and
touch ~85% of the score.

## What we already do WELL (don't churn these)
- **Decision rule is the exact NPV sign**, not a PD threshold — and it **beats the prior
  underwriter 1.73×** on realized val P&L. (AmEx decision-economics: nailed.)
- **Missingness as signal**: NaN passed natively + explicit indicators. (AmEx: nailed.)
- **C is channel-aware with correct causal treatment**: self-reports → 0 effect,
  `do(x=observed)`≈baseline (|Δ|=0), `do(requested_amount)` monotone (corr +0.71),
  descendants recompute live (no stale bug), unit-tested. (Illinois/WiDS causal: nailed.)
- **B is monotone**; **decisions are robust** to ±20% PD / ±50% recovery (5% flips);
  **no leakage** (canary + legitimate top-10). Validator **PASSES**.

## Where our DATA outranks the playbook
- **"Do reject inference (IPW)" (AmEx/credit-risk lore) — infeasible here.** A propensity
  model `P(approve|X)` reaches **AUC 1.000 without any selection features**: approved and
  declined populations are **perfectly separable** in covariate space → positivity is
  violated → IPW/reweighting cannot be computed. Our deferral is partly *correct*; only
  RD-style local extrapolation is even available. (Item 2.2 is WEAK but mostly unfixable.)
- **"Drop unstable features" (Home Credit) — would hurt us.** The features that drift most
  (`invoice_payment_delinquency_rate`, `observed_revenue_trend_3mo`) are also our
  **strongest predictors**. The data says **recalibrate/monitor, don't drop** (E1/E4),
  else AUC collapses.
- **Intervals**: the playbook prefers tight intervals; our val data confirms we are the
  *opposite* of under-covering (≈100% coverage) → we can safely **tighten** (E2) with no
  coverage risk — a rare "free" S_cal gain.

## One-line bottom line
Solid, complete, validator-passing submission with **no algorithmic bugs** found; the
**highest-leverage moves are non-modeling**: ship the **PDF** (B1) and **recalibrate the
PD level + tighten intervals** (E1, E2) — together they touch ~85% of the score for ~a
few hours of work.
