# Deliverable D — Technical Writeup

**Team:** Vector Ventures

## 1. Problem framing & assumptions violated

Every assumption a credit scorecard leans on — random labels, population overlap,
a stationary window, honest inputs — breaks on this book; we name each break and
respond to it rather than modelling straight through it. We are re-underwriting a
historical SMB loan book to maximize realized portfolio NPV (total dollars earned),
not classification accuracy.

- **Labels are missing-not-at-random behind a sharp gate.** Outcomes exist only
  for loans the prior lender approved, and approval is, to numerical precision,
  `prior_underwriter_score >= 0.273`: a propensity model separates approved from
  declined with AUC 1.00 and **zero** score overlap. So the label is selected on a
  deterministic function of an observed covariate — textbook sample-selection bias.
  Inverse-propensity reweighting is infeasible by positivity violation, not by our
  choice: a propensity model on ordinary covariates alone still hits AUC 1.00. Only
  local extrapolation near the cutoff is available; we widen intervals on declines
  rather than pretend we can reweight.
- **The split is forward in time.** Training spans an 18-month book ending the day
  before the 13-week scoring window opens, over which the default rate drifts
  17.5% → 20.6%. We learn default *timing* on the history but recalibrate the
  *level* in-window — the history tells us the shape, only the window tells us
  the rate.
- **Missingness is information, not absence.** "Never declined elsewhere" (null)
  loans default at 14.3% vs 21.3% when present; no-bank-feed at 19.1% vs 16.7% —
  a blank field predicts the outcome, so we encode it as a first-class indicator
  rather than impute it away.
- **Self-reports are optimistically biased.** Applicants who overstate revenue by
  more than 1.5× default at 38.6% vs 16.1%. The *gap* between stated and bank-feed
  revenue — not the stated figure — is the signal, central to our causal treatment
  of `stated_annual_revenue` below.

## 2. Methodology

One discrete-time model serves all three deliverables: a calibrated probability
that a loan defaults, paired with a per-loan two-mode timing decomposition, read
out as an NPV sign for the approve/decline call.

- **PD model.** Five-fold `GroupKFold`(`business_id`) LightGBM ensemble over 55
  engineered features (OOF AUC 0.774, Brier 0.117), isotonic-calibrated then
  **refit on in-window val** to correct the 17.5% → 20.6% drift. Features are
  buffer-centric: loans are ~1.3% of annual revenue, so default is cash-buffer
  bound — debt-service coverage, buffer-to-payment, overdrafts, credit utilization
  dominate.
- **Two-mode hazard for timing.** 77.5% of defaults are missed-draw events over
  days 3–60; **zero** defaults occur on days 61–89; the remaining 22.5% spike
  exactly at day 90 (open-balance sweep). We model the two modes as separate heads:
  the PD head above, plus a `d90_frac` classifier `P(day-90 sweep | default, x)`
  (OOF AUC 0.72). Per loan, `CDR_i(a) = PD_i · (1−d90_frac_i) · F_early^{band(i)}(a)`
  for `a < 13` and `CDR_i(13) = PD_i`, where `F_early^b` is the band-conditional
  cumulative-default fraction within days 3–60. A's `E[NPV]` uses per-loan
  `E[t* | default, i] = (1−d90_frac_i) · E[t | early, band(i)] + d90_frac_i · 90`
  (exact under linear-in-t NPV).
- **Deliverable A — decision = NPV sign.** Approve iff `E[NPV_i] > 0`. Break-even
  PD ≈ 0.51 under two-mode timing. We shift the decision PD by `κ·σ_i` (σ_i fold
  disagreement, κ=2.25 by cross-fit on realized val P&L: OOF $3.89M vs $3.91M at
  κ=0), declining the most uncertain near-break-even loans. We approve **67%** of
  the 13,306 scored applicants; on labeled val the book returns **1.83×** the prior
  underwriter (capture **0.60** of perfect-foresight oracle).
- **Deliverable B — shape × level.** `CDR_{w,a}` averages the per-loan two-mode
  CDR above. The PD model has no cohort signal (training predates the window), so
  we shrink each cohort's *level* toward the val realized rate (pseudo-count 15)
  and *shape* toward the model band shape (Dirichlet c=50); we also rescale
  per-cohort PD for the two biased cohorts (5, 13; |bias| > 0.04). Val and test
  share the same 13 calendar weeks, so the empirical shape transfers. MAE
  0.013 → 0.003.

## 3. Causal reasoning & counterfactual methodology

The counterfactual target is `P(y=1 | do(f=v), X_{-f}=x_{-f})`, which differs
from the observational `P(y | f=v, X_{-f})` whenever `f` is confounded. Our
feature **registry** is the consistency layer: each engineered feature declares
its parents, so setting a raw feature and recomputing propagates the
intervention to exactly its descendants — never an internally-contradictory
applicant. E.g. `do(requested_amount)` flows to daily-payment, buffer-to-payment,
leverage, and ratio features; the leaky provided `requested_amount_to_observed_revenue`
is never reused.

We classify intervenable features by causal status:

- **Mechanical causes** — `requested_amount`: changing the number changes real
  payment burden, so we perturb and propagate at full strength.
- **Confounded proxies** — bureau and bank-feed symptoms of latent business
  health (utilization, delinquency, revenue and cash signals): we keep only the
  causal fraction `λ̂ = β_adj / β_naive` from sibling-adjusted logistic
  regression (per-family ~0.52 / 0.51 / 0.64 for bureau / bank-feed / behavioral),
  averaged with the heuristic, falling back where `λ̂ ∉ (0,1)`. **Adjustment
  sets are derived algorithmically from an explicit DAG** by the backdoor
  criterion (condition on co-proxies of latent H, exclude graph descendants).
  Excluding cash's descendants (invoice delinquency, overdrafts) corrects
  `λ̂_cash` from an implausible 0.03 to 0.44.
- **Self-reports** — `stated_annual_revenue`, `stated_time_in_business`: return
  the observational PD. A different number on the form changes nothing about the
  business, so the interventional effect is ≈ 0 — the causally correct answer,
  not naive re-prediction.
- **Immutable / proxy-by-fiat** — `sector`, `geography_region`, `vintage_years`,
  `has_linked_bank_feed`: not levers but queried anyway; perturb, shrink, widen,
  flag.

**Regulator defense.** Drivers explained with SHAP over an inspectable,
de-correlated feature set. Top drivers are invoice-payment delinquency, credit
utilization, revenue volatility, requested amount, overdrafts, and the
affordability ratios — all recognized credit/affordability signals in the
expected direction, none a prohibited or proxy-discriminatory attribute. A
transparent logistic regression on the same engineered features matches the
gradient-boosted ensemble (OOF AUC 0.775 vs 0.770, equal Brier) and is *more*
robust to train-to-window drift (val AUC 0.759 vs 0.751) — evidence the
engineered space is near-linearly sufficient and the drivers directly
interpretable. Several queried values lie outside observed support (e.g.
`prior_loans_count` is in-support only 29% of the time); those receive wider
intervals and an explicit extrapolation caveat.

## 4. Calibration & uncertainty quantification

- **Point PD** is isotonic-calibrated and **refit on in-window validation**
  (17.5% → 20.6% drift). Valid on the approved manifold; extrapolated on declines.
- **90% PD intervals (A, C)** are additive, `p ± α·z·σ` (σ = fold disagreement,
  α=1.1 fit by 5-fold cross-fit on val). On labeled val: decile coverage 1.00,
  per-cohort coverage 12/13, mean width 0.064. The cross-fit reading already
  hits 0.90, so the shipped over-coverage is the post-isotonic-refit tail
  margin, not slack.
- **90% trajectory intervals (B)** combine within-cohort bootstrap with a
  conformal band against true val cohort trajectories. Coverage ≈ 0.94, mean
  width 0.048. Out-of-support C intervals are widened further.

## 5. Limitations & what we'd do differently

Our honest gaps trace to one fact — we never observe outcomes for declined
applicants — so the decline and counterfactual estimates rest on extrapolation
we cannot check.

- **The unverifiable core.** All C effects lack labels (the queried applicants
  were never decided), and declines never overlap the approved manifold — so
  reject inference is infeasible, only RD-style local extrapolation remains
  (deferred), and decline-side PD stays unchecked. Self-report and immutable
  counterfactuals stay heuristic; confounded proxies use the DAG-derived `λ̂`,
  but latent H breaks causal sufficiency.
- **Val-tuned knobs** (κ=2.25, level pseudo-count 15, shape c=50, per-cohort PD
  scaling for cohorts 5 and 13) transfer to test only as far as the windows
  match (same 13 weeks; per-cohort n ≈ 150 adds variance). The κ rule absorbs
  the flagged over-approval, but a residual oracle gap remains (≈ 0.60 of
  perfect-foresight P&L), irreducible without label foresight on declines.
- **With another day:** sharp-RD local-linear PD anchor across the 0.273 cutoff
  to calibrate decline-side PD (identifiable via continuity even though
  positivity fails), `cohort_week` as a feature in the d90 head (would tighten
  the per-cohort d90-share residual, ~7pp on cohort 13), and a sealed
  val-tuning holdout to honestly trade off the seven val-fit calibrations
  against single-split overfit risk.
