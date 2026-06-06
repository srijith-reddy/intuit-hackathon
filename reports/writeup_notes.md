# Deliverable D — Running Writeup Notes

Quotable material logged as we discover it, under D's five required headings.
Section 3 (causal) carries the most weight — log causal arguments aggressively.

## 1. Problem framing & assumptions violated
- **MNAR labels via a sharp deterministic gate.** Outcomes exist iff prior lender
  approved, and approval = `prior_underwriter_score ≥ 0.273` exactly (AUC 1.0, 0%
  score overlap). i.i.d. and MCAR both broken; the training label is selected on a
  deterministic function of an observed covariate → textbook sample-selection bias.
- **Covariate/label shift (forward-in-time split).** Train (2024-01→2025-06) precedes
  val/test (the 13 cohort weeks, 2025-07→2025-09). Default rate drifts 17.45%→20.62%.
  Standard CV understates test error.
- **Positivity violated for the obvious reweighting.** Zero score overlap ⇒ IPW on
  the prior score is undefined; reject inference must use overlapping covariates or
  the RD structure at the 0.273 cutoff.
- **"Missingness" is mostly structural information, not absent data** (never-declined,
  no-inquiry, no-feed) — violates the usual impute-and-forget assumption.

## 2. Methodology
- Buffer-centric affordability (loans are ~1.3% of annual revenue; the constraint is
  cash buffer vs daily ACH ≈ $420, not revenue coverage).
- **A is a profit-maximization decision, not a PD classifier.** Approve iff E[profit]>0
  ⇔ PD < break-even. Revenue if repaid = 8.75% of principal (3% fee + 5.75% interest/60d,
  ~$2,077 avg). LGD is the swing factor: defaulters pre-pay ~59% of schedule via ACH
  before defaulting (default ~day 36/60), so **draws-aware LGD≈0.32 → PD*≈0.21**, while
  **recovery-only LGD≈0.91 → PD*≈0.09** (post-hoc recovery rate is only ~9%). Portfolio
  default rate 17.4% sits between → the loss assumption flips the policy. We make the
  decision robust across LGD∈[0.32,0.91] and state the assumption explicitly.
- Baseline GroupKFold LGBM: OOF AUC 0.770, Brier 0.118 — PD accuracy ceiling is modest;
  points are in economics + calibration + causal story, not AUC.
- **EXACT scoring (brief p.14): S = 0.30·P&L + 0.25·Trajectory + 0.20·Calibration +
  0.10·Counterfactual + 0.15·Writeup.** Priority A>B>Cal>Writeup>C. (C is only 10%.)
- **EXACT NPV (brief p.8):** repaid → F+R·r·T/365; default@t* → F+D(t*-1)+rec-R. Decision
  d=1[E[NPV|approve]>0], not a flat PD threshold. Draws-aware LGD=0.274, break-even PD≈0.24.
- **A↔B coupling (brief p.9):** NPV depends on default *day* (day-5 loss ≈ -R; day-55 ≈
  break-even). So E[NPV] needs the timing distribution → A and B share ONE discrete-time
  hazard model. This is the architectural keystone.
- C scored as closeness to TRUE interventional effects; "not naive re-prediction" — naive
  perturbation penalized when the feature is confounded (brief p.10).
- GroupKFold(business_id) OOF; Ridge/Elastic baseline beside LGBM; SHAP for drivers.
- Recompute affordability ratios cleanly (provided `requested_amount_to_observed_revenue`
  is requested÷observed-*annual* revenue with a silent stated fallback — leaky/mislabeled).
- B: discrete-time hazard (no censoring; payoffs all at day 60), shape from train,
  level per-cohort from val, monotone by cumulating non-neg hazards.
- **B is two default modes, not one curve:** 77.5% of defaults are missed-draw events
  spread over days 3–60 (weeks 1–9); 22.5% are a single point mass at day 90 (the
  balance>0 sweep) with a dead zone days 61–89. Trajectory: concave rise wk1–9, flat
  wk9–12, +3.9pp cliff at wk13. A smooth survival fit mis-shapes this; model the two
  components separately.

## 3. Causal reasoning & counterfactual methodology  ← heaviest weight
- **Observational vs interventional, made concrete by the data:** `stated_annual_revenue`
  is a self-report; `do(stated_annual_revenue=x)` should have ~0 *causal* effect on
  default (it changes a claim, not the borrower's cash flow), yet a purely
  observational model attaches risk to it via its correlation with misreporting. The
  overstatement→default lift (38.6% vs 16.1%) is *associational*, driven by the
  *gap* to observed, not by the stated level itself. This is the cleanest
  observational-vs-interventional example in the dataset.
- **`do(requested_amount=x)` is a genuine intervention:** it changes the actual daily
  ACH burden, so its descendants (buffer-to-payment, DSCR, payment-shock) must be
  recomputed — motivates the feature registry as the causal-consistency layer.
- **Intervenable-by-fiat vs by-nature:** the query set forces `do()` on immutable
  identity features (sector, geography, vintage_years, has_linked_bank_feed). We will
  state plainly which interventions are physically meaningful vs proxy-perturbations,
  and widen intervals / flag extrapolation for out-of-support values
  (prior_loans_count queries only 29% in-support).
- **RD as a regulator-defensible causal anchor:** the sharp 0.273 cutoff is a natural
  experiment we can invoke to argue local causal validity.

## 4. Calibration & uncertainty quantification
- Calibrate PD on in-window val (the only labeled in-window data), but remember val is
  itself approval-selected → calibration is valid on the approved manifold, extrapolated
  on declines (state this).
- B intervals dominated by within-cohort sampling error (n≈150–220/cohort), not model
  variance → bootstrap loans within cohort.
- C intervals must widen for out-of-support `do()` values.

## 5. Limitations & what we'd do differently
- PD on declined applicants is extrapolation; no decline outcomes anywhere (not even val).
- Counterfactuals on immutable/proxy features are not true causal effects.
- Reject inference (RD-based or IPW-on-covariates) deferred to Phase 2.
