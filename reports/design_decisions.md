# Design Decisions (forward-looking, per deliverable)

How the feature pipeline serves each future deliverable, and the decisions locked in
Phase 1. Cross-references: `eda_findings.md`, `feature_catalog.md`, `writeup_notes.md`.

## A — Lending policy (approve/decline + calibrated PD + 90% interval)
- **Decision rule = maximize expected profit, not threshold PD at 0.5.** Approve iff
  `E[profit] > 0` ⇔ `PD < break-even`. Under the product economics
  (`src/economics.py`): revenue if repaid = 8.75% of principal (3% fee + 5.75%
  interest/60d); LGD from `final_recovered_amount` **and** pre-default ACH draws.
- **The LGD assumption is the dominant lever** (break-even PD ∈ [0.09, 0.21] for LGD
  ∈ [0.91, 0.32]). Policy will be reported across that range and the assumption
  stated explicitly; default to the draws-aware LGD (the lender truly collected those
  draws) unless the scoring rewards the conservative one.
- **FE supports this:** a calibrated PD model (engineered set, AUC 0.774) + the
  recovery-rate distribution from EDA (mean 9% recovery; draws-aware LGD ≈ 0.32).
- **PD on declines is extrapolation** (no decline outcomes anywhere). Decline PDs get
  wider intervals; reject-inference reweighting deferred to Phase 2.

## B — Default-timing trajectory (13×13, monotone, intervals)
- **Two-component model** (locked): discrete-time missed-draw hazard over weeks 1–9
  + a day-90 open-balance point mass in week 13. Fit the *shape* on train (no
  censoring), tilt the *level* per cohort on val. Source: `make_survival_long` +
  `day90_open_balance`.
- **Monotonicity by construction**: cumulate non-negative weekly hazards, then add the
  day-90 mass — never decreasing, satisfies the validator.
- **Intervals** dominated by within-cohort sampling error (n≈150–220/cohort) →
  bootstrap loans within cohort, not model variance.
- A single-hazard fallback is retained but disfavored (it smears the day-90 mass).

## C — Counterfactuals (900 do() queries, 300 test applicants)
- **The registry is the causal-consistency layer.** `recompute_under_intervention(df,
  feat, value, art)` sets the raw column on a copy and recomputes exactly the
  descendants (tested: only descendants move). So `do(requested_amount=x)` correctly
  propagates to `daily_payment`, `buffer_to_payment`, `leverage_total`, the req/rev
  ratios, etc., while leaving unrelated features fixed.
- **Causal classification of the intervenables (writeup §3):**
  - *Manipulable causes* — `requested_amount` (changes the real payment burden),
    `aggregate_credit_utilization`, `existing_debt_obligations`,
    `invoice_payment_delinquency_rate`: perturb and propagate; effect is meaningful.
  - *Self-reports* — `stated_annual_revenue`, `stated_time_in_business`:
    `do()` changes a *claim*, not cash flow → near-zero true causal effect; the
    observational model's sensitivity here is the misreporting correlation, which we
    explicitly down-weight / flag.
  - *Immutable / proxy-by-fiat* — `sector`, `geography_region`, `vintage_years`,
    `has_linked_bank_feed`, `prior_loans_count`: the query set forces `do()` on these
    though they aren't manipulable causes; we answer via model perturbation and state
    plainly that these are proxy-perturbations, not policy levers.
- **Out-of-support honesty**: several queried values are outside observed support
  (`prior_loans_count` 29% in-support, `recent_inquiries_count_6mo` 45%, …) → wider C
  intervals and an extrapolation caveat.
- `intervention_queries.csv` parsed; per-feature counts and support tabulated in
  `eda_stats.json` → `intervention_support`.

## Calibration & uncertainty (scored on A and B)
- **Point PD**: isotonic (or Platt) calibration fit on the in-window val labeled set;
  remember val is approval-selected → valid on the approved manifold, extrapolated on
  declines.
- **90% intervals**: conformal / quantile-based; record empirical coverage on val.
  Interval *calibration* is explicitly scored, so we tune width to hit ~90% coverage.
- **B intervals**: within-cohort bootstrap. **C intervals**: widen for out-of-support.

## Locked Phase-1 choices
1. `prior_underwriter_score` **kept** as a feature (pre-decision in the re-underwriting
   framing; AUC 0.587 for default but the RD instrument for selection). Argument both
   ways logged here; CV + leakage canary cleared it (no univariate AUC>0.90).
2. **Reject inference deferred to Phase 2**; Phase-1 builds the scaffolding only
   (`rd_distance`, `above_rd_cutoff`, selection indicators). IPW-on-score is off the
   table (0% overlap → positivity violated); Phase 2 uses RD or covariate-overlap.
3. **No destructive cleaning / imputation of raw columns** — all handling lives in the
   engineered layer so `do()` stays well-defined on clean raw values.
