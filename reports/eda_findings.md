# EDA Findings — SMB Underwriting Challenge

All numbers reproduced by `python -m src.eda` (dumps `reports/eda_stats.json`,
figures in `reports/figures/`, pooled hazard table `reports/pooled_hazard.csv`).
Seed `20260605`. Organized around the five structural issues that decide this
challenge, then the boring-but-necessary checks, then deliverable implications.

> **Reading convention:** "labeled" = a repayment outcome is observed, which
> happens **iff** the prior lender approved the loan **and** it matured. Every
> default rate below is conditional on that selection unless stated otherwise.

---

## 0. Dataset shape & the split that frames everything

| split | rows | labeled | labeled % | default rate (labeled) | timestamp span | cohort coverage |
|---|---|---|---|---|---|---|
| train | 85,340 | 51,722 | **60.6%** | **17.45%** | 2024-01-01 → 2025-06-29 | 0% |
| validation | 4,489 | 2,551 | **56.8%** | **20.62%** | 2025-06-30 → 2025-09-28 | 100% |
| test | 8,817 | 0 | 0% | — (withheld) | 2025-06-30 → 2025-09-28 | 100% |

**This is a forward-in-time split.** Train is an 18-month historical book that
ends the day before the cohort window opens; val and test are the *same* 13-week
window (2025-06-30 → 2025-09-28), differing only in that val keeps labels for
prior-approved loans and test withholds everything. Cohort weeks 1–13 (Deliverable
B) exist **only** in val+test — train has zero cohort coverage.

Consequences:
- **Temporal drift is real, not noise.** Default rate rises 17.45% (train) →
  20.62% (val) across an 18-month gap. Deliverable B must model the *scoring
  quarter*, so train gives the default-*timing shape* and val gives the per-cohort
  *level*. (Per-cohort val rates below.)
- Standard i.i.d. CV under-states test error; CV folds should respect the time
  ordering where possible, and we calibrate on val (in-window).

---

## 1.1 Selection bias / reject inference — **the central issue**

**The label mechanism is a sharp, deterministic threshold on `prior_underwriter_score`.**

| | approved (prior_decision=1) | declined (prior_decision=0) |
|---|---|---|
| score range | **[0.273, 1.000]** | **[0.000, 0.273]** |
| score mean | 0.781 | 0.073 |
| distribution overlap | **0.0%** | — |
| AUC(score → approval) | **1.000** | — |

The prior policy is, to numerical precision, **approve ⇔ `prior_underwriter_score`
≥ 0.273** (fig01). This is the single most important structural fact:

1. **A propensity model `P(approved | score)` is degenerate** — perfect separation,
   zero overlap → the positivity/overlap assumption for IPW reject inference *on the
   score* is violated outright. Inverse-propensity weighting on the prior score is
   impossible. Reject inference must lean on the *other* covariates (which overlap)
   or on the **regression-discontinuity** structure at 0.273.
2. **It is a clean sharp-RD design.** Just above the cutoff we observe outcomes;
   just below we don't. Local comparability near 0.273 lets us (a) bound selection
   bias and (b) sanity-check causal claims later. We log this as a Deliverable-D /
   reject-inference asset.
3. `prior_approved_amount` is null **iff** declined → it is a perfect proxy for the
   label-selection indicator. Treat with care (see leakage, §1.5).

**Approved applicants are systematically lower-risk on observables** (top |SMD|,
prior-approved vs prior-declined; full table in `eda_stats.json`):

| feature | approved mean | declined mean | SMD | KS |
|---|---|---|---|---|
| aggregate_credit_utilization | 0.439 | 0.597 | −0.71 | 0.275 |
| invoice_payment_delinquency_rate | 0.178 | 0.273 | −0.68 | 0.277 |
| observed_cash_balance_p10 | **+1,165** | **−757** | +0.67 | 0.265 |
| owner_personal_credit_band | 2.31 | 1.50 | +0.60 | 0.240 |
| requested_amount | 23,724 | 26,768 | −0.51 | 0.198 |
| payroll_regularity_score | 0.540 | 0.468 | +0.38 | 0.147 |
| observed_revenue_volatility | 1.058 | 1.263 | −0.32 | 0.142 |
| recent_inquiries_count_6mo | 0.539 | 0.694 | −0.19 | 0.085 |

Every feature shifts in the credit-sensible direction. The declined population is a
**different, riskier distribution** — naïvely training on approved-only and scoring
the full applicant pool is extrapolation, and our PD on declines is the least
trustworthy part of Deliverable A. **Validation is selection-conditioned too**
(only 56.8% labeled), so even our calibration set can't directly observe decline
outcomes — honest offline policy evaluation is confined to the approved subpopulation.

---

## 1.2 Censoring & timing structure (drives Deliverable B)

- **No within-sample censoring.** `observation_status` ∈ {`matured`} only;
  `repayment_status` ∈ {`paid_in_full`, `defaulted`} only — no `open`. Every
  approved loan's full 90-day outcome is known. The *only* "censoring" is the
  policy selection itself (§1.1). This simplifies B: a discrete-time hazard needs
  no competing-risk censoring term, just defaults + payoffs.
- **Payoffs are degenerate:** `days_to_full_repayment` is **always exactly 60** for
  paid loans (min=max=mean=60). Non-defaulters all clear at the term end; nobody
  prepays. So the risk set stays full (minus defaults) until day 60.
- **`days_to_default` ∈ [3, 90]**, mean 43.1, median 37. Hazard turns on at day 3
  (no day-1/2 defaults — consistent with the "3 consecutive misses" rule needing ≥3
  days).
- **⚠️ Default timing is bimodal — two distinct default modes (fig02):**

  | day band | defaults | share of defaulters | mode |
  |---|---|---|---|
  | 1–2 | 0 | 0.0% | (impossible: needs ≥3 missed draws) |
  | 3–60 | 6,997 | **77.5%** | missed-draw defaults, ~flat hazard 0.0024–0.0033/day |
  | 61–89 | **0** | **0.0%** | (dead zone — nothing defaults here) |
  | 90 | 2,027 | **22.5%** | **balance>0-at-day-90 cliff (single point mass)** |

  All non-defaulters pay in full at **exactly day 60** (payoffs: 42,698 at day 60,
  zero elsewhere). So after day 60 the book is frozen — survivors have paid, and the
  remaining open loans don't resolve until the day-90 sweep declares them defaulted.
- **Pooled cumulative trajectory by loan-age week** (the canonical B *shape*,
  `pooled_hazard.csv`, fig03):

  | wk | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | cum default | .013 | .033 | .050 | .068 | .085 | .100 | .114 | .128 | .135 | .135 | .135 | .135 | **.175** |

  → **Concave rise weeks 1–9, dead flat weeks 9–12, then a +3.9pp vertical jump at
  week 13.** This is monotone by construction (good for B's constraint) but a **trap
  for any smooth parametric survival fit**, which would smear the day-90 mass across
  weeks 10–13 and miss both the flat stretch and the cliff. **Model B as two
  components: a missed-draw discrete hazard (days 3–60 → weeks 1–9) plus a day-90
  point mass** (the open-balance sweep), each possibly with its own drivers. Level
  tilts per cohort via val.

**Per-cohort default rate (val, labeled)** — noisy (~150–220 loans/cohort), centered
near 0.20, all above the train level:

| cohort | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate | .208 | .232 | .196 | .168 | **.260** | .201 | .190 | .225 | .190 | .209 | .198 | .240 | **.153** |

Small per-cohort n ⇒ wide sampling error ⇒ the 90% intervals in B will be driven by
bootstrap-over-loans-within-cohort, not by model uncertainty. (fig04)

---

## 1.3 Informative missingness (missingness *is* a feature)

Null patterns are stable across splits (no missingness drift):

| column | null% train | null% val | null% test | meaning of null |
|---|---|---|---|---|
| days_since_last_external_decline | 50.2% | 49.7% | 49.0% | **never declined elsewhere (good)** |
| days_since_last_inquiry_elsewhere | 49.9% | 48.9% | 49.0% | **no prior inquiry (good)** |
| prior_approved_amount | 39.4% | 43.2% | 43.7% | declined by prior lender (= selection flag) |
| bank-feed block (7 cols) | 35.7% | 36.6% | 36.7% | no linked feed |

**Null carries strong signal — and in the "good" direction:**

| split | default rate |
|---|---|
| baseline (labeled) | 0.1745 |
| never declined elsewhere (null) | **0.1428** |
| was declined elsewhere (present) | **0.2128** |
| no prior inquiry (null) | 0.1415 |
| had prior inquiry (present) | 0.2139 |
| has bank feed | 0.1665 |
| no bank feed | 0.1907 |

→ Missingness indicators must be **first-class features** (the dict's null semantics
are credit information, not absence). Imputing them away destroys signal. Note the
direction: null = *lower* risk for the decline/inquiry columns (absence of a bad
event), but no-feed = *higher* risk (self-selection out of transparency).

---

## 1.4 Self-reported vs observed discrepancies (misreporting signal)

- On feed-linked rows, **stated/12 vs observed monthly revenue** is well-calibrated
  on average (median ratio 0.949, p10 0.718, p90 1.269; fig06) — most applicants
  are roughly honest, with a slight *under*statement bias at the median.
- **But the tail is where the signal lives:** the 2.4% who **overstate revenue >1.5×**
  default at **38.6%** vs **16.1%** for everyone else — a 2.4× lift. Overstatement is
  a sharp fraud/misreporting flag.
- **Time-in-business inflation:** 32.9% state a longer tenure than `vintage_years`
  (median gap +0.27 yr). A `stated > observed` indicator is a cheap discipline proxy.
- **⚠️ The provided `requested_amount_to_observed_revenue` is mislabeled and leaky.**
  Decoded exactly: it equals `requested_amount / (observed_monthly_revenue_avg_3mo ×
  12)` — requested ÷ observed **annual** revenue (not monthly) — **and silently
  swaps to `requested_amount / stated_annual_revenue` when no feed** (never null
  even without a feed). So one column means two different things across rows and
  quietly encodes the observed-vs-stated discrepancy. **We will recompute
  affordability ratios ourselves** with an explicit `denominator_is_observed` flag,
  and not reuse the provided column verbatim.

**Scale reframing (matters for the affordability feature family):** observed monthly
revenue median ≈ \$156.8k (annual ≈ \$1.88M), loans ≈ \$24k → a loan is ~1.3% of
annual revenue. **Default is not revenue-affordability-bound.** The binding
constraint is the **cash buffer**: daily ACH ≈ \$420 (= req × (1+0.03+0.0575)/60)
vs `observed_cash_balance_p10` ≈ \$1,165 for approved (negative 28.6% of the time
overall). Buffer-to-payment and reliability (overdrafts, delinquency) should
dominate revenue-coverage ratios.

---

## 1.5 Entity & leakage structure

- **`business_id` never spans splits** (overlap train/val/test = 0/0/0 — dict claim
  verified). Within train, 68,364 businesses; 14,425 have multiple applications
  (max 4/business). Repeats exist **within** train ⇒ use **GroupKFold by
  `business_id`** for honest OOF (belt-and-suspenders even though splits don't overlap).
- 67.6% of train are first-time borrowers (`prior_loans_count = 0`); max prior loans
  = 6, max `repeat_application_count` = 3. The platform-relationship features (§FE 5)
  only fire for the 32.4% with history → shrinkage toward the global mean is needed.
- **Leakage canary (single-feature AUC for default, labeled rows): clean.** Highest
  are `invoice_payment_delinquency_rate` 0.751 and `aggregate_credit_utilization`
  0.735 — strong but plausible credit signals, nothing near the >0.95 "this is the
  label in disguise" zone. No target leakage detected.
- **`prior_underwriter_score` alone is a weak default predictor (AUC 0.587)** within
  the approved population — partly range restriction (all approved scores ≥0.273).
  So the prior score is *highly* informative about *selection* (AUC 1.0) but only
  mildly about *default among the approved*. This supports keeping it as a feature
  (it is pre-decision in our re-underwriting framing) while noting it is mostly a
  selection instrument; the for/against argument is logged in `design_decisions.md`.
- **Class balance** (default rate, labeled):
  - `owner_personal_credit_band`: monotone **0.287 → 0.227 → 0.184 → 0.144 → 0.102**
    (band 0→4) — strong, ordinal, treat as ordered.
  - `sector`: 0.111 (sector 3) … 0.194 (sector 0) — moderate.
  - `application_channel`: 0.171 / 0.174 / 0.189 — mild.
  - `geography_region`: 0.171–0.179 — essentially flat (weak feature).

---

## 2. Boring-but-necessary

- **No constant columns, no duplicate rows** (excluding `applicant_id`).
- **Dtype vs dictionary:** only mismatch is `default_flag` read as float64 (it holds
  NaNs for unlabeled rows) — expected, not an error. **`prior_decision` is coded
  `{0,1}`, not the `"approved"/"declined"` strings the dictionary describes** — flag.
- **Outliers / ranges:** `observed_cash_balance_p10` < 0 for **28.6%** (real
  overdraft signal, keep). `requested_amount` ∈ [\$7,590, \$45,903], **0% outside**
  the stated \$5k–\$50k band. `existing_debt_obligations` p99 \$27k, max \$69k
  (long right tail → log transform).
- **Categorical cardinality is low** (sector 5, geography 4, employee_bucket 4,
  intended_use 4, credit_band 5, channel 3) ⇒ one-hot is cheap and interpretable;
  target encoding is optional, not required, and we'll let CV decide.

### 2.1 Data-integrity verdict — no cleaning pass needed (`integrity_checks`)

Synthetic, internally consistent data: **0 total integrity violations** across every
check below, and **0** unseen category codes in val/test.

| check | result |
|---|---|
| label cross-field (`default_flag` ⇔ `repayment_status` ⇔ `days_to_default` ⇔ recovery-only-on-default) | 0 violations |
| `days_to_default` ∈ [1,90] | 0 out of range |
| `prior_loans_default_count ≤ prior_loans_count` | 0 violations |
| `prior_approved_amount` present ⇔ approved; approved amount ≤ requested | 0 violations |
| observed-feed cols present ⇔ `has_linked_bank_feed` | 0 violations |
| bounded features ∈ [0,1]; no negative revenue/tenure/vintage/debt | 0 violations |
| unseen category codes in val/test | 0 |
| duplicate rows (exact, and business+amount+timestamp) | 0 |
| train↔test null-rate drift | max 4.3pp (`prior_approved_amount`), else <1.2pp |

**So classical cleaning is unnecessary — and the "messy-looking" features are signal,
not dirt:** negative `observed_cash_balance_p10` (28.6%) is overdraft risk (keep),
35–50% nulls are structural MNAR (encode as indicators), fat monetary tails are real
(log-transform, don't clip), and the leaky `requested_amount_to_observed_revenue` is
recomputed not patched. **All such handling lives in `features.py` (fit-on-train,
registry-tracked) — we never mutate the raw intervenable columns**, so Deliverable C's
`do()` interventions stay well-defined.

---

## 3. Deliverable C — intervention-query support (prep)

900 queries over **300 test applicants** (~3 each); all 300 are in `test` (none in
train/val) → **no labels for any C target**, pure model extrapolation.

- **14 queried features are marked `intervenable=False` in the dictionary** — incl.
  `sector`, `geography_region`, `vintage_years`, `employee_count_bucket`,
  `has_linked_bank_feed`, `prior_loans_count/_default_count/_amount_total`,
  `account_age_days`, `days_since_last_*`. We must still answer `do()` for these even
  though several are not plausibly manipulable causes. The writeup §3 must classify
  each as *manipulable cause* vs *proxy/immutable* and treat the latter honestly.
- **Several queried values sit outside observed support** (frac of query values in
  test's 1–99 percentile band): `prior_loans_count` **0.29**,
  `recent_inquiries_count_6mo` 0.45, `platform_active_months` 0.58,
  `prior_loans_default_count` 0.62, `observed_overdraft_count_3mo` 0.64,
  `multi_lender_inquiry_count_30d` 0.65, `observed_revenue_trend_3mo` 0.72. These
  out-of-support `do()` values demand wider C intervals and explicit extrapolation
  caveats.

---

## 4. Implications for feature engineering & the deliverables

1. **Affordability** must be **buffer-/reliability-centric**, not revenue-centric:
   buffer-to-daily-payment, overdrafts, delinquency, payroll regularity. Recompute
   the requested-to-revenue ratio cleanly (observed vs stated, with a flag); do not
   reuse the leaky provided column.
2. **Missingness indicators are features** (never-declined, no-inquiry, no-feed,
   first-time) with the directions above; add no-feed × requested-amount interaction.
3. **Discrepancy features** (revenue overstatement >1.5×, stated>vintage tenure) are
   high-lift and cheap.
4. **Selection / RD features:** distance of `prior_underwriter_score` from 0.273,
   score percentile — predictors now, reject-inference scaffolding for Phase 2.
   (Propensity-IPW on the score alone is *off the table* — no overlap.)
5. **Survival long-format** for B is clean (no censoring); learn hazard shape on
   train, calibrate level per cohort on val, enforce monotonicity by cumulating
   non-negative hazards, bootstrap loans-within-cohort for intervals.
6. **CV = GroupKFold(business_id)**; transparent Ridge/Elastic baseline beside LGBM;
   SHAP for the regulator-facing driver story; everything seeded & clipped.

## Open flags (data contradicts docs — trusting data)
- `prior_decision` is `{0,1}` not strings.
- `requested_amount_to_observed_revenue` is requested ÷ observed **annual** revenue
  with a silent stated-revenue fallback (name says "monthly", and it's not feed-gated).
- `intervention_queries.csv` intervenes on 14 features the dictionary marks
  `intervenable=False`.
- README calls validation "applications with outcomes filled in," but only 56.8%
  (the prior-approved) actually have outcomes.
