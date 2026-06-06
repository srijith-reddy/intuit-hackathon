# Feature Catalog

Every engineered feature, its formula, hypothesis, expected sign on **default
risk**, and group. Implemented in `src/features.py` as a **registry** (`REGISTRY`):
each feature declares its `parents`, so intervening on a raw column (Deliverable C)
recomputes exactly its descendants. Fit-on-train artifacts (medians, τ, RD cutoff,
target maps, Beta prior) are frozen and applied to val/test — no leakage.

- **62 features** across 11 groups. OOF GroupKFold(business_id) LGBM: **AUC 0.774,
  PR-AUC 0.490, Brier 0.117** (base rate 0.174). Leakage canary: no univariate
  AUC > 0.90. Importances in `reports/feature_importance_preview.csv`.
- Sign legend: **+** risk-increasing, **−** risk-decreasing, **?** ambiguous.
- All ratios are null-safe (`_safe_div`) and clipped to documented bounds; recency
  uses `exp(-days/τ)` with null→0 (event never happened).

## Family 1 — Affordability / cash-flow coverage (buffer-centric)
EDA reframing: loans are ~1.3% of annual revenue, so default is **cash-buffer-bound,
not revenue-bound**. Daily ACH draw ≈ principal × (1+interest)/60 ≈ 0.0176 × principal.

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `daily_payment` | `requested_amount × (1+INT_TERM)/60` | the actual daily ACH burden; parent of all coverage ratios | ? |
| `buffer_to_payment` | `observed_cash_balance_p10 / daily_payment` (clip −50,200) | **key**: days of buffer vs the draw; thin/negative buffer → default | − |
| `dscr_daily` | `(obs_monthly_rev/30) / daily_payment` | daily revenue coverage of the draw | − |
| `dscr_daily_stated` | `(stated_annual/365) / daily_payment` | fallback coverage when no feed | − |
| `payment_to_revenue` | `30·daily_payment / obs_monthly_rev` (clip 0,5) | payment as share of revenue | + |
| `leverage_total` | `(existing_debt + requested) / annual_rev` | post-loan leverage (observed rev, stated fallback) | + |

## Family 2 — Discrepancy / misreporting
EDA: overstating revenue >1.5× → 38.6% default vs 16.1%. The *gap* to observed is the
signal, not the stated level (key for the causal story: `do(stated_revenue)`≈0 effect).

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `rev_log_ratio` | `log( (stated_annual/12) / obs_monthly )` (clip) | overstatement magnitude | + |
| `revenue_overstated` | `1[ stated/12 > 1.5 × obs_monthly ]` | sharp misreport flag (2.4× lift) | + |
| `tib_gap` | `stated_time_in_business − vintage_years` | tenure inflation | + |
| `tib_inflated` | `1[ tib_gap > 0.5 ]` | discipline proxy | + |
| `req_to_obs_annual` | `requested / (obs_monthly×12)` | clean recompute of the leaky provided ratio (observed) | + |
| `req_to_stated_annual` | `requested / stated_annual` | stated-side ask intensity | + |

## Family 3 — Informative missingness (first-class)
EDA: null = signal, stable across splits. never-declined null→0.143 vs present 0.213;
no-feed→0.191 vs 0.167.

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `no_bank_feed` | `1[ ¬has_linked_bank_feed ]` | opacity self-selection | + |
| `never_declined_external` | `1[ days_since_last_external_decline is null ]` | never declined elsewhere = good | − |
| `no_inquiry_elsewhere` | `1[ days_since_last_inquiry_elsewhere is null ]` | no shopping = good | − |
| `is_first_time_borrower` | `1[ prior_loans_count = 0 ]` | no platform history | ? |
| `prior_declined_elsewhere_flag` | `1[ prior_approved_amount is null ]` | prior-lender decline (selection proxy) | + |
| `no_feed_x_amount` | `no_bank_feed × z(requested_amount)` | large unverified ask | + |

## Family 4 — Credit-stress composites

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `util_x_inquiries` | `aggregate_credit_utilization × recent_inquiries_6mo` | stretched + shopping | + |
| `debt_to_revenue` | `existing_debt / annual_rev` | pre-existing leverage | + |
| `overdrafts_per_month` | `observed_overdraft_count_3mo / 3` | cash-management failures | + |
| `inquiry_velocity` | `multi_lender_inquiry_30d / (recent_inquiries_6mo+1)` | recent burst vs baseline | + |
| `decline_recency` | `exp(−days_since_last_external_decline/τ)` | recent decline = fresh adverse signal | + |
| `inquiry_recency` | `exp(−days_since_last_inquiry_elsewhere/τ)` | recent shopping | + |

## Family 5 — Platform-relationship (empirical-Bayes shrinkage)
Only 32% have history → shrink toward the global default rate with a Beta(αβ) prior
(pseudo-count 5; α=base·5, β=(1−base)·5).

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `prior_default_rate_shrunk` | `(prior_defaults + α)/(prior_loans + α + β)` | own track record, shrunk | + |
| `avg_prior_loan_size` | `prior_loans_amount_total / prior_loans_count` | prior scale | ? |
| `prior_size_vs_request` | `avg_prior_loan_size / requested_amount` | ask vs precedent | ? |
| `engagement_intensity` | `platform_active_months / (account_age_days/30)` | active-use ratio | − |
| `bookkeeping_recency_decay` | `exp(−bookkeeping_recency_days/τ)` | recent bookkeeping = engaged | − |

## Family 6 — Volatility / stability

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `volatility_level` | `observed_revenue_volatility` | raw revenue instability | + |
| `rev_vol_x_negtrend` | `volatility × (−trend)` | volatile **and** declining | + |
| `payroll_regularity` | `payroll_regularity_score` | operational discipline | − |

## Family 9 — Selection / regression-discontinuity
The prior policy is `approve ⇔ prior_underwriter_score ≥ 0.273` (sharp, 0% overlap).

| feature | formula | hypothesis | sign |
|---|---|---|---|
| `prior_score` | `prior_underwriter_score` | prior model's risk read | − |
| `rd_distance` | `prior_underwriter_score − 0.273` | distance from the approval cutoff | − |
| `above_rd_cutoff` | `1[ score ≥ 0.273 ]` | would the prior lender approve | − |

## Transforms & passthroughs
- `log_*` (6): `sign(x)·log1p(|x|)` on monetary cols (`requested_amount`,
  `stated_annual_revenue`, `observed_monthly_revenue_avg_3mo`, `existing_debt_obligations`,
  `observed_cash_balance_p10`, `prior_loans_amount_total`) — for the linear baseline.
- `raw_*` (15): clean numeric signals passed through unchanged for the tree model.

## Family 8 — Categorical encoding
- **Nominal** (`sector`, `geography_region`, `intended_use_of_funds`, `application_channel`):
  `te_*` = OOF target encoding, GroupKFold(business_id), smoothing m=20 toward the
  global mean. Train rows get out-of-fold values; val/test/intervention get the frozen
  full-fit map. (One-hot kept as a CV-time alternative.)
- **Ordinal** (`owner_personal_credit_band`, `employee_count_bucket`): `ord_*` =
  ordered integer passthrough (dict says ordering is meaningful; credit_band is
  monotone 0.287→0.102 in EDA).

## Family 10 — Survival long-format (Deliverable B) — `make_survival_long`
One row per (labeled loan, loan-age week 1..13) with a discrete-time `event`. No
censoring (all matured). Emits `day90_open_balance = 1[default ∧ days_to_default ≥ 89]`
so B is modeled as **two components**: missed-draw hazard (weeks 1–9) + day-90 mass
(week 13). Validated: total events = 9,024 = n defaults; 2,027 events in week 13.
