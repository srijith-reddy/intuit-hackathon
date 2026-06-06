# Audit 01 — Temporal Stability (Home Credit playbook)

Lesson: models that fluctuate across time slices bleed points where scoring is
per-period. Our 13 cohort weeks are that structure (B = 25%, and per-cohort
calibration feeds the 20%). Evaluated on **our final `submission_A` PD** over the
2,551 labeled val applicants and **our `submission_B` grid** vs realized val CDR.
Reproduce: `python audit/scripts/s1_stability.py`. (No retraining; the only model
trained is an independent drift classifier in the audit script.)

Overall val: AUC **0.755**, Brier **0.135** (vs our reported OOF AUC 0.774 — the
train→val drop is itself a drift signal).

## 1.1 Per-cohort-week stability — **WEAK**
| metric | value |
|---|---|
| AUC mean / min / max | 0.757 / **0.711 (wk11)** / 0.826 (wk13) |
| **AUC spread** | **0.116** |
| Brier mean / worst | 0.134 / 0.168 (wk5) |
| \|calibration gap\| mean / worst | 0.026 / **0.062 (wk5)** |

- AUC swings **0.116** across cohorts — the exact fluctuation Home Credit penalizes.
- Calibration gaps are **mostly negative** (we under-predict): cohort 5 (highest
  default rate 0.26) is under-predicted by 0.062; cohort 13 (lowest, 0.15) is
  over-predicted by 0.028. Our PD doesn't track the per-cohort level — it sits near
  the pooled rate and misses the high/low cohorts. This directly hurts B and S_cal.
- **Verdict: WEAK** — meaningful per-cohort instability, concentrated at the extreme
  cohorts (5 high, 13 low).

## 1.2 Drift / are we leaning on unstable features? — **WEAK**
Independent adversarial validation (standalone LGBM, 3-fold AUC):
| split | adversarial AUC |
|---|---|
| train-early vs train-late | **0.904** |
| train vs validation | **0.949** |

- Both are far above 0.5 → **strong temporal drift** within train and into val
  (expected for a forward split, but large).
- **Our model leans on drifting features.** Of our top-12 importance features, these
  also rank as top drift drivers (train→val): **`raw_invoice_payment_delinquency_rate`
  (our #1 feature), `raw_observed_revenue_trend_3mo`, `rev_vol_x_negtrend`.** Revenue
  trend/volatility and invoice delinquency shift across time and we weight them
  heavily → a mechanism for the per-cohort AUC swing in 1.1.
- Selection features (`above_rd_cutoff`, `prior_score`) drift hardest train→val
  because val has a different approval mix — fine as signal but worth noting.
- **Verdict: WEAK** — no leakage, but stability of the top features is not managed
  (no time-aware CV weighting, no drift-robust feature selection).

## 1.3 Deliverable B grid — monotonicity **PASS**, accuracy **WEAK**
- **Monotone in age for every cohort: PASS** (asserted numerically over all 13).
- Vs realized CDR on our approved val set (169 cells): **MAE 0.016, RMSE 0.022**,
  mean signed error **+0.008** (slight systematic over-prediction).
- **Worst cells are the extreme cohorts:** cohort 13 ages 8–12 predicted 0.117 vs
  realized **0.051** (+0.066 over); cohort 5 age 13 predicted 0.156 vs realized
  **0.212** (−0.056 under). Same cohorts as the 1.1 calibration misses → the B error
  is inherited from per-cohort PD level error **and** the pooled-shape issue (1.4).
- **Verdict: PASS on monotonicity, WEAK on accuracy at the tails.**

## 1.4 Does our timing model distinguish early vs late defaults? — **WEAK**
- Our model is `F_i(t) = PD_i · S(t)` with a **single pooled shape `S(t)`**. Every
  loan shares one timing curve, scaled only by its PD level. **It cannot say loan X
  defaults earlier than loan Y** — a day-5 and a day-55 default have identical
  per-loan shape, and A's E[NPV] uses a pooled `t_bar = 43` for all loans. This is
  precisely the distinction the brief (p.9) says matters for both B and NPV.
- **And the shape genuinely varies by risk segment** (model-free, on train defaults):
  | credit band | median days-to-default | day>60 share |
  |---|---|---|
  | 0 (worst credit) | 33 | 0.155 |
  | 2 | 37 | 0.216 |
  | 4 (best credit) | 46 | 0.339 |

  Worse-credit borrowers default **~13 days earlier** and carry far less day-90 mass.
  The pooled shape therefore **mis-times per-segment**: it over-states early CDR for
  good-credit cohorts (who default late) and under-states it for bad-credit cohorts —
  the mechanism behind the cohort-13 over-prediction in 1.3.
- **Censoring vs competing risk is moot here:** there is no right-censoring (all loans
  matured) and all repayments occur at exactly day 60, so the realized cumulative
  default fraction by day `t` is observed exactly. We computed the realized grid
  directly from outcomes; neither a censoring nor a competing-risk treatment would
  change that ground truth. So this sub-question is **N/A by data**, and our grid's
  error is a shape/level problem, not a censoring problem.
- **Verdict: WEAK** — pooled shape is a real limitation with measurable, signed impact
  on B (25%) and a smaller one on A's E[NPV]; a PD- or band-conditional shape would fix it.

## Headline verdicts
| Item | Verdict |
|---|---|
| 1.1 per-cohort stability | **WEAK** (AUC spread 0.116; under-predicts high cohorts) |
| 1.2 drifting features | **WEAK** (top features incl. #1 drift train→val) |
| 1.3a B monotonicity | **PASS** |
| 1.3b B accuracy vs realized | **WEAK** (MAE 0.016; worst ±0.06 at tail cohorts) |
| 1.4 early/late timing | **WEAK** (pooled shape; shape varies ~13d by credit band) |
| 1.4 censoring vs competing risk | **N/A** (no censoring in data; ground truth unaffected) |
