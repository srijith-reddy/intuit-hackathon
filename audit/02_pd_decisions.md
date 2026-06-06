# Audit 02 — Missingness, Reject Inference, Calibration, Decisions (AmEx playbook)

Lesson: missingness is signal; curated features + calibrated, well-understood models
win; and label selection must be addressed. Evaluated on **our final predictions /
decisions** over 2,551 labeled val applicants + realized NPV from true val outcomes.
Reproduce: `python audit/scripts/s2_pd_decisions.py`.

## 2.1 Missingness handling — **PASS** (one caveat)
We **pass NaN natively to LightGBM** and add explicit indicators (`no_bank_feed`,
`never_declined_external`, `no_inquiry_elsewhere`, `is_first_time_borrower`,
`prior_declined_elsewhere_flag`). Default rate by pattern on **val**:

| pattern | n | default |
|---|---|---|
| bank feed present / absent | 1696 / 855 | 0.206 / **0.207** |
| ext-decline null (good) / present | 1391 / 1160 | 0.188 / 0.228 |
| inquiry null (good) / present | 1363 / 1188 | 0.176 / 0.241 |
| first-time / repeat | 1611 / 940 | 0.218 / 0.186 |

- Decline/inquiry/first-time signals persist on val and **we encode them** → good.
- **Caveat (drift):** the bank-feed signal **washed out on val** (0.206 vs 0.207; it
  was 0.167 vs 0.191 on train). We still carry `no_bank_feed` and its interaction —
  not harmful, but it's a stale-on-val feature (ties to the 01 drift finding).
- **Verdict: PASS.**

## 2.2 Reject inference — **WEAK / NOT DONE** (but partly unavoidable)
We **deferred** reject inference (selection features only: `rd_distance`,
`above_rd_cutoff`; no IPW/reweighting). Two measured facts:

- **Overlap is essentially zero.** A propensity model `P(prior-approve | X)` trained
  **without** any selection features still gets **OOF AUC 1.000** — approval is a
  near-deterministic function of the ordinary covariates, so approved vs declined
  populations don't overlap in feature space. **IPW/reweighting is genuinely
  infeasible** (positivity violated on covariates, not just on the score). The
  deferral is therefore partly justified — but the resulting bias is unaddressed.
- **Calibration degrades toward the decline boundary.** Among labeled (approved) val,
  split by approval propensity:
  | tertile | n | pred PD | obs | gap |
  |---|---|---|---|---|
  | high (approve-like) | 847 | 0.120 | 0.129 | −0.008 |
  | mid | 847 | 0.218 | 0.242 | −0.024 |
  | low (decline-like) | 857 | 0.216 | 0.247 | **−0.032** |

  As we move toward the decline region the PD increasingly **under-predicts**.
  Extrapolating to *true* declines (which have no labels anywhere) the under-prediction
  plausibly worsens — and that is the population A is scored on. **Verdict: WEAK** —
  real, measurable bias toward the scored population, uncorrected (though hard to fix).

## 2.3 Calibration (20% of score) — **WEAK** (good points, over-wide intervals + level bias)
- **Pooled ECE (10-bin) = 0.0245** — decent point calibration.
- **Interval coverage is 100%, not 90%** — over-covering ⇒ **needlessly wide**:
  - per-decile: empirical rate inside mean interval for **10/10** deciles; **mean
    width 0.152** (up to 0.34 in the top decile).
  - per-cohort: **13/13** cohorts covered.
  The width calibration (targeted 90%) overshot to ~100% real coverage → we likely
  leave S_cal points on the table to "needless width," the exact AmEx trade-off.
- **Point PD under-predicts in the mid-high deciles** (decile 6: pred 0.147 vs obs
  0.210; top decile 0.571 vs 0.626) — same negative bias as 2.2, ties to the train→val
  drift (trained 17.5%, val 20.6%).
- **Verdict: WEAK** — calibration is *safe* (always covers) but **over-wide and
  level-biased low**; both directions cost S_cal, and the level bias also costs S_P&L (below).

## 2.4 Decision economics — **rule PASS / robustness PASS / P&L-capture WEAK**
Decision rule reconstructed from `submit.py`: **`d = 1[E[NPV|approve] > 0]`** with the
exact brief NPV — correct, not a PD threshold. Realized NPV on labeled val (true
outcomes):

| policy | approve rate | realized P&L |
|---|---|---|
| **ours** | 0.882 | **$3.69M** |
| approve-all = prior underwriter | 1.000 | $2.14M |
| oracle (true NPV>0) | 0.853 | $6.55M |

- **We beat the prior underwriter / approve-all by 1.73×** ($3.69M vs $2.14M) — the
  NPV rule genuinely adds value by declining the worst loans. **Strong.**
- **But we capture only 0.564 of oracle**, and we **approve *more* than oracle**
  (0.882 vs 0.853) → we fund negative-NPV loans the oracle rejects. The cause is the
  **PD under-prediction** (2.2/2.3): low PD → optimistic E[NPV] → over-approval.
- **Stress test (decisions are robust):**
  | shock | flips | realized P&L |
  |---|---|---|
  | PD +20% | 133 (5.2%) | **$3.78M** |
  | PD −20% | 128 (5.0%) | $3.62M |
  | recovery +50% | 128 (5.0%) | $3.62M |
  | recovery −50% | 133 (5.2%) | $3.78M |

  Only ~5% of decisions flip under ±20% PD / ±50% recovery → **robust**. Notably,
  **PD +20% *increases* realized P&L** ($3.78M) — confirming we are **slightly too
  aggressive**; nudging the PD level up (correcting the negative bias) would tighten
  the approve set toward oracle and **lift S_P&L**.
- **Verdict: rule PASS, robustness PASS, P&L-capture WEAK** (under-prediction ⇒
  over-approval ⇒ ~44% of oracle P&L unrealized; correcting the level bias helps).

## Headline verdicts
| Item | Verdict |
|---|---|
| 2.1 missingness handling | **PASS** (feed signal stale on val — minor) |
| 2.2 reject inference | **WEAK/NOT DONE** (uncorrected bias −0.032 toward declines; IPW infeasible) |
| 2.3 calibration | **WEAK** (ECE 0.025 ok, but coverage ~100% ⇒ over-wide; PD biased low) |
| 2.4 decision rule | **PASS** (correct NPV rule; 1.73× prior underwriter; robust) |
| 2.4 P&L capture | **WEAK** (0.564 of oracle; over-approves; fix = correct PD level) |

## Cross-cutting insight (feeds Step 4)
A single root cause — **PD under-predicts (level bias low, ~−0.025 pooled, −0.032 near
declines)** — leaks from **two** scored components at once: S_cal (level + over-wide
intervals) and S_P&L (over-approval, 0.564 oracle capture). A level recalibration is
the highest-leverage single fix.
