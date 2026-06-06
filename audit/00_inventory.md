# Audit 00 — Inventory (read-only)

Scoring context: **S = 0.30·S_P&L + 0.25·S_traj + 0.20·S_cal + 0.10·S_C + 0.15·S_write**.
This audit modifies nothing; it reads our artifacts/data and writes only under `audit/`.

## Validator result (verbatim)
```
Expecting 13,306 applicants, 900 queries, 13x13 trajectory grid.

ISSUES
------------------------------------------------------------------------------
  [warn ] files / missing_writeup: writeup 'submission_D_writeup.pdf' not found (include it in your real submission)
------------------------------------------------------------------------------

RESULT: PASS  (0 errors, 1 warning(s))
```
→ A/B/C are structurally valid. **D exists only as markdown, not PDF** (warning, not error; but the real submission requires the PDF).

## Deliverable-by-deliverable inventory

| Deliv. | Artifact | Rows | Produced by | Status |
|---|---|---|---|---|
| A | `submissions/submission_A_decisions.csv` | 13,306 | `src/submit.py::build` | complete: `decision`, `predicted_pd`, `pd_lower_90`, `pd_upper_90` |
| B | `submissions/submission_B_trajectory.csv` | 169 | `src/submit.py::build` | complete: 13×13 CDR grid + `cdr_lower_90/upper_90` |
| C | `submissions/submission_C_counterfactuals.csv` | 900 | `src/submit.py::build` | complete: `predicted_pd_cf` + interval |
| D | `submissions/submission_D_writeup.md` | 5 sections | hand-written | **markdown only — not exported to PDF** |

## Pipeline map (who produces what)
- `src/config.py` — paths, `SEED=20260605`, `PRODUCT` economics, column groups.
- `src/data.py` — loaders, `assign_cohort_week`, `labeled_mask`, column-group helpers.
- `src/features.py` — **feature registry** (`build_features`, `model_features` deduped view, `recompute_under_intervention`, `make_survival_long`). 62 features, 55 after dedup.
- `src/economics.py` — **exact brief NPV** (`revenue_if_full`, `npv_if_default`, `expected_npv`), LGD, break-even.
- `src/survival.py` — canonical timing shape `S(t)`, `mean_default_day`, `mean_recovery_frac`.
- `src/models.py` — `PDModel` (5-fold GroupKFold LGBM ensemble + isotonic calibration + `alpha` width scale), `train_pd_model`.
- `src/calibration.py` — A binned-coverage width scaling; B conformal width from val ground-truth trajectories.
- `src/submit.py` — orchestrates A/B/C, writes CSVs, runs validator.
- Notebooks: `01_eda.ipynb` (executed+HTML), `02_economics_and_strategy.ipynb`.
- Reports: `eda_findings.md`, `feature_catalog.md`, `design_decisions.md`, `writeup_notes.md`, `eda_stats.json`, `feature_importance_preview.csv`, `pooled_hazard.csv`.
- Tests: `tests/test_features.py` (10 tests, registry/leakage/intervention/monotonicity).

## Models trained
- **One** shared PD model: 5-fold `GroupKFold(business_id)` LightGBM **ensemble** (5 boosters), isotonic-calibrated on OOF, reported OOF AUC 0.7746 / Brier 0.1171. Plus a fold-ensemble std → 90% PD intervals (width scale `alpha` calibrated on val).
- Per the brief keystone, the **same** model feeds A (E[NPV] decision), B (`F_i(t)=PD_i·S(t)`), and C (re-predict under `do()`).

## Features engineered
55 model features (62 registry nodes − 7 deduped twins): affordability/buffer (6), discrepancy (6), informative-missingness indicators (6), credit-stress (6), platform/EB-shrink (5), volatility (3), selection/RD (3), log transforms (6), raw passthrough (15), ordinal cats (2), OOF target-encoded cats (4). Catalog: `reports/feature_catalog.md`.

## Validation already done by us
- OOF GroupKFold AUC/Brier/PR-AUC; reliability table (in-sample isotonic).
- Leakage canary (no univariate AUC > 0.90 / 0.95).
- **A interval coverage** calibrated + reported on val (binned, 94.7%).
- **B interval coverage** calibrated against true val cohort trajectories (93.5%).
- 10 unit tests (registry, OOF leakage, intervention recompute, monotonicity).

## ⚠️ What's missing / half-finished / NOT EXPOSED (matters for this audit)
1. **Trained model + OOF predictions are NOT persisted.** `submit.py` trains in-memory each run; only `feature_artifacts.pkl` (medians, τ, rd_cutoff, EB prior, target maps, seed) is saved. → Per the hard rule, **this audit will not retrain**. We evaluate **our final predictions** by recovering the val subset from `submission_A` (the 4,489 val applicants are inside the 13,306; **2,551 are labeled**) joined to val outcomes. Feature importances are read from the existing `reports/feature_importance_preview.csv` (preview LGBM, full 62-feature set — **caveat: not the exact submission ensemble**).
2. **D not exported to PDF.** Content done; format gate (PDF, ≤4 pages) unverified.
3. **No persisted per-cohort metrics, no SHAP values** saved as artifacts (SHAP referenced in writeup but not produced).
4. C queries are all on **test** applicants (no labels) → C accuracy cannot be directly validated; only the internal sanity battery (ranges, do(observed)≈base, monotonicity, channel behavior) is checkable.

## Read-only evaluation basis for Steps 1–3
- Our PD/decision/intervals on **2,551 labeled val applicants** (join `submission_A` × `validation.csv`).
- Our B grid vs **realized val CDR** (val is labeled + matured + in the 13 cohort weeks).
- Our C `p_cf` (sanity battery only; recompute `do()` effects by re-reading `submission_C`).
- Standalone diagnostics permitted in `audit/scripts/` (adversarial validation, propensity) — these are NEW classifiers that do not touch our model.
