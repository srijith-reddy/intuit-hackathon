# SMB Underwriting — NPV-Optimal Lending under Selection Bias

> Decide whom to fund, forecast how loans default over time, and answer causal what-ifs — optimizing realized portfolio **NPV**, not classification accuracy.

## What This Is

A complete underwriting pipeline for the Intuit SMB lending challenge. One calibrated discrete-time model drives an approve/decline policy (**A**), a per-cohort default-timing forecast (**B**), and interventional counterfactuals (**C**), defended in a 4-page technical writeup (**D**).

**Not a default classifier — an NPV decision engine.** We approve on the sign of expected net present value, not a PD threshold; we model *when* a loan defaults, not just *whether*; and counterfactuals answer `do(x)` (what an intervention *causes*), not `x` (what it *predicts*).

Every number below is measured on held-out validation with realized outcomes — test labels are withheld, so there are no leaderboard guesses here.

## The Challenge

A historical book of SMB loans, with three twists that break standard modelling: labels exist only for loans a prior underwriter approved (hard cutoff at `prior_underwriter_score ≥ 0.273`, **zero overlap** → positivity violated), the scoring window is forward in time (default rate drifts 17.5% → 20.6%), and self-reported fields are optimistically biased. Four deliverables — see [`dataset/README.md`](dataset/README.md) for the data guide:

- **A** — approve/decline + calibrated PD with 90% intervals, every applicant (13,306).
- **B** — the 13×13 cumulative-default trajectory grid, per origination cohort.
- **C** — ~900 `do(feature = value)` counterfactual PD queries.
- **D** — a ≤4-page methodology defense.

## Approach

- **Selection bias, stated not hidden** — a propensity model recovers approval at AUC 1.00 with zero feature overlap, so inverse-propensity reweighting is infeasible *by positivity violation, not choice*. We extrapolate locally and widen intervals on declines rather than pretend to reweight.
- **One shared timing model** — each loan's cumulative-default curve is `F_i(t) = PD_i · S_b(t)`: a calibrated *level* (GroupKFold LightGBM ensemble over 55 features, isotonic-calibrated, OOF AUC 0.774) times a band-conditional *timing shape*. The bimodal shape (missed-draw mass on days 3–60, empty 61–89, a day-90 sweep spike) is encoded empirically, not smoothed.
- **NPV-sign decisions with an uncertainty shift** — approve iff `E[NPV(PD_i + κ·σ_i)] > 0` using exact product economics. Collected daily draws are credited, so a day-5 default loses ~principal while a late/day-90 default is nearly whole — the NPV-sign bar is far more permissive than a flat 0.5 PD cut, and our approved book runs at **~13% PD** against a 17–21% population rate. `κ` (the uncertainty shift, in fold-disagreement units) is cross-fit on realized validation P&L; the curve is flat within ~1%, so it currently selects **`κ ≈ 0`** and the rule reduces to the brief's literal `E[NPV] > 0`.
- **Hierarchical cohort correction** — the PD model predates the window, so each cohort's *level* shrinks toward the realized validation rate and its *timing shape* toward the model band shape (Dirichlet `c = 50`, split-half cross-fit). The policy applies the same per-cohort level correction to PD (pseudo-count 75) only where the validation gap exceeds sampling noise. The shape term pulls sparse tail cohorts onto the data.
- **Causal counterfactuals via an explicit DAG** — `do()` interventions recompute only a feature's registry descendants (no internally-contradictory applicants). Features are typed: mechanical (full effect), confounded proxies (kept at an estimated causal fraction λ̂ from sibling-adjusted logistic, with adjustment sets **derived algorithmically from the DAG** by the backdoor criterion — descendants excluded by graph reachability, corroborated by FDR partial-correlation + PC/GES), and self-reports (≈0 interventional effect: a different number on the form changes nothing).
- **Calibrated uncertainty** — fold-ensemble disagreement → 90% intervals whose width scale is cross-fit on validation (decile coverage ≈ 0.90); trajectory bands from within-cohort bootstrap + conformal residuals.
- **Reproducible & audited** — tested `src/` modules, a 12-script audit battery (selection, stability, decision economics, causal sanity), and every writeup claim traced back to `reports/`.

## Results (held-out validation, realized outcomes)

| Deliverable | Metric | Value |
|---|---|---|
| A — PD model | out-of-fold AUC / Brier | 0.774 / 0.117 |
| A — policy | approve rate (full book / labeled-val subset) | 0.65 / 0.77 |
| A — policy | realized P&L vs. prior underwriter (labeled val) | **1.83×** |
| A — policy | capture vs. perfect-foresight oracle | 0.60 |
| A — intervals | decile coverage (cross-fit) / mean width | 0.90 / 0.06 |
| B — trajectory | mean abs. error vs. realized CDR / coverage | **0.003** / 0.93 |
| C — counterfactuals | sanity battery (self-report≈0, do(x=obs)≈0, monotone) | pass |

## Architecture

```
src/                tested pipeline modules
  data.py             loaders, labeled-mask, cohort-week assignment
  features.py         feature registry (declared parents → do() recompute)
  models.py           GroupKFold LightGBM ensemble + isotonic + intervals
  survival.py         band-conditional timing shapes & default-day distributions
  economics.py        exact brief NPV, break-even, expected-NPV decision
  calibration.py      cross-fit interval scale, κ decision-shift, shape shrinkage
  causal_graph.py     explicit DAG + backdoor adjustment-set derivation (C)
  submit.py           end-to-end build of submissions A / B / C
  explain.py          SHAP driver explanation
audit/scripts/      read-only measurement battery (selection, stability, causal)
reports/            findings, lambda_hat tables, figures (the writeup's evidence)
notebooks/          01_eda, 02_economics_and_strategy (thin, over tested modules)
submissions/        the four graded deliverables (+ archive/)
```

## Run it

```bash
make venv && make install          # uv venv (Python 3.12) + pinned deps
python -m src.submit               # build submissions A/B/C and run the validator
make test                          # pytest suite
make pdf                           # compile the Deliverable D writeup → PDF
python validate_submission.py submissions   # must print PASS
```

The DAG-derived λ̂ and the causal-discovery diagnostic regenerate via
`audit/scripts/p6_lambda_dag.py`; the DAG figure via `audit/scripts/p7_dag_figure.py`.

## Honest limitations

The PD model sits at the **information ceiling** for these features — OOF AUC 0.774 matches the Bayes-optimal ~0.776 estimated by calibrated-outcome simulation — so the ~40% gap to the perfect-foresight oracle is irreducible aleatoric risk, not a tuning deficiency; further P&L gains come from calibration and robustness, not discrimination. Realized NPV follows the brief's literal formula, crediting one daily draw per day up to the default day *including the day-90 open-balance defaults*; if the scorer instead caps draws at the 60-day term, those defaults are near-break-even rather than profitable and the policy would tighten. Declined applicants have no labels anywhere, so decline-side PD and *all* counterfactuals rest on extrapolation that cannot be validated. The κ and shrinkage knobs are tuned on validation and transfer to test only insofar as the two windows match (they share the same 13 calendar weeks); κ in particular is the argmax of a flat (~1%) curve, so we keep it near 0. The proxy causal fractions remain heuristic; latent business health breaks causal sufficiency, so PC/GES corroborate the DAG, they do not prove it.
