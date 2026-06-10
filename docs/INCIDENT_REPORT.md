# Incident report - v0 training run

**Date:** 2026-05-30 → 2026-05-31
**Project version:** 0.1.0
**Author:** Felipe Toro
**Status:** Resolved (model artifact recovered via salvage script)

## Summary

The first end-to-end Optuna sweep on the IBM AML HI-Small dataset (5.08M transactions, 3.55M in the training fold) completed sweeps for XGBoost, LightGBM, and Random Forest in ~17 hours of CPU wall-clock on an Apple M1 MacBook Pro. The Logistic Regression sweep then hung indefinitely on its first trial. After ~6 hours with the parent process burning 99% CPU but no trial completing in MLflow, the run was killed (`kill -9 8490`).

Rather than restart the 25+ hour sweep, a salvage driver (`scripts/salvage_train.py`) recovered the winning XGBoost hyperparameters from the MLflow run store and completed the downstream calibration, anomaly fit, threshold tuning, and persistence steps in ~1h 37min. The resulting `models/ensemble.pkl` and `mlruns/training_summary.json` are byte-compatible with what the original driver would have produced given the same inputs.

The v0 model artifact is in production-ready format and is what the API and UI load. Three substantive issues were uncovered along the way and are scheduled for v1.

## Timeline

| Time (local, MacBook) | Event |
|---|---|
| 2026-05-30 13:13:05 | Training run started via `bash scripts/train.sh`. |
| 13:13:16 | Data loaded: 5,078,345 transactions, positive rate 0.00102. |
| 13:55:01 | Feature engineering complete (42 min). 78 feature columns produced. |
| 13:55:10 | Temporal train/val/test split: 3,554,841 / 761,752 / 761,752 rows. |
| 13:55:10 | XGBoost Optuna sweep begins (40 trials × 3-fold CV). |
| 15:36:01 | XGBoost sweep complete. Best objective −55,966.25. |
| 15:36:01 | LightGBM Optuna sweep begins. |
| 17:49:29 | LightGBM sweep complete. Best objective −56,737.71. |
| 17:49:29 | Random Forest Optuna sweep begins. |
| 2026-05-31 06:04:09 | Random Forest sweep complete (12h 15min). Best objective −70,109.65. |
| 06:04:09 | Logistic Regression Optuna sweep begins. |
| 06:04:09 → ~12:30 | No new trial completed. Process at 99% CPU. MLflow showed 0 LogReg runs. |
| ~12:30 | Process killed via `kill -9 8490` after diagnosis. |
| ~16:30 | Salvage driver started. |
| ~16:30 → 17:53 | (Re-running data load + feature engineering on the salvage driver.) |
| 19:29:41 | Salvage driver complete. `models/ensemble.pkl` and `training_summary.json` written. |

## Diagnosis

### The hang

Logistic Regression's Optuna sweep had been running for ~6 hours without a single trial appearing in MLflow. The parent Python process was at 99% CPU (PID 8490, CPU time 130+ hours across cores) - actively computing, but stuck inside a single trial.

Inspection of `configs/model_config.yaml` revealed the LogReg hyperparameter search range:

```yaml
logistic_regression:
  C:
    type: float
    low: 0.001
    high: 100.0
    log: true
  penalty:
    type: categorical
    choices: [l1, l2]
  solver:
    type: categorical
    choices: [liblinear, saga]
```

The pathological combination is `solver=saga + penalty=l1 + C=0.001`:

- **C=0.001** is very strong L1 regularization, requiring many gradient steps to satisfy convergence tolerance.
- **L1 penalty** is non-differentiable, materially slower than L2 for any solver.
- **SAGA** on 3.55M rows × 78 features without an explicit `max_iter` defaults to scikit-learn's `1000` iteration cap, and each iteration is itself O(n_features) per sample.
- **3-fold CV** multiplies wall time by three.

At those settings a single trial can plausibly take 6+ hours. With Optuna's TPE sampler weighted toward "promising" (in this case: nearly-uniformly-zero coefficient) regions, trial 0 was probabilistically likely to land near `C=0.001` and never finish.

### Why XGBoost / LightGBM / RF didn't have this problem

Tree-based methods don't have continuous regularization parameters that push toward degenerate convergence regimes. Their hyperparameter ranges are "more trees" or "deeper trees," which scale predictably with wall time. The fastest LogReg trials would have been instant (`C=100, penalty=l2, solver=liblinear`); the slowest are unbounded.

### Why the test/val objectives turned out inconsistent

A separate finding, surfaced after the salvage completed: the validation-fold cost-weighted Precision@k reported in `training_summary.json` (−94.84) and the test-fold value (−173,415) are not directly comparable.

`src/models/train.py:_tune_threshold` evaluates the cost-weighted objective at `k = predictions-above-threshold`, which is a moving target as the threshold sweeps. The function then returns the threshold with the best objective at *its own k*. On v0's data this picked a very low threshold (0.0368) that flags ~452,812 of 761,752 validation transactions - essentially "flag everything." That has 0 false negatives (so no FN cost), a huge FP volume but spread across ~106k investigator hours, so the objective looks great: −$94/hour.

Test evaluation, in contrast, uses `cost_weighted_precision_at_k` with default `k = cost_matrix.k_per_day = 384`. Same scores, much smaller k, completely different cost arithmetic. That's the bug.

## Decision

After the LogReg hang was characterized, three options were considered:

1. **Wait it out.** Unknown remaining time - could be hours or weeks. Process was at 99% CPU but no completion guarantee.
2. **Restart with a constrained LogReg config.** Lose 25 hours of completed sweep work. Same ~25 hours to redo XGBoost + LightGBM + RF on the MacBook.
3. **Kill and salvage.** Skip the LogReg sweep entirely. Reuse the XGBoost winner identified by the completed XGBoost sweep. Run downstream calibration + threshold tuning + persist. Expected wall time: ~1.5 hours.

Option 3 was chosen because:

- XGBoost had already won the supervised sweep by a wide margin: −55,966 vs LightGBM −56,737 vs Random Forest −70,109. LogReg almost never beats gradient-boosted trees on imbalanced tabular data; it was unlikely to change the model-selection outcome.
- The original sweep code wrote `models/ensemble.pkl` and `mlruns/training_summary.json` only at the *very end* of the run. Killing the process mid-LogReg meant losing all 121 completed MLflow trial records *unless* a salvage pathway could preserve them.
- Salvage produces a model that is byte-compatible with what the original driver would have written. Downstream consumers (the API, the UI, the notebooks, `scripts/update_results.py`) cannot distinguish a salvage-produced artifact from a sweep-produced one.

The salvage driver (`scripts/salvage_train.py`) imports the existing helpers from `src/models/train.py` (data load, feature engineering, calibration, anomaly fit, threshold tuning, ensemble assembly, summary writing) and substitutes only the model-selection step with a hardcoded winner read from MLflow. The salvage rationale is logged at the top of the run and embedded in `training_summary.json["salvage"]`, so any downstream consumer can detect a salvage-produced artifact.

## Outcome

The v0 model artifact (`models/ensemble.pkl`) carries:

- A calibrated XGBoost classifier (isotonic, 3-fold CV) wrapping the best hyperparameters from the completed XGBoost sweep.
- A fitted Isolation Forest anomaly head using midpoint hyperparameters from the anomaly search space (no Optuna sweep - anomaly head was never swept in either the original or salvage paths).
- A tuned decision threshold of 0.0368 (subject to the val/test k-mismatch caveat below).
- The same `EnsembleMetadata` block any non-salvage artifact would carry, plus a `salvage` block in `training_summary.json` documenting why the run took this path.

The headline operating-point metrics:

| Metric | Value |
|---|---|
| Test Precision@k (k=384/day) | 54.7% |
| Test recall | 13.5% |
| Test true positives | 210 |
| Test false positives | 174 |
| Test false negatives | 1,351 |
| Decision threshold | 0.0368 |

At the operating point, the model is roughly 270× more precise than a uniform-random alerter (54.7% / 0.2%). Recall of 13.5% is significantly lower than what a production compliance team would tolerate; this is partly the model and partly the cost matrix understating the false-negative regulatory exposure (see v1 items below).

## v1 fixes

These are the substantive issues v1 will address. Each one came directly out of this incident.

### Fix the threshold tuner

`_tune_threshold` currently sweeps the threshold against `k = predictions-above-threshold`. The result is a threshold that maximizes the objective at *its own k* but is meaningless for the operational k=384/day the model actually deploys at. v1 changes the tuner to sweep thresholds at *fixed k=384* - keep the top 384 scores regardless of threshold value, evaluate at that fixed k throughout. Expected to make val and test directly comparable and to pull the chosen threshold toward a much higher value than 0.0368.

### Replace weighted ensemble with stacking

The current ensemble combination is `0.35 * anomaly + 0.65 * supervised`, with weights hardcoded in `configs/api_config.yaml`. Empirically the Isolation Forest score range turned out narrow (0.32-0.42), so the anomaly contribution is roughly a constant offset rather than a useful per-transaction signal. v1 feeds the Isolation Forest score as a feature column into XGBoost, letting the supervised model learn the empirical weighting non-linearly. This is the standard recommendation when one component dominates the other.

### Recalibrate the cost matrix

`false_negative_cost_usd = $11,500` is a defensible figure for the average illicit dollars per missed alert in pre-screening contexts, but it badly understates the marginal regulatory cost: a single SAR-worthy missed alert that surfaces in a consent-order examination can carry six-to-seven-figure penalties plus institutional reputation damage. v1 recomputes the FN cost as a blend of average-illicit-dollars-per-miss *plus* expected regulatory penalty per miss. This will pull the cost-weighted objective toward higher-recall operating regions.

### Feature caching

The 42-minute feature-engineering step destroyed iteration speed. v1 writes `data/processed/features.parquet` after the first build and reloads it on subsequent runs unless the source CSV changes. Drops iteration time from 42 min to ~30 sec.

### Fix or remove LogReg

Either constrain the SAGA + L1 + small-C parameter region (set explicit `max_iter`, restrict `C >= 0.01`, drop SAGA from the solver list when L1 is selected), or drop LogReg from the family list entirely with a one-paragraph justification. The latter is leaner - LogReg never wins on imbalanced tabular data anyway.

### GPU XGBoost

v1 runs on a workstation with an RTX 5080 (16GB VRAM, Blackwell architecture). XGBoost with `tree_method="hist", device="cuda"` is typically 10-20× faster than CPU on tabular data of this scale. The original 25-hour sweep should complete in 1-2 hours, making ablation studies tractable.

### Pre-execute notebooks in CI

v0 caught two cases of notebook outputs drifting from code reality (cells with `exec=None` but committed source). v1 runs `jupyter nbconvert --to notebook --execute` on every notebook as part of the test suite, so notebook outputs cannot drift from the code that produced them without CI failing.

## Lessons learned

A few takeaways worth carrying forward to subsequent projects:

**Test the hyperparameter search regions before running them at full scale.** A 10-minute smoke run on a 100k subset would have caught the SAGA hang in time to constrain the config before kicking off the full 25-hour sweep. v1 adds a `--sample-fraction` flag to `src.models.train` for exactly this purpose.

**Write intermediate artifacts often.** The original driver wrote `models/ensemble.pkl` only at the very end of a 25-hour run. A crash at hour 24 would have lost everything. v1 will persist intermediate checkpoints after each family's sweep completes, so a hang or crash is recoverable without restarting.

**Audit the metric implementation under the operating conditions you'll evaluate at.** The `_tune_threshold` bug existed at code-review time but only surfaced when val and test were compared post-hoc. v1 adds a unit test that asserts `_tune_threshold`'s threshold choice produces exactly `k=384` predictions on a synthetic input.

**MLflow's run store is the source of truth, not the terminal log.** The terminal showed only family-level milestones; MLflow showed every per-trial completion. Diagnosis of the LogReg hang was only possible by reading the MLflow run directory directly (the UI was running, but the underlying file layout - `mlruns/<exp_id>/<run_id>/params/` etc. - was the authoritative signal). Operators should treat MLflow's file layout, not its UI, as the system of record.

**Salvage is a first-class engineering pattern.** Production ML systems should be able to recover from partial completion without requiring full re-runs. `scripts/salvage_train.py` is now a permanent fixture of this repo and will be ported (with the bug fixes above) into v1.

---

*This incident report is part of the project's audit trail and is referenced from the [v1 roadmap](../README.md#roadmap). It exists because production ML systems fail in interesting ways and dealing with those failures pragmatically is a more important skill to demonstrate than producing tutorial-grade output where nothing goes wrong.*
