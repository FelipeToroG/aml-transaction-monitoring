# Evaluation methodology

## Why cost-weighted Precision@k

Banking ML evaluation differs from academic evaluation in one decisive way: the cost matrix is asymmetric and known. False negatives cost laundered dollars plus regulatory exposure; false positives cost investigator time. A model that scores 0.92 AUC-PR but produces four times the alert volume of the production baseline is not deployable — investigators have finite review capacity, and a higher alert volume at constant team size simply means alerts age past their SLA.

The training objective in `src.models.train` is **investigator-hour-cost-weighted Precision@k**, where `k` is calibrated to the team's actual daily review throughput. This is what model selection looks like when investigator time is the binding constraint.

## Headline metric definition

For the top-k scored items at a fixed `k`:

```
total_cost  = false_positives * c_FP + false_negatives * c_FN
hours_spent = k * (c_FP / hourly_rate)
objective   = -(total_cost / hours_spent)        # negated so larger is better
```

Costs sourced from `configs/cost_matrix.yaml`. Default values:

| Parameter | Value | Source |
|---|---|---|
| `false_negative_cost_usd` | $16,000 | Avg illicit dollars per missed alert ($8,500, FinCEN SAR aggregates) + expected regulatory penalty ($25,000 × 30% detection probability = $7,500) |
| `false_positive_cost_usd` | $22.17 | $95/hour fully loaded investigator rate × 14 min review = $22.17 |
| `k_per_day` | 384 | 8 analysts × 48 alerts/analyst/day |
| `investigator_hourly_rate_usd` | $95 | Tier-2 compliance analyst, US-based fintech |

## Why not AUC-PR

AUC-PR is the standard academic discrimination metric on imbalanced data. It is used in this project as the **CV scoring metric** inside Optuna trials because it is fast to compute and produces dense gradients for the TPE sampler. It is **not** used for final model selection.

The reason: AUC-PR integrates over every operating point. In production the model operates at one threshold, calibrated to the team's review capacity. A model that ranks well on average but badly at the operating threshold scores high on AUC-PR but performs poorly in production. Cost-weighted Precision@k evaluates exactly the operating point the model will be deployed at.

## Threshold tuning

After selecting the winning family on cost-weighted Precision@k, the training driver tunes the decision threshold by sweeping 200 candidate thresholds between the 1st and 99th percentile of the validation-set score distribution. The cost-optimal threshold is selected.

The candidate range concentrates the search where decisions flip, not on `[0, 1]` uniformly. The vast majority of scores in a real AML distribution fall in a narrow band; sweeping `[0, 1]` uniformly wastes 99% of the search budget on operating points the threshold will never sit at.

## Per-family comparison

Five model families are evaluated in the Optuna sweep (Isolation Forest is the anomaly head, not included in the supervised sweep):

| Family | Why included |
|---|---|
| XGBoost | Consistent strongest performer on imbalanced tabular data |
| LightGBM | Competitive accuracy with significantly faster training |
| Random Forest | Stable baseline; less overfitting risk than boosted variants |
| Logistic Regression | Linear baseline so the README can quantify ensemble lift |

The winning family is whichever achieves the highest validation-set cost-weighted Precision@k. The `all_families` block in the training summary preserves all winners for the per-family comparison table in the README.

## Calibration

Tree ensembles produce systematically over-confident predictions on imbalanced data. The final selected model is wrapped in `CalibratedClassifierCV` with isotonic regression (3-fold). Calibration is *not* applied during the Optuna sweep because three-fold CV adds 3× training cost per trial; the marginal AUC-PR gain from calibration is not large enough to change the model-family ranking.

Three calibration metrics are computed in `src.evaluation.calibration`:

- **Brier score** — properly scored discrimination + calibration combined scalar
- **Expected Calibration Error (ECE)** — bin-weighted absolute gap between predicted and empirical positive rate
- **Reliability curve** — visual binning for the README and notebook figures

A model that misrepresents its own confidence to game any of these scores will score worse than one that reports true probabilities — properly scored metrics are the right primary diagnostics.

## Investigator simulation

`src.evaluation.investigator_simulator` runs a discrete-event simulation of the alert queue under a configured analyst pool and SLA targets. Outputs:

- Per-alert wait, review, and disposition timestamps
- SLA attainment rate per tier
- Wait-time percentiles (p50 / p95 / p99)
- End-of-window backlog

This is the operational evidence that the model is *deployable*, not just accurate. A model that passes Precision@k but produces alerts faster than investigators can clear them will fail in production. The simulator surfaces that failure before deployment.

## Snapshot artifacts

Every training run writes `mlruns/training_summary.json` — a structured record carrying the winning family, hyperparameters, decision threshold, val and test metrics, cost matrix, and the split's temporal boundaries.

The README's results table is rebuilt from this JSON by `scripts/update_results.py` at the end of each training run. The same JSON is the audit-snapshot source for `src.evaluation.reports.build_audit_snapshot` — regulators reviewing a historical alert can reconstruct the exact model performance and cost matrix that produced it.
