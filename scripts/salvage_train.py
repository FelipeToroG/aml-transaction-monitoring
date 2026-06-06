"""Salvage training driver.

This script exists to recover from a specific failure mode encountered
during the project's first end-to-end training run: the original
``python -m src.models.train`` driver completed the XGBoost, LightGBM,
and Random Forest Optuna sweeps successfully, then hung indefinitely on
the Logistic Regression sweep because of a SAGA-solver + L1-penalty +
very-small-C convergence pathology on the 3.5M-row training fold.

Rather than re-run the entire 25+ hour sweep (and risk hitting the same
hang), this script:

1. Skips the Optuna sweep entirely.
2. Hardcodes the winning XGBoost hyperparameters that the original sweep
   identified (recovered from the MLflow run store).
3. Runs everything *downstream* of model selection exactly as the
   original driver would: feature engineering, temporal split, refit on
   train+val, isotonic calibration, anomaly-head fit, threshold sweep,
   test evaluation, ensemble persistence, training-summary JSON.

The resulting ``models/ensemble.pkl`` and ``mlruns/training_summary.json``
are bit-compatible with what the original driver would have produced
given the same data and hyperparameters — the API, notebooks, and
README updater all consume them transparently.

Per-family comparison data (objective and hyperparameters for each
family that completed its sweep) is pulled from the existing MLflow run
store so the README's per-family table reflects the empirical results
of the original sweep.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import __version__
from src.data.loader import DEFAULT_RAW_PATH, LABEL_COLUMN, DataLoader
from src.data.splits import temporal_train_val_test_split
from src.evaluation.metrics import CostMatrix, cost_weighted_precision_at_k
from src.features.pipelines import build_engineered_frame
from src.models.ensemble import build_ensemble_from_components
from src.models.train import (
    _build_training_summary,
    _configure_logging,
    _default_hyperparameters_from_search_space,
    _fit_supervised_component,
    _load_yaml,
    _tune_threshold,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Frozen sweep results, recovered from MLflow run 71c6d1fd... on
# experiment "aml-transaction-monitoring" after the LogReg hang.
# -----------------------------------------------------------------------

# The overall winner across all completed families. XGBoost beat LightGBM
# by ~$770/investigator-hour and Random Forest by ~$14k/investigator-hour
# on the cost-weighted Precision@k objective.
WINNING_FAMILY = "xgboost"
WINNING_HYPERPARAMETERS: dict[str, Any] = {
    "colsample_bytree": 0.5879059107650935,
    "learning_rate": 0.10127241400894953,
    "max_depth": 11,
    "min_child_weight": 0.9135390394225726,
    "n_estimators": 900,
    "reg_alpha": 2.4915336621055992,
    "reg_lambda": 4.645441918866278,
    "scale_pos_weight": 13.311115736807617,
    "subsample": 0.6311631983195253,
}

# Per-family validation objectives and hyperparameters from the
# completed sweeps. Used to build the README's per-family comparison
# block in the training summary.
COMPLETED_FAMILY_WINNERS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "family": "xgboost",
        "objective": -55966.2514,
        "hyperparameters": WINNING_HYPERPARAMETERS,
    },
    "lightgbm": {
        "family": "lightgbm",
        "objective": -56737.7093,
        "hyperparameters": {
            "bagging_fraction": 0.8380549877235326,
            "bagging_freq": 8,
            "feature_fraction": 0.8122875290749312,
            "is_unbalance": False,
            "learning_rate": 0.03092971246289403,
            "min_child_samples": 38,
            "n_estimators": 800,
            "num_leaves": 131,
            "reg_alpha": 0.0022713387108984264,
            "reg_lambda": 9.873402705069454,
        },
    },
    "random_forest": {
        "family": "random_forest",
        "objective": -70109.6458,
        "hyperparameters": {
            "class_weight": "balanced_subsample",
            "max_depth": 30,
            "max_features": 0.5,
            "min_samples_leaf": 17,
            "min_samples_split": 15,
            "n_estimators": 200,
        },
    },
}


def main(
    *,
    raw_data_path: Path | None = None,
    model_config_path: Path = Path("configs/model_config.yaml"),
    cost_matrix_path: Path = Path("configs/cost_matrix.yaml"),
    api_config_path: Path = Path("configs/api_config.yaml"),
    model_output_path: Path = Path("models/ensemble.pkl"),
    summary_output_path: Path = Path("mlruns/training_summary.json"),
) -> dict[str, Any]:
    """Run the salvage training pipeline.

    Mirrors ``src.models.train.main`` from step 4 onward (data already
    loaded, sweep results known). Returns the training summary dict for
    programmatic callers.
    """
    _configure_logging()
    logger.info("Starting AML salvage training, service_version=%s", __version__)
    logger.info(
        "Skipping Optuna sweep: winner is %s with objective %.4f "
        "(recovered from MLflow after LogReg sweep hang)",
        WINNING_FAMILY,
        COMPLETED_FAMILY_WINNERS[WINNING_FAMILY]["objective"],
    )

    # ----- Load configs -----------------------------------------------
    model_config = _load_yaml(model_config_path)
    # api_config is no longer read here: stacking removed the blend weights
    # the trainer used to pull from it. The path arg is kept for CLI stability.
    cost_matrix = CostMatrix.from_yaml(str(cost_matrix_path))

    # ----- Load and prepare data --------------------------------------
    raw_path = raw_data_path if raw_data_path is not None else DEFAULT_RAW_PATH
    logger.info("Loading raw data from %s", raw_path)
    loader = DataLoader(raw_path=raw_path)
    frame = loader.load()
    logger.info(
        "Loaded %d transactions; positive rate %.5f",
        len(frame),
        frame[LABEL_COLUMN].mean(),
    )

    logger.info("Engineering features (this is the slow step, ~40 min)...")
    bundle = build_engineered_frame(frame)
    logger.info(
        "Engineered %d feature columns",
        len(bundle.numerical_columns) + len(bundle.categorical_columns),
    )

    logger.info("Performing temporal train/val/test split")
    split = temporal_train_val_test_split(bundle.frame)
    split_summary = split.describe()
    logger.info(
        "Split: train=%d val=%d test=%d",
        split_summary["train_rows"],
        split_summary["val_rows"],
        split_summary["test_rows"],
    )

    y_train = split.train[LABEL_COLUMN].to_numpy()
    y_val = split.val[LABEL_COLUMN].to_numpy()
    y_test = split.test[LABEL_COLUMN].to_numpy()

    # ----- Refit winner on train + val (with isotonic calibration) ----
    # This is the calibrated supervised head the production ensemble
    # serves. The fit is on train + val combined so the final model sees
    # the maximum training data the data layout permits while preserving
    # the test fold as held-out evidence.
    train_val_frame = pd.concat([split.train, split.val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])

    # The Isolation Forest is not swept; its config is the midpoint of the
    # search-space ranges and it is fit inside the feature pipeline as the
    # stacking feature, mirroring the production driver.
    anomaly_hp = _default_hyperparameters_from_search_space(
        model_config["families"]["isolation_forest"]
    )

    logger.info(
        "Refitting %s on train+val (%d rows) with isotonic calibration (3-fold)...",
        WINNING_FAMILY,
        len(train_val_frame),
    )
    final_pipeline, final_classifier = _fit_supervised_component(
        bundle=bundle,
        frame=train_val_frame,
        y=y_train_val,
        family=WINNING_FAMILY,
        hyperparameters=WINNING_HYPERPARAMETERS,
        calibrate=True,
        anomaly_hp=anomaly_hp,
    )
    logger.info("Supervised head fit complete")

    # The Isolation Forest now lives inside the fitted pipeline as the
    # stacking feature; read the fitted scorer back out for the metadata.
    anomaly_scorer = final_pipeline.named_steps["anomaly"].anomaly_scorer_

    # ----- Tune decision threshold on val -----------------------------
    logger.info("Tuning decision threshold on validation fold")
    feature_columns = list(
        bundle.numerical_columns + bundle.categorical_columns
    )
    # Stacking: the anomaly score is a feature inside final_pipeline, so the
    # risk score is the calibrated supervised probability on the augmented
    # matrix. No post-hoc weighted blend.
    X_val = final_pipeline.transform(split.val[feature_columns])
    combined_val = final_classifier.predict_proba(X_val)[:, 1]

    # Same fixed operational alert budget the metric uses for the test
    # evaluation below; thread it into both so val tuning and test eval
    # select at an identical k.
    operational_k = cost_matrix.k_per_day

    decision_threshold, threshold_eval = _tune_threshold(
        scores=combined_val,
        y_true=y_val,
        cost_matrix=cost_matrix,
        operational_k=operational_k,
    )
    logger.info(
        "Optimal threshold: %.4f (val objective %.4f)",
        decision_threshold,
        threshold_eval["objective_cost_per_investigator_hour_usd"],
    )

    # ----- Evaluate on held-out test fold -----------------------------
    logger.info("Evaluating on held-out test fold")
    X_test = final_pipeline.transform(split.test[feature_columns])
    combined_test = final_classifier.predict_proba(X_test)[:, 1]
    test_eval = cost_weighted_precision_at_k(
        y_true=y_test,
        scores=combined_test,
        cost_matrix=cost_matrix,
        k=operational_k,
    )
    logger.info(
        "Test objective: %.4f  precision@k: %.4f",
        test_eval["objective_cost_per_investigator_hour_usd"],
        test_eval["precision_at_k"],
    )

    # ----- Assemble and persist ensemble ------------------------------
    ensemble = build_ensemble_from_components(
        bundle=bundle,
        fitted_pipeline=final_pipeline,
        fitted_anomaly=anomaly_scorer,
        fitted_classifier=final_classifier,
        decision_threshold=decision_threshold,
        selected_family=WINNING_FAMILY,
        selected_hyperparameters=WINNING_HYPERPARAMETERS,
        training_data_rows=len(train_val_frame),
        training_data_temporal_range=(
            split_summary["val_start"],
            split_summary["test_start"],
        ),
        eval_metrics={
            "val_objective": threshold_eval["objective_cost_per_investigator_hour_usd"],
            "test_objective": test_eval["objective_cost_per_investigator_hour_usd"],
            "test_precision_at_k": test_eval["precision_at_k"],
            "test_true_positives": float(test_eval["true_positives"]),
            "test_false_positives": float(test_eval["false_positives"]),
            "test_false_negatives": float(test_eval["false_negatives"]),
        },
    )
    ensemble.save(model_output_path)
    logger.info("Saved ensemble artifact to %s", model_output_path)

    # ----- Write training summary for update_results.py ---------------
    # The summary's per-family block reflects the empirical results of
    # the original Optuna sweep (XGBoost, LightGBM, Random Forest). A
    # ``salvage`` flag is added so any consumer reading the JSON can
    # tell this run was assembled from a salvage script rather than a
    # complete end-to-end sweep.
    summary = _build_training_summary(
        winning_family=WINNING_FAMILY,
        winning_hyperparameters=WINNING_HYPERPARAMETERS,
        family_winners=COMPLETED_FAMILY_WINNERS,
        anomaly_metadata=anomaly_scorer.get_metadata(),
        threshold=decision_threshold,
        val_eval=threshold_eval,
        test_eval=test_eval,
        split_summary=split_summary,
        cost_matrix=cost_matrix,
    )
    summary["salvage"] = {
        "salvage_used": True,
        "reason": (
            "Original sweep hung on Logistic Regression trial 0 due to "
            "SAGA + L1 + small-C convergence pathology on 3.5M rows. "
            "XGBoost, LightGBM, and Random Forest sweeps completed "
            "normally; their results are reflected in `all_families` "
            "above. Logistic Regression is excluded from the per-family "
            "comparison because no trial completed."
        ),
        "salvage_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote training summary to %s", summary_output_path)

    return summary


if __name__ == "__main__":
    main()
