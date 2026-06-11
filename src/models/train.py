"""Training driver for the AML hybrid ensemble.

This is the single entry point for end-to-end model training. The script
(invoked via ``scripts/train.sh`` or ``python -m src.models.train``):

1. Loads the raw IBM AML HI-Small dataset.
2. Builds the engineered feature frame once (causal, so safe before
   splitting).
3. Performs the temporal train/val/test split.
4. For each model family in ``configs/model_config.yaml``, runs an
   Optuna sweep with the cost-weighted Precision@k objective on the
   validation fold. Every trial is logged to MLflow.
5. Selects the overall winner across families on the validation
   objective.
6. Refits the winner on ``train + val``, fits the anomaly component
   alongside, assembles the ensemble.
7. Tunes the decision threshold on the validation fold by sweeping
   candidate thresholds and selecting the cost-optimal cut.
8. Evaluates the assembled ensemble on the held-out test fold.
9. Saves the ensemble artifact and writes a training-summary JSON to
   ``mlruns/`` for ``scripts/update_results.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src import __version__
from src.data.loader import (
    DEFAULT_RAW_PATH,
    LABEL_COLUMN,
    TIMESTAMP_COLUMN,
    DataLoader,
)
from src.data.splits import temporal_train_val_test_split
from src.evaluation.metrics import (
    CostMatrix,
    cost_weighted_precision_at_k,
)
from src.features.pipelines import (
    FeatureBundle,
    build_engineered_frame,
    build_feature_pipeline,
)
from src.models.classifier import build_classifier
from src.models.ensemble import build_ensemble_from_components

logger = logging.getLogger(__name__)

# Optuna's default verbosity is too chatty for batch training; we only
# want WARNING-level Optuna logs so the trial-level logging from this
# module is the dominant signal in the console output.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Engineered-feature cache. Building the engineered frame is the slow
# step (rolling per-entity aggregates over millions of rows), so the
# result is memoised to parquet alongside a JSON sidecar. The cache key
# is the pair (raw CSV mtime, sample_fraction): either value changing
# invalidates the cache. The sample_fraction half is load-bearing,
# without it a fractionally sampled frame from an iteration run could be
# served to a full-data run and silently train on a sliver of the rows.
_FEATURE_CACHE_DIR = Path("data/processed")
_FEATURE_CACHE_FRAME_PATH = _FEATURE_CACHE_DIR / "features.parquet"
_FEATURE_CACHE_META_PATH = _FEATURE_CACHE_DIR / "features.meta.json"


def main(
    *,
    raw_data_path: Path | None = None,
    model_config_path: Path = Path("configs/model_config.yaml"),
    cost_matrix_path: Path = Path("configs/cost_matrix.yaml"),
    api_config_path: Path = Path("configs/api_config.yaml"),
    model_output_path: Path = Path("models/ensemble.pkl"),
    summary_output_path: Path = Path("mlruns/training_summary.json"),
    mlflow_experiment: str = "aml-transaction-monitoring",
    sample_fraction: float | None = None,
) -> dict[str, Any]:
    """Run the end-to-end training pipeline.

    Returns the training summary dict for programmatic callers (e.g.,
    notebooks). The same dict is written to ``summary_output_path`` for
    ``scripts/update_results.py``.
    """
    _configure_logging()
    logger.info("Starting AML training run, service_version=%s", __version__)

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
    logger.info("Loaded %d transactions; positive rate %.5f",
                len(frame), frame[LABEL_COLUMN].mean())

    if sample_fraction is not None:
        # Pipeline-verification tool only: this trims wall-clock so the
        # slow feature build runs over a fraction of the rows during
        # plumbing checks. Metrics produced at small fractions are not
        # meaningful, and a small early slice may contain very few (or
        # zero) positives given the ~0.1% base rate.
        #
        # The slice is a contiguous early time window, not a random
        # sample. Engineered features are rolling per-entity aggregates
        # and the train/val/test split is chronological, so random
        # sampling would tear holes in entity histories and scramble the
        # temporal ordering the split depends on. Sorting by timestamp
        # and taking the leading rows preserves both.
        original_rows = len(frame)
        frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable")
        frame = frame.head(int(original_rows * sample_fraction)).reset_index(drop=True)
        logger.info("Subsampled to %d/%d rows (fraction=%.4f, contiguous early slice)",
                    len(frame), original_rows, sample_fraction)

    bundle = _load_or_build_features(
        frame, raw_path=raw_path, sample_fraction=sample_fraction
    )
    logger.info("Engineered %d feature columns", len(bundle.numerical_columns) + len(bundle.categorical_columns))

    logger.info("Performing temporal train/val/test split")
    split = temporal_train_val_test_split(bundle.frame)
    split_summary = split.describe()
    logger.info("Split: train=%d val=%d test=%d",
                split_summary["train_rows"], split_summary["val_rows"], split_summary["test_rows"])

    # Build label vectors. The model trains on numeric features only;
    # the label is held out and rejoined positionally during evaluation.
    y_train = split.train[LABEL_COLUMN].to_numpy()
    y_val = split.val[LABEL_COLUMN].to_numpy()
    y_test = split.test[LABEL_COLUMN].to_numpy()

    # ----- MLflow setup -----------------------------------------------
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"sweep-{datetime.now(timezone.utc).isoformat()}"):
        # Log run-level context so every trial in the nested runs ties
        # back to the data version and config that produced it.
        mlflow.log_params(
            {
                "service_version": __version__,
                "training_rows": split_summary["train_rows"],
                "validation_rows": split_summary["val_rows"],
                "test_rows": split_summary["test_rows"],
                "k_per_day": cost_matrix.k_per_day,
            }
        )

        # ----- Precompute the stacking feature once -------------------
        # The Isolation Forest is not swept (the supervised side dominates
        # the objective), so its config is the midpoint of the search-space
        # ranges, computed once and shared by the sweep-time stacking feature
        # and the final model so both train against the same anomaly signal.
        anomaly_hp = _default_hyperparameters_from_search_space(
            model_config["families"]["isolation_forest"]
        )
        # Preprocessing and the IF are family-independent and the IF is not
        # swept, so fit them once and reuse the augmented train/val matrices
        # across every family's trials. This collapses the IF fit from
        # (families x trials) to one and drops the per-trial preprocessing
        # refit the old loop paid for nothing.
        X_train_aug, X_val_aug = _build_sweep_matrices(
            bundle=bundle,
            split_train=split.train,
            split_val=split.val,
            anomaly_hp=anomaly_hp,
        )

        # ----- Optuna sweep per family --------------------------------
        family_winners: dict[str, dict[str, Any]] = {}
        for family, search_space in model_config["families"].items():
            if family == "isolation_forest":
                # The anomaly head is fitted later from a separate set of
                # hyperparameters; it is not part of the supervised sweep.
                continue
            logger.info("Optuna sweep: %s (%d trials)",
                        family, model_config["global"]["n_trials_per_family"])

            best_trial_info = _run_family_sweep(
                family=family,
                search_space=search_space,
                X_train=X_train_aug,
                y_train=y_train,
                X_val=X_val_aug,
                y_val=y_val,
                cost_matrix=cost_matrix,
                global_cfg=model_config["global"],
            )
            family_winners[family] = best_trial_info
            logger.info("Family %s best objective: %.4f",
                        family, best_trial_info["objective"])

        # ----- Select overall winner ----------------------------------
        winning_family, winning_info = max(
            family_winners.items(),
            key=lambda kv: kv[1]["objective"],
        )
        logger.info("Overall winner: family=%s objective=%.4f",
                    winning_family, winning_info["objective"])

        # ----- Refit winner on train + val ----------------------------
        train_val_frame = pd.concat([split.train, split.val], ignore_index=True)
        y_train_val = np.concatenate([y_train, y_val])

        logger.info("Refitting winner on train+val (%d rows)", len(train_val_frame))
        final_pipeline, final_classifier = _fit_supervised_component(
            bundle=bundle,
            frame=train_val_frame,
            y=y_train_val,
            family=winning_family,
            hyperparameters=winning_info["hyperparameters"],
            calibrate=True,
            anomaly_hp=anomaly_hp,
        )

        # The Isolation Forest now lives inside the fitted pipeline as the
        # stacking feature. Read the fitted scorer back out for the ensemble
        # metadata and audit record; it is not a separate scoring path.
        anomaly_scorer = final_pipeline.named_steps["anomaly"].anomaly_scorer_

        # ----- Tune decision threshold on val -------------------------
        logger.info("Tuning decision threshold on validation fold")
        feature_columns = list(bundle.numerical_columns + bundle.categorical_columns)

        # Stacking: the anomaly score is a feature inside final_pipeline, so
        # the risk score is just the calibrated supervised probability on the
        # augmented matrix. There is no post-hoc weighted blend.
        X_val_final = final_pipeline.transform(split.val[feature_columns])
        combined_val = final_classifier.predict_proba(X_val_final)[:, 1]

        # The fold's fixed operational alert budget: the team reviews
        # exactly k_per_day alerts, and this is the same k the metric
        # defaults to for the final test evaluation below. Computing it
        # once here and threading it into both the threshold tuner and the
        # test eval makes the shared budget explicit, so val tuning and
        # test selection provably operate at an identical k.
        operational_k = cost_matrix.k_per_day

        decision_threshold, threshold_eval = _tune_threshold(
            scores=combined_val,
            y_true=y_val,
            cost_matrix=cost_matrix,
            operational_k=operational_k,
        )
        logger.info("Optimal threshold: %.4f (objective %.4f)",
                    decision_threshold, threshold_eval["objective_cost_per_investigator_hour_usd"])

        # ----- Evaluate on test ---------------------------------------
        logger.info("Evaluating on held-out test fold")
        X_test_final = final_pipeline.transform(split.test[feature_columns])
        combined_test = final_classifier.predict_proba(X_test_final)[:, 1]
        test_eval = cost_weighted_precision_at_k(
            y_true=y_test,
            scores=combined_test,
            cost_matrix=cost_matrix,
            k=operational_k,
        )
        logger.info("Test objective: %.4f precision@k: %.4f",
                    test_eval["objective_cost_per_investigator_hour_usd"],
                    test_eval["precision_at_k"])

        # ----- Assemble and persist ensemble --------------------------
        ensemble = build_ensemble_from_components(
            bundle=bundle,
            fitted_pipeline=final_pipeline,
            fitted_anomaly=anomaly_scorer,
            fitted_classifier=final_classifier,
            decision_threshold=decision_threshold,
            selected_family=winning_family,
            selected_hyperparameters=winning_info["hyperparameters"],
            training_data_rows=len(train_val_frame),
            training_data_temporal_range=(
                split_summary["val_start"],  # train+val begins at start of train
                split_summary["test_start"],  # ends at start of test
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

        # ----- Log final metrics to MLflow ----------------------------
        mlflow.log_metrics(
            {
                "val_objective": float(threshold_eval["objective_cost_per_investigator_hour_usd"]),
                "test_objective": float(test_eval["objective_cost_per_investigator_hour_usd"]),
                "test_precision_at_k": float(test_eval["precision_at_k"]),
                "decision_threshold": float(decision_threshold),
            }
        )
        mlflow.log_artifact(str(model_output_path))

        # ----- Write training summary for update_results.py -----------
        summary = _build_training_summary(
            winning_family=winning_family,
            winning_hyperparameters=winning_info["hyperparameters"],
            family_winners=family_winners,
            anomaly_metadata=anomaly_scorer.get_metadata(),
            threshold=decision_threshold,
            val_eval=threshold_eval,
            test_eval=test_eval,
            split_summary=split_summary,
            cost_matrix=cost_matrix,
        )
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Wrote training summary to %s", summary_output_path)

    return summary


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _load_or_build_features(
    frame: pd.DataFrame,
    *,
    raw_path: Path,
    sample_fraction: float | None,
) -> FeatureBundle:
    """Return the engineered bundle, reusing the parquet cache when valid.

    The cache is reused only when both the raw CSV mtime and the
    sample_fraction match the sidecar a prior run wrote. A mismatch on
    either rebuilds and overwrites, which is what keeps a sampled feature
    frame from ever being served to a full run.
    """
    raw_mtime = raw_path.stat().st_mtime
    cached = _read_feature_cache(raw_mtime=raw_mtime, sample_fraction=sample_fraction)
    if cached is not None:
        logger.info("Feature cache hit at %s; skipping the slow feature build",
                    _FEATURE_CACHE_FRAME_PATH)
        return cached

    logger.info("Feature cache miss; engineering features (this is the slow step)...")
    bundle = build_engineered_frame(frame)
    _write_feature_cache(bundle, raw_mtime=raw_mtime, sample_fraction=sample_fraction)
    logger.info("Rebuilt feature cache at %s", _FEATURE_CACHE_FRAME_PATH)
    return bundle


def _read_feature_cache(
    *, raw_mtime: float, sample_fraction: float | None
) -> FeatureBundle | None:
    """Reconstruct a FeatureBundle from the cache when its key matches.

    Returns None when the cache is absent or stale so the caller falls
    back to a rebuild. The frame is restored from parquet while the
    column-list metadata travels in the sidecar, so the reconstructed
    bundle is indistinguishable from a freshly built one and every
    downstream consumer stays unchanged.
    """
    if not (_FEATURE_CACHE_FRAME_PATH.exists() and _FEATURE_CACHE_META_PATH.exists()):
        return None

    meta = json.loads(_FEATURE_CACHE_META_PATH.read_text())
    # Both halves of the key must match. Comparing sample_fraction here is
    # the guard that stops a fractionally sampled frame from satisfying a
    # full-data request (None vs a float never compares equal).
    if (
        meta.get("raw_mtime") != raw_mtime
        or meta.get("sample_fraction") != sample_fraction
    ):
        return None

    frame = pd.read_parquet(_FEATURE_CACHE_FRAME_PATH)
    return FeatureBundle(
        frame=frame,
        numerical_columns=tuple(meta["numerical_columns"]),
        categorical_columns=tuple(meta["categorical_columns"]),
    )


def _write_feature_cache(
    bundle: FeatureBundle, *, raw_mtime: float, sample_fraction: float | None
) -> None:
    """Persist the engineered frame and the sidecar that keys the cache."""
    _FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bundle.frame.to_parquet(_FEATURE_CACHE_FRAME_PATH)
    # The column lists are recorded so a cache hit can rebuild the bundle
    # without re-deriving them, and the key fields so the next run can
    # decide hit-or-rebuild without touching the (large) parquet file.
    meta = {
        "raw_mtime": raw_mtime,
        "sample_fraction": sample_fraction,
        "numerical_columns": list(bundle.numerical_columns),
        "categorical_columns": list(bundle.categorical_columns),
    }
    _FEATURE_CACHE_META_PATH.write_text(json.dumps(meta, indent=2))


def _run_family_sweep(
    *,
    family: str,
    search_space: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cost_matrix: CostMatrix,
    global_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run an Optuna sweep for a single classifier family.

    Preprocessing and the Isolation Forest stacking feature are fit once by
    the caller and passed in as the augmented ``X_train`` / ``X_val``
    matrices, so each trial only fits and scores the supervised classifier:
    sample hyperparameters → fit on the augmented train matrix → score the
    augmented val matrix → cost-weighted Precision@k. The best trial wins
    the family.
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters per the search space definition. The
        # _sample_search_space helper dispatches on the ``type`` key to
        # the appropriate Optuna suggest_* call.
        hp = _sample_search_space(trial, search_space)
        classifier = build_classifier(
            family=family,
            hyperparameters=hp,
            random_state=global_cfg["random_state"],
            calibrate=False,  # calibration is applied to the final winner
        )
        classifier.fit(X_train, y_train)
        proba = classifier.predict_proba(X_val)[:, 1]
        result = cost_weighted_precision_at_k(
            y_true=y_val,
            scores=proba,
            cost_matrix=cost_matrix,
        )

        # Per-trial MLflow logging happens in a nested run so the
        # Optuna sweep produces a coherent run tree in MLflow's UI.
        with mlflow.start_run(nested=True, run_name=f"{family}-trial-{trial.number}"):
            mlflow.log_params({**hp, "family": family})
            mlflow.log_metrics(
                {
                    "objective": result["objective_cost_per_investigator_hour_usd"],
                    "precision_at_k": result["precision_at_k"],
                    "true_positives": result["true_positives"],
                    "false_positives": result["false_positives"],
                    "false_negatives": result["false_negatives"],
                }
            )

        return result["objective_cost_per_investigator_hour_usd"]

    sampler = TPESampler(seed=global_cfg["random_state"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=global_cfg["n_trials_per_family"],
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    return {
        "family": family,
        "hyperparameters": best_trial.params,
        "objective": best_trial.value,
    }


def _sample_search_space(
    trial: optuna.Trial, search_space: dict[str, Any]
) -> dict[str, Any]:
    """Translate a YAML search-space dict into Optuna ``suggest_*`` calls.

    The YAML keys' ``type`` field selects the Optuna API:
    ``int`` → ``suggest_int``, ``float`` → ``suggest_float`` with the
    optional ``log`` flag, ``categorical`` → ``suggest_categorical``.
    """
    sampled: dict[str, Any] = {}
    for name, spec in search_space.items():
        spec_type = spec["type"]
        if spec_type == "int":
            sampled[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif spec_type == "float":
            sampled[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif spec_type == "categorical":
            sampled[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(
                f"Unknown search-space type {spec_type!r} for parameter {name!r}"
            )
    return sampled


def _default_hyperparameters_from_search_space(
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Pick the midpoint of each parameter's range.

    Used for the anomaly head, which is not swept (the supervised side
    dominates the objective; sweeping both would multiply training cost
    without proportional gain). Midpoint selection is a defensible
    starting point that can be retuned in a follow-up if drift in the
    anomaly head's quality emerges in production.
    """
    defaults: dict[str, Any] = {}
    for name, spec in search_space.items():
        spec_type = spec["type"]
        if spec_type == "int":
            defaults[name] = (spec["low"] + spec["high"]) // 2
        elif spec_type == "float":
            defaults[name] = (spec["low"] + spec["high"]) / 2.0
        elif spec_type == "categorical":
            defaults[name] = spec["choices"][0]
    return defaults


def _fit_supervised_component(
    *,
    bundle: FeatureBundle,
    frame: pd.DataFrame,
    y: np.ndarray,
    family: str,
    hyperparameters: dict[str, Any],
    calibrate: bool,
    anomaly_hp: dict[str, Any],
) -> tuple[Pipeline, BaseEstimator]:
    """Fit the augmented feature pipeline and supervised classifier on a fold.

    The pipeline includes the Isolation Forest stacking step (``anomaly_hp``),
    so the classifier is trained on the augmented matrix and the fitted forest
    is reachable at ``pipeline.named_steps["anomaly"].anomaly_scorer_``.
    """
    feature_columns = list(bundle.numerical_columns + bundle.categorical_columns)
    pipeline = build_feature_pipeline(
        numerical_columns=bundle.numerical_columns,
        categorical_columns=bundle.categorical_columns,
        anomaly=anomaly_hp,
    )
    X = pipeline.fit_transform(frame[feature_columns])

    classifier = build_classifier(
        family=family,
        hyperparameters=hyperparameters,
        calibrate=calibrate,
    )
    classifier.fit(X, y)
    return pipeline, classifier


def _build_sweep_matrices(
    *,
    bundle: FeatureBundle,
    split_train: pd.DataFrame,
    split_val: pd.DataFrame,
    anomaly_hp: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing + the Isolation Forest once and return augmented X.

    The scaler/encoder and the stacking forest are family-independent and the
    forest is not swept, so they are fit a single time on the train fold and
    applied to train and val. Every family's trials then reuse the returned
    augmented matrices, which is what keeps the IF fit out of the per-trial
    loop. The fitted pipeline is intentionally discarded: the final model
    refits its own pipeline on train+val.
    """
    feature_columns = list(bundle.numerical_columns + bundle.categorical_columns)
    pipeline = build_feature_pipeline(
        numerical_columns=bundle.numerical_columns,
        categorical_columns=bundle.categorical_columns,
        anomaly=anomaly_hp,
    )
    X_train = pipeline.fit_transform(split_train[feature_columns])
    X_val = pipeline.transform(split_val[feature_columns])
    return X_train, X_val


def _tune_threshold(
    *,
    scores: np.ndarray,
    y_true: np.ndarray,
    cost_matrix: CostMatrix,
    operational_k: int,
) -> tuple[float, dict[str, Any]]:
    """Set the decision threshold at the fixed operational alert capacity.

    At a fixed alert budget the threshold is a capacity constraint, not an
    optimisation target: investigators review exactly ``operational_k``
    alerts per fold, so the threshold is whatever score admits that many
    and no more. There is nothing to sweep.

    The v0 tuner swept candidate thresholds and evaluated each at a
    *variable* k equal to the count of predictions above it, then returned
    the threshold whose objective was best at its own k. That let it settle
    on a near-zero threshold flagging most of the validation fold: zero
    false negatives spread over a huge investigator-hour denominator looked
    optimal, but the chosen k bore no relation to the fixed k=384/day the
    model is actually evaluated and deployed at, so val and test were never
    comparable. See the "threshold tuner misaligned with operational k"
    finding in docs/INCIDENT_REPORT.md.

    ``operational_k`` is supplied by the caller from the same cost matrix
    that drives the final test evaluation, so val tuning and test eval
    select on an identical alert budget.
    """
    n = len(scores)
    # A fold smaller than the daily budget cannot surface more alerts than
    # it has rows, so clamp rather than letting the rank index run off the
    # end of the array.
    k = min(operational_k, n)

    # Threshold = the k-th largest score (descending rank k). With distinct
    # scores this admits exactly k alerts: (scores >= threshold).sum() == k.
    # np.partition puts the (n - k)-th ascending element in its sorted slot
    # (that element is the k-th largest) in O(n), avoiding a full sort.
    # Boundary ties can push the count slightly above k when several rows
    # share the k-th score; that is accepted and documented rather than
    # broken with an arbitrary tie-break.
    threshold = float(np.partition(scores, n - k)[n - k])

    result = cost_weighted_precision_at_k(
        y_true=y_true,
        scores=scores,
        cost_matrix=cost_matrix,
        k=k,
    )
    return threshold, result


def _build_training_summary(
    *,
    winning_family: str,
    winning_hyperparameters: dict[str, Any],
    family_winners: dict[str, dict[str, Any]],
    anomaly_metadata: dict[str, Any],
    threshold: float,
    val_eval: dict[str, Any],
    test_eval: dict[str, Any],
    split_summary: dict[str, Any],
    cost_matrix: CostMatrix,
) -> dict[str, Any]:
    """Assemble the JSON summary consumed by scripts/update_results.py."""
    return {
        "service_version": __version__,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "winning_family": winning_family,
        "winning_hyperparameters": winning_hyperparameters,
        "all_families": {
            name: {
                "objective": info["objective"],
                "hyperparameters": info["hyperparameters"],
            }
            for name, info in family_winners.items()
        },
        "anomaly_metadata": anomaly_metadata,
        "decision_threshold": threshold,
        "val": val_eval,
        "test": test_eval,
        "split": split_summary,
        "cost_matrix": {
            "k_per_day": cost_matrix.k_per_day,
            "false_negative_cost_usd": cost_matrix.false_negative_cost_usd,
            "false_positive_cost_usd": cost_matrix.false_positive_cost_usd,
        },
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def _configure_logging() -> None:
    """Set up structured-enough logging for batch training runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AML hybrid ensemble.")
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=None,
        help="Override the raw data path (defaults to data/raw/HI-Small_Trans.csv).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to the model configuration YAML.",
    )
    parser.add_argument(
        "--cost-matrix",
        type=Path,
        default=Path("configs/cost_matrix.yaml"),
        help="Path to the cost matrix YAML.",
    )
    parser.add_argument(
        "--api-config",
        type=Path,
        default=Path("configs/api_config.yaml"),
        help="Path to the API configuration YAML (provides ensemble weights).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/ensemble.pkl"),
        help="Where to write the serialised ensemble artifact.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("mlruns/training_summary.json"),
        help="Where to write the training summary JSON.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help=(
            "Train on a contiguous early time slice of this fraction of the "
            "rows (e.g. 0.02). Pipeline-verification tool only: metrics at "
            "small fractions are not meaningful. Default trains on full data."
        ),
    )
    args = parser.parse_args()

    main(
        raw_data_path=args.raw_data_path,
        model_config_path=args.model_config,
        cost_matrix_path=args.cost_matrix,
        api_config_path=args.api_config,
        model_output_path=args.model_output,
        summary_output_path=args.summary_output,
        sample_fraction=args.sample_fraction,
    )
