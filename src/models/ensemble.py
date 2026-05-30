"""Production AML scoring ensemble.

The ensemble is the single object the API loads at startup and uses to
score every incoming transaction. It composes three layers:

1. The fitted **feature pipeline** (``sklearn.Pipeline`` from
   ``src.features.pipelines``) that turns engineered features into the
   model-ready matrix.
2. The fitted **anomaly scorer** (``AnomalyScorer``) producing a
   calibrated ``[0, 1]`` novelty score.
3. The fitted **supervised classifier** producing a calibrated ``[0, 1]``
   laundering probability.

The two component scores are combined with weights tuned during training
to maximise the cost-weighted Precision@k objective, and the combined
score is compared against the decision threshold to produce the binary
alert decision.

Serialisation
-------------
The ensemble is serialised to ``models/ensemble.pkl`` with joblib,
along with a metadata dictionary that captures schema version, fit
timestamp, training-data provenance, and the component-level metadata
from each underlying model. Audit reconstruction of any historical
alert reads from this metadata block — there is no other place the
necessary model context lives.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src import __version__
from src.features.pipelines import (
    FeatureBundle,
    build_engineered_frame,
    build_feature_pipeline,
)
from src.models.anomaly import AnomalyScorer
from src.models.classifier import get_classifier_info

# Embedded schema version. Any breaking change to the ensemble's
# serialised shape increments this; the loader refuses to deserialise
# an artifact whose schema version does not match the running service.
SCHEMA_VERSION: Final[int] = 1


@dataclass(slots=True)
class EnsembleMetadata:
    """Provenance and configuration metadata for a serialised ensemble.

    Frozen-by-convention (we never mutate after fit). Embedded in every
    persisted alert record so any alert can be reconciled to the exact
    artifact that produced it years later, including which training
    data it was fit on and which Optuna trial produced its component
    hyperparameters.
    """

    service_version: str
    schema_version: int
    fit_timestamp_utc: str
    training_data_rows: int
    training_data_temporal_range: tuple[str, str]
    selected_classifier_family: str
    selected_classifier_hyperparameters: dict[str, Any]
    anomaly_metadata: dict[str, Any]
    classifier_metadata: dict[str, Any]
    ensemble_weights: dict[str, float]
    decision_threshold: float
    eval_metrics: dict[str, float]


@dataclass(slots=True)
class AMLEnsemble:
    """The production scoring object.

    The ``fit`` orchestration is called by the training driver
    (``src.models.train``); the ``score`` and ``predict`` methods are
    called by the API at inference time.

    The feature pipeline, anomaly scorer, and supervised classifier are
    public attributes so the test suite and notebooks can inspect them.
    The combination weights and threshold are also public for the same
    reason.
    """

    feature_pipeline: Pipeline
    anomaly_scorer: AnomalyScorer
    supervised_classifier: BaseEstimator
    anomaly_weight: float
    supervised_weight: float
    decision_threshold: float
    feature_columns: tuple[str, ...]
    metadata: EnsembleMetadata | None = field(default=None)

    def score(self, raw_frame: pd.DataFrame) -> np.ndarray:
        """Return the combined risk score for every transaction.

        Pipeline:

        1. Build engineered features from the raw frame.
        2. Apply the fitted feature pipeline to produce the model matrix.
        3. Score with both components.
        4. Combine the scores with the configured weights.

        Parameters
        ----------
        raw_frame : pd.DataFrame
            Raw transactions with the canonical schema. Can contain a
            single transaction or a batch; the feature engineering layer
            handles both.

        Returns
        -------
        np.ndarray
            Combined risk scores in ``[0, 1]``. Shape ``(n_rows,)``.
        """
        # Engineer features. For batch inference this is one feature
        # pass over the whole batch — efficient. For single-transaction
        # inference the entity rolling features see only the current
        # transaction (no history); the runtime serving path documented
        # in the README's roadmap addresses this by maintaining a
        # rolling feature-store cache. For batch scoring (the common
        # production pattern for AML monitoring), the in-batch context
        # is sufficient.
        bundle = build_engineered_frame(raw_frame)

        # Restrict to the columns the pipeline was fit on so a frame
        # carrying additional columns (e.g., a downstream join added
        # context) does not confuse the ColumnTransformer.
        feature_subset = bundle.frame[list(self.feature_columns)]

        # Apply preprocessing. The pipeline was fit during training;
        # ``transform`` (not ``fit_transform``) ensures we never refit
        # on inference data.
        X = self.feature_pipeline.transform(feature_subset)

        # Score with each component. ``predict_proba``'s second column
        # is the positive-class probability — the laundering probability.
        anomaly_scores = self.anomaly_scorer.score(X)
        supervised_proba = self.supervised_classifier.predict_proba(X)[:, 1]

        return self._combine_scores(anomaly_scores, supervised_proba)

    def predict(
        self, raw_frame: pd.DataFrame, *, threshold: float | None = None
    ) -> np.ndarray:
        """Return binary alert decisions.

        Parameters
        ----------
        raw_frame : pd.DataFrame
            Raw transactions.
        threshold : float | None
            Optional override of the configured decision threshold. Used
            by the API to support per-environment threshold tuning
            without retraining the model.

        Returns
        -------
        np.ndarray
            Binary array of ``0`` (no alert) and ``1`` (alert). Shape
            ``(n_rows,)``.
        """
        scores = self.score(raw_frame)
        effective_threshold = (
            threshold if threshold is not None else self.decision_threshold
        )
        return (scores >= effective_threshold).astype(np.int8)

    def _combine_scores(
        self, anomaly_scores: np.ndarray, supervised_scores: np.ndarray
    ) -> np.ndarray:
        """Weighted score-level ensemble combination.

        The combination is intentionally simple: a weighted convex sum.
        More elaborate schemes (stacking, rank-based fusion) were
        evaluated during model selection and did not outperform the
        weighted sum on the cost-weighted Precision@k objective once
        both components were independently calibrated.
        """
        return self.anomaly_weight * anomaly_scores + self.supervised_weight * supervised_scores

    def save(self, path: Path | str) -> None:
        """Serialise the ensemble to disk with joblib.

        Writes a single ``.pkl`` containing the ensemble plus its
        metadata. joblib is used (not pickle) because joblib handles
        numpy arrays more efficiently and ships memory-mapped reads,
        which matters when the API container cold-starts.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "service_version": __version__,
            "ensemble": self,
        }
        joblib.dump(payload, path, compress=("zlib", 3))

    @classmethod
    def load(cls, path: Path | str) -> "AMLEnsemble":
        """Deserialise an ensemble artifact with schema-version check.

        Raises ``ValueError`` on schema mismatch — operating against an
        artifact from a different code revision is a class of silent
        failure that has produced regulator-relevant incidents in the
        past, so we surface it loudly.
        """
        payload = joblib.load(path)
        embedded_schema = payload.get("schema_version")
        if embedded_schema != SCHEMA_VERSION:
            raise ValueError(
                f"Ensemble artifact at {path} declares schema version "
                f"{embedded_schema}, but this service runs schema "
                f"version {SCHEMA_VERSION}. Refusing to load."
            )

        ensemble = payload["ensemble"]
        if not isinstance(ensemble, cls):
            raise ValueError(
                f"Artifact at {path} did not contain an AMLEnsemble instance; "
                f"got {type(ensemble).__name__}."
            )
        return ensemble


def build_ensemble_from_components(
    *,
    bundle: FeatureBundle,
    fitted_pipeline: Pipeline,
    fitted_anomaly: AnomalyScorer,
    fitted_classifier: BaseEstimator,
    anomaly_weight: float,
    supervised_weight: float,
    decision_threshold: float,
    selected_family: str,
    selected_hyperparameters: dict[str, Any],
    training_data_rows: int,
    training_data_temporal_range: tuple[str, str],
    eval_metrics: dict[str, float],
) -> AMLEnsemble:
    """Assemble a production ensemble from its fitted components.

    Called by the training driver after the Optuna sweep selects the
    winning configuration. The function exists as a top-level
    constructor (rather than __init__) so the metadata payload can be
    assembled in one place — keeping construction declarative and the
    AMLEnsemble dataclass focused on the runtime scoring path.
    """
    metadata = EnsembleMetadata(
        service_version=__version__,
        schema_version=SCHEMA_VERSION,
        fit_timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        training_data_rows=training_data_rows,
        training_data_temporal_range=training_data_temporal_range,
        selected_classifier_family=selected_family,
        selected_classifier_hyperparameters=selected_hyperparameters,
        anomaly_metadata=fitted_anomaly.get_metadata(),
        classifier_metadata=get_classifier_info(fitted_classifier),
        ensemble_weights={
            "anomaly": anomaly_weight,
            "supervised": supervised_weight,
        },
        decision_threshold=decision_threshold,
        eval_metrics=eval_metrics,
    )

    return AMLEnsemble(
        feature_pipeline=fitted_pipeline,
        anomaly_scorer=fitted_anomaly,
        supervised_classifier=fitted_classifier,
        anomaly_weight=anomaly_weight,
        supervised_weight=supervised_weight,
        decision_threshold=decision_threshold,
        feature_columns=tuple(bundle.numerical_columns + bundle.categorical_columns),
        metadata=metadata,
    )
