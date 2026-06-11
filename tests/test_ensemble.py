"""Tests for the stacked AMLEnsemble: save/load round-trip and schema guard.

The ensemble no longer blends two scores; the Isolation Forest is a feature
inside the fitted pipeline and the supervised model emits the risk score
directly. These tests pin the serialised contract at schema v2 and confirm
the loader rejects a pre-stacking (v1) artifact.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.loader import LABEL_COLUMN
from src.features.pipelines import build_engineered_frame, build_feature_pipeline
from src.models.ensemble import (
    SCHEMA_VERSION,
    AMLEnsemble,
    build_ensemble_from_components,
)

_FAST_IF = {"n_estimators": 30, "random_state": 0}


def _build_tiny_ensemble(raw_frame) -> AMLEnsemble:
    """Assemble a minimal stacked ensemble on a synthetic frame."""
    bundle = build_engineered_frame(raw_frame)
    cols = list(bundle.numerical_columns + bundle.categorical_columns)
    pipeline = build_feature_pipeline(
        numerical_columns=bundle.numerical_columns,
        categorical_columns=bundle.categorical_columns,
        anomaly=_FAST_IF,
    )
    X = pipeline.fit_transform(bundle.frame[cols])
    y = raw_frame[LABEL_COLUMN].to_numpy()
    classifier = LogisticRegression(max_iter=2000).fit(X, y)
    anomaly_scorer = pipeline.named_steps["anomaly"].anomaly_scorer_

    return build_ensemble_from_components(
        bundle=bundle,
        fitted_pipeline=pipeline,
        fitted_anomaly=anomaly_scorer,
        fitted_classifier=classifier,
        decision_threshold=0.5,
        selected_family="logistic_regression",
        selected_hyperparameters={},
        training_data_rows=len(bundle.frame),
        training_data_temporal_range=("2026-01-01", "2026-02-01"),
        eval_metrics={"test_precision_at_k": 0.5},
    )


def test_ensemble_round_trip_schema_2(tmp_path, synthetic_transactions):
    """A stacked ensemble saves and loads, and scores through the pipeline."""
    ensemble = _build_tiny_ensemble(synthetic_transactions)
    path = tmp_path / "ensemble.pkl"
    ensemble.save(path)
    loaded = AMLEnsemble.load(path)

    assert SCHEMA_VERSION == 2
    assert loaded.metadata.schema_version == 2
    # The weighted-blend metadata field is gone under stacking.
    assert not hasattr(loaded.metadata, "ensemble_weights")

    risk = loaded.score(synthetic_transactions)
    assert risk.shape == (len(synthetic_transactions),)
    assert np.all((risk >= 0.0) & (risk <= 1.0))

    # score_components returns the same risk plus the anomaly feature column,
    # both aligned to the input rows.
    risk2, anomaly = loaded.score_components(synthetic_transactions)
    np.testing.assert_array_equal(risk, risk2)
    assert anomaly.shape == (len(synthetic_transactions),)

    preds = loaded.predict(synthetic_transactions)
    assert set(np.unique(preds)).issubset({0, 1})


def test_load_rejects_pre_stacking_schema(tmp_path, synthetic_transactions):
    """A v1-schema payload is refused loudly rather than mis-loaded."""
    ensemble = _build_tiny_ensemble(synthetic_transactions)
    path = tmp_path / "old.pkl"
    # Simulate a pre-stacking artifact by stamping schema_version=1.
    joblib.dump(
        {"schema_version": 1, "service_version": "0.0.0", "ensemble": ensemble},
        path,
    )

    with pytest.raises(ValueError, match="schema version"):
        AMLEnsemble.load(path)
