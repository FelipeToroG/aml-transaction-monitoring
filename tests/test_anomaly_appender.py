"""Tests for the AnomalyScoreAppender stacking transformer.

The appender fits an Isolation Forest in ``fit`` and appends its calibrated
score as one trailing column in ``transform``. These tests pin the three
properties the stacking design depends on: it adds exactly one column, it
never refits on ``transform``, and the only fitted state used at transform
time is the train-fit forest plus its train-fit calibration band (the
zero-leakage guarantee). The transformer is purely additive in this commit;
nothing wires it into training, the ensemble, or the API yet.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.pipelines import (
    ANOMALY_FEATURE_NAME,
    AnomalyScoreAppender,
    build_engineered_frame,
    build_feature_pipeline,
)

# Small forests keep these tests fast; the behaviour under test does not
# depend on tree count.
_FAST_IF = {"n_estimators": 50, "random_state": 0}


def test_appender_appends_exactly_one_column():
    """fit_transform adds one trailing column and leaves the rest intact."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))

    out = AnomalyScoreAppender(anomaly_hyperparameters=_FAST_IF).fit_transform(X)

    assert out.shape == (200, 7)
    # The existing columns pass through untouched and in order.
    np.testing.assert_array_equal(out[:, :-1], X)


def test_transform_does_not_refit():
    """transform only scores: fitted forest, calibration, and output are stable."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5))

    appender = AnomalyScoreAppender(anomaly_hyperparameters=_FAST_IF).fit(X)
    scorer_identity = id(appender.anomaly_scorer_)
    low = appender.anomaly_scorer_._calibration_low
    high = appender.anomaly_scorer_._calibration_high

    out1 = appender.transform(X)
    out2 = appender.transform(X)

    assert id(appender.anomaly_scorer_) == scorer_identity
    assert appender.anomaly_scorer_._calibration_low == low
    assert appender.anomaly_scorer_._calibration_high == high
    np.testing.assert_array_equal(out1, out2)


def test_transform_before_fit_raises():
    """Scoring before fit is a NotFittedError, not a silent wrong answer."""
    appender = AnomalyScoreAppender()
    with pytest.raises(NotFittedError):
        appender.transform(np.zeros((3, 4)))


def test_no_leakage_calibration_frozen_at_fit():
    """transform uses only train-fit state; val statistics never leak in.

    Fits on a clean train distribution, then transforms a deliberately
    shifted val set. Two assertions establish no leakage: the calibration
    band is unchanged by scoring val, and the appended val column equals
    re-scoring the val matrix with the frozen train-fit forest.
    """
    rng = np.random.default_rng(2)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(300, 4))
    X_val = rng.normal(loc=5.0, scale=3.0, size=(120, 4))  # shifted on purpose

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("anomaly", AnomalyScoreAppender(anomaly_hyperparameters=_FAST_IF)),
        ]
    )
    pipe.fit(X_train)
    appender = pipe.named_steps["anomaly"]
    low = appender.anomaly_scorer_._calibration_low
    high = appender.anomaly_scorer_._calibration_high

    out_val = pipe.transform(X_val)

    # Scoring val did not move the calibration band fit on train.
    assert appender.anomaly_scorer_._calibration_low == low
    assert appender.anomaly_scorer_._calibration_high == high

    # The appended column is exactly the train-fit forest scoring the
    # train-fit-scaled val matrix: the only fitted state in play is from fit.
    scaled_val = pipe.named_steps["scaler"].transform(X_val)
    expected = appender.anomaly_scorer_.score(scaled_val)
    np.testing.assert_array_equal(out_val[:, -1], expected)
    assert out_val.shape == (120, 5)


def test_get_feature_names_out_appends_anomaly_name():
    """The audit feature-name list carries the anomaly column last."""
    X = np.random.default_rng(3).normal(size=(50, 3))
    appender = AnomalyScoreAppender(anomaly_hyperparameters=_FAST_IF).fit(X)

    names = list(appender.get_feature_names_out(["a", "b", "c"]))

    assert names[:-1] == ["a", "b", "c"]
    assert names[-1] == ANOMALY_FEATURE_NAME


def test_build_feature_pipeline_anomaly_is_default_off(synthetic_transactions):
    """Default-off matches prior behaviour; anomaly=... adds exactly one column."""
    bundle = build_engineered_frame(synthetic_transactions)
    cols = list(bundle.numerical_columns + bundle.categorical_columns)

    base = build_feature_pipeline(
        numerical_columns=bundle.numerical_columns,
        categorical_columns=bundle.categorical_columns,
    )
    X_base = base.fit_transform(bundle.frame[cols])

    stacked = build_feature_pipeline(
        numerical_columns=bundle.numerical_columns,
        categorical_columns=bundle.categorical_columns,
        anomaly=_FAST_IF,
    )
    X_stacked = stacked.fit_transform(bundle.frame[cols])

    assert "anomaly" not in base.named_steps
    assert "anomaly" in stacked.named_steps
    assert X_stacked.shape[1] == X_base.shape[1] + 1
