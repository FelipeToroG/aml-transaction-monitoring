"""Tests for the feature engineering pipeline."""

from __future__ import annotations

from src.features.entity_features import compute_entity_features
from src.features.pipelines import build_engineered_frame, build_feature_pipeline


def test_entity_features_produce_named_columns(synthetic_transactions):
    """Entity features have the expected naming pattern."""
    features = compute_entity_features(synthetic_transactions, entity_role="source")
    expected_substrings = ["src_1h_", "src_24h_", "src_7d_", "src_dormancy_"]
    for substring in expected_substrings:
        assert any(substring in col for col in features.columns), substring


def test_entity_features_count_columns_have_no_nan(synthetic_transactions):
    """Count features are filled with 0 for first-transaction-per-entity rows."""
    features = compute_entity_features(synthetic_transactions, entity_role="source")
    count_cols = [c for c in features.columns if c.endswith("_txn_count")]
    for col in count_cols:
        assert features[col].isna().sum() == 0, col


def test_engineered_frame_includes_temporal_features(synthetic_transactions):
    """The bundle includes the cyclical / calendar features."""
    bundle = build_engineered_frame(synthetic_transactions)
    for feature in ("hour_of_day", "day_of_week", "is_weekend", "is_night"):
        assert feature in bundle.frame.columns


def test_pipeline_constructs_without_error(synthetic_transactions):
    """The sklearn Pipeline composes and yields a transformable object."""
    bundle = build_engineered_frame(synthetic_transactions)
    pipeline = build_feature_pipeline(
        numerical_columns=bundle.numerical_columns,
        categorical_columns=bundle.categorical_columns,
    )
    feature_cols = list(bundle.numerical_columns + bundle.categorical_columns)
    X = pipeline.fit_transform(bundle.frame[feature_cols])
    assert X.shape[0] == len(bundle.frame)
    assert X.shape[1] > 0
