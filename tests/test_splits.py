"""Tests for the temporal split helper."""

from __future__ import annotations

import pytest

from src.data.splits import temporal_train_val_test_split


def test_split_is_chronological(synthetic_transactions):
    """train < val < test by timestamp."""
    split = temporal_train_val_test_split(synthetic_transactions)
    assert split.train["Timestamp"].max() <= split.val_start
    assert split.val["Timestamp"].max() <= split.test_start
    assert split.test["Timestamp"].min() >= split.test_start


def test_split_describe_includes_boundaries(synthetic_transactions):
    """The audit-traceability dict carries the boundary timestamps."""
    split = temporal_train_val_test_split(synthetic_transactions)
    described = split.describe()
    assert "val_start" in described
    assert "test_start" in described
    assert described["train_rows"] + described["val_rows"] + described["test_rows"] == len(
        synthetic_transactions
    )


def test_split_rejects_invalid_fractions(synthetic_transactions):
    """Fractions outside (0, 1) or summing to >= 1 are rejected with ValueError."""
    with pytest.raises(ValueError):
        temporal_train_val_test_split(synthetic_transactions, train_fraction=1.5)

    with pytest.raises(ValueError):
        temporal_train_val_test_split(
            synthetic_transactions, train_fraction=0.6, val_fraction=0.5
        )


def test_split_rejects_empty_frame(synthetic_transactions):
    """Empty input is a hard error rather than a silent empty-split."""
    empty = synthetic_transactions.iloc[0:0]
    with pytest.raises(ValueError):
        temporal_train_val_test_split(empty)
