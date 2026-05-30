"""Tests for the schema-validating data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import (
    CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
    NUMERICAL_COLUMNS,
    RAW_COLUMNS,
    DataLoader,
    SchemaValidationError,
)


def test_raw_columns_contains_label_and_timestamp():
    """The schema constant must include the label and timestamp columns."""
    assert LABEL_COLUMN in RAW_COLUMNS
    assert "Timestamp" in RAW_COLUMNS


def test_categorical_and_numerical_columns_disjoint():
    """A column may be either numerical or categorical, not both."""
    assert not set(CATEGORICAL_COLUMNS) & set(NUMERICAL_COLUMNS)


def test_loader_raises_when_file_missing(tmp_path):
    """The loader fails fast with an actionable message when the CSV is absent."""
    loader = DataLoader(raw_path=tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load()
    assert "download_data.sh" in str(exc_info.value)


def test_loader_rejects_unexpected_schema(tmp_path):
    """A frame with extra or missing columns triggers SchemaValidationError."""
    bad_path = tmp_path / "bad.csv"
    # Build a CSV missing the label column to trigger schema validation failure.
    frame = pd.DataFrame(
        {
            "Timestamp": ["2026-01-01"],
            "From Bank": [1],
            "Account": ["a"],
            "To Bank": [2],
            "Account.1": ["b"],
            "Amount Received": [100.0],
            "Receiving Currency": ["USD"],
            "Amount Paid": [100.0],
            "Payment Currency": ["USD"],
            "Payment Format": ["Wire"],
            # Note: "Is Laundering" intentionally omitted.
            "extra_column": ["x"],
        }
    )
    frame.to_csv(bad_path, index=False)

    loader = DataLoader(raw_path=bad_path)
    with pytest.raises(SchemaValidationError) as exc_info:
        loader.load()
    # The structured exception carries the diff.
    assert LABEL_COLUMN in exc_info.value.missing_columns
    assert "extra_column" in exc_info.value.unexpected_columns
