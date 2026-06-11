"""Temporal train/validation/test splitting for AML data.

Why temporal and not random:

AML transaction data is a time series. Laundering typologies evolve as
adversaries adapt to detection. A random split mixes future transactions
into the training set, which leaks information about patterns the model
will face in production and inflates evaluation metrics. Every
production AML system splits by time. This module makes that choice
non-bypassable: the only public function returns sequential chronological
partitions and records the date boundaries for audit traceability.

The default split ratio (70/15/15) follows the convention used in the
IBM AML benchmark paper (Altman et al., 2023). Operators may override
the ratios for retraining cadences that prefer larger validation
windows, but the temporal ordering is fixed and not configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.data.loader import TIMESTAMP_COLUMN

# Default ratio used in published IBM AML benchmark experiments. Matching
# the literature default keeps any future comparison against published
# baselines methodologically apples-to-apples.
DEFAULT_TRAIN_FRACTION: Final[float] = 0.70
DEFAULT_VAL_FRACTION: Final[float] = 0.15
# Test fraction is implied (1 - train - val) but stored for documentation.
DEFAULT_TEST_FRACTION: Final[float] = 0.15


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Result of a temporal partition with full provenance.

    The boundaries are timestamps rather than indices so the split can
    be reproduced or audited against the original timeline regardless of
    how rows were ordered or filtered upstream. Frozen because a split
    record is an immutable historical fact about a training run.

    Attributes
    ----------
    train : pd.DataFrame
        Transactions strictly before ``val_start``.
    val : pd.DataFrame
        Transactions in [``val_start``, ``test_start``).
    test : pd.DataFrame
        Transactions at or after ``test_start``.
    val_start : pd.Timestamp
        Inclusive lower bound of the validation window.
    test_start : pd.Timestamp
        Inclusive lower bound of the test window.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: pd.Timestamp
    test_start: pd.Timestamp

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable summary for the audit log.

        Training runs persist this dict alongside the trial metrics so
        any model artifact can be traced back to the exact temporal
        partition that produced it. Includes class balance per split so
        downstream consumers can detect distributional drift in the
        labelled cohort over time.
        """
        from src.data.loader import LABEL_COLUMN

        return {
            "train_rows": int(len(self.train)),
            "val_rows": int(len(self.val)),
            "test_rows": int(len(self.test)),
            "val_start": self.val_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "train_positive_rate": float(self.train[LABEL_COLUMN].mean()),
            "val_positive_rate": float(self.val[LABEL_COLUMN].mean()),
            "test_positive_rate": float(self.test[LABEL_COLUMN].mean()),
        }


def temporal_train_val_test_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> TemporalSplit:
    """Partition the frame into chronological train/val/test windows.

    The function operates by sorting the frame by timestamp and selecting
    cut points at the requested cumulative fractions. The cut timestamps
    are then stored on the returned :class:`TemporalSplit` so the
    partition is reproducible even if the input is later resorted.

    Parameters
    ----------
    frame : pd.DataFrame
        The raw transaction frame. Must contain ``timestamp_column``.
    train_fraction : float
        Fraction of rows assigned to the training window. Must lie in
        the open interval (0, 1).
    val_fraction : float
        Fraction of rows assigned to the validation window. ``train +
        val`` must be strictly less than 1 so the test window is
        non-empty.
    timestamp_column : str
        Name of the timestamp column. Defaults to the dataset's canonical
        timestamp name; overridable for testing.

    Returns
    -------
    TemporalSplit
        The partition with provenance.

    Raises
    ------
    ValueError
        If the requested fractions are invalid, the frame is empty, or
        the timestamp column is missing.
    """
    # Validate inputs explicitly. Silent fallback to defaults on invalid
    # input is the antipattern that produces "the model trained but the
    # numbers look wrong" bug reports months later.
    if timestamp_column not in frame.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' not present in frame; "
            f"got columns: {list(frame.columns)}"
        )
    if len(frame) == 0:
        raise ValueError("Cannot split an empty frame.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be strictly < 1 so the test "
            f"window is non-empty; got {train_fraction + val_fraction}"
        )

    # Sort chronologically. We use stable sort so ties (same-timestamp
    # transactions) keep their original arrival order - important for
    # multi-leg laundering flows that share a wall-clock timestamp.
    ordered = frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)

    # Index-based slicing on a chronologically ordered frame is
    # equivalent to time-based slicing, but cheaper and more robust to
    # duplicate timestamps. The cut indices are computed once and reused.
    n_rows = len(ordered)
    train_end_idx = int(n_rows * train_fraction)
    val_end_idx = int(n_rows * (train_fraction + val_fraction))

    train = ordered.iloc[:train_end_idx].reset_index(drop=True)
    val = ordered.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
    test = ordered.iloc[val_end_idx:].reset_index(drop=True)

    # Record the boundary timestamps for audit traceability. We use the
    # first timestamp of each downstream window rather than the last of
    # the previous one to make boundary inclusion unambiguous: every
    # window is [start, next_start).
    val_start = pd.Timestamp(val[timestamp_column].iloc[0])
    test_start = pd.Timestamp(test[timestamp_column].iloc[0])

    return TemporalSplit(
        train=train,
        val=val,
        test=test,
        val_start=val_start,
        test_start=test_start,
    )
