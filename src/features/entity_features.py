"""Entity-level rolling features for AML transaction scoring.

For each transaction, this module computes aggregations describing the
recent activity of the source and destination entities over multiple
time windows. These features are the heart of supervised AML detection:
laundering patterns deviate from per-entity behavioural baselines, so
the model learns to recognise the baselines themselves.

Zero-leakage construction
-------------------------
Every aggregation is computed on the half-open interval
``[transaction_timestamp - window, transaction_timestamp)``, strictly
excluding the current transaction. The same computation runs at training
time and at runtime serving time, eliminating train/serve skew without
requiring a separate offline/online feature pipeline.

Feature catalog (per entity role × window)
------------------------------------------
The function emits the following named features for each
``(role, window)`` combination, where ``role`` is one of ``src`` or
``dst`` and ``window`` is one of ``1h``, ``24h``, ``7d``:

* ``{role}_{window}_txn_count``: number of transactions
* ``{role}_{window}_amount_sum``: total amount paid
* ``{role}_{window}_amount_mean``: mean amount
* ``{role}_{window}_amount_std``: standard deviation of amounts
* ``{role}_{window}_amount_max``: maximum amount
* ``{role}_{window}_sub_threshold_count``: count of transactions below
  the US CTR threshold (USD 10,000), the structuring signal
* ``{role}_{window}_sub_threshold_share``: fraction of transactions
  below the threshold
* ``{role}_{window}_round_amount_count``: count of round-dollar
  transactions (multiples of 100 at or above 1,000), the round-amount
  anomaly signal
* ``{role}_{window}_round_amount_share``: fraction of round-dollar
  transactions

Plus a window-independent feature per role:

* ``{role}_dormancy_seconds``: time elapsed since this entity's previous
  transaction. Long dormancy followed by activity is the canonical
  integration signal.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd

from src.data.loader import (
    AMOUNT_PAID_COLUMN,
    DEST_ACCOUNT_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    TIMESTAMP_COLUMN,
)

# US Bank Secrecy Act Currency Transaction Report threshold. Transactions
# below this value avoid mandatory reporting and are therefore the
# canonical structuring target. The threshold is fixed in 31 CFR
# §1010.311 and has not moved since enactment.
STRUCTURING_THRESHOLD_USD: Final[float] = 10_000.0

# Heuristic for a "round" amount. We require the amount to be a multiple
# of 100 and at least 1,000 because legitimate commerce produces many
# naturally round small amounts (USD 100 lunch, USD 500 rent share) but
# few naturally round large ones. The 1,000 floor concentrates the
# signal on amounts where roundness genuinely indicates intent.
ROUND_AMOUNT_FLOOR_USD: Final[float] = 1_000.0
ROUND_AMOUNT_MODULUS_USD: Final[float] = 100.0

# Time windows used for rolling aggregations. Multiple windows let the
# model learn fast-moving signals (1h: rapid in-out, money-mule pattern)
# and slow-moving signals (7d: integration via dormant-then-active
# accounts) simultaneously.
DEFAULT_TIME_WINDOWS: Final[tuple[str, ...]] = ("1h", "24h", "7d")


def compute_entity_features(
    frame: pd.DataFrame,
    *,
    entity_role: Literal["source", "destination"] = "source",
    time_windows: tuple[str, ...] = DEFAULT_TIME_WINDOWS,
) -> pd.DataFrame:
    """Compute rolling per-entity features for one entity role.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw transaction frame with the canonical schema from
        :mod:`src.data.loader`. Must contain the timestamp column, the
        amount column, and the entity column for the requested role.
    entity_role : {'source', 'destination'}
        Whether to compute features for the source or destination entity
        of each transaction. The function is called twice in production
 - once for each role - and the two output frames are joined to
        produce the full feature set.
    time_windows : tuple[str, ...]
        Pandas-style window offset strings (e.g., ``"1h"``, ``"24h"``,
        ``"7d"``). Each window contributes its own feature set.

    Returns
    -------
    pd.DataFrame
        A frame aligned with the input by the original row index,
        containing the new feature columns named per the catalog in the
        module docstring. The input frame is not mutated.
    """
    if entity_role == "source":
        entity_column = SOURCE_ACCOUNT_COLUMN
        prefix = "src"
    elif entity_role == "destination":
        entity_column = DEST_ACCOUNT_COLUMN
        prefix = "dst"
    else:
        raise ValueError(
            f"entity_role must be 'source' or 'destination'; got {entity_role!r}"
        )

    # Work on a minimal projection so the intermediate frames stay small
    # in memory. The function emits features keyed back to the original
    # row index, so we explicitly preserve it on the working frame.
    work = frame[[entity_column, TIMESTAMP_COLUMN, AMOUNT_PAID_COLUMN]].copy()
    work["_original_index"] = frame.index

    # Add the derived per-transaction boolean signals before windowing.
    # These become count-able 0/1 columns inside the rolling window so
    # we get sub-threshold and round-amount counts essentially for free
    # via the same .sum() aggregation we use for total amount.
    work["_is_sub_threshold"] = (
        work[AMOUNT_PAID_COLUMN] < STRUCTURING_THRESHOLD_USD
    ).astype(np.int8)
    work["_is_round_amount"] = (
        (work[AMOUNT_PAID_COLUMN] >= ROUND_AMOUNT_FLOOR_USD)
        & (work[AMOUNT_PAID_COLUMN] % ROUND_AMOUNT_MODULUS_USD == 0.0)
    ).astype(np.int8)

    # Sort by (entity, timestamp) so groupby + rolling produce causal
    # windows per entity. Stable sort preserves arrival order on
    # same-timestamp transactions, which matters for chained laundering
    # flows that share a wall-clock timestamp.
    work = work.sort_values([entity_column, TIMESTAMP_COLUMN], kind="stable")

    # Time-based rolling needs the timestamp as the index of the
    # underlying frame. We index after sort so the rolling operator
    # sees the canonical ordering.
    work = work.set_index(TIMESTAMP_COLUMN)

    feature_blocks: list[pd.DataFrame] = []

    for window in time_windows:
        feature_blocks.append(
            _compute_window_aggregates(
                work=work,
                entity_column=entity_column,
                window=window,
                prefix=prefix,
            )
        )

    dormancy = _compute_dormancy(
        work=work,
        entity_column=entity_column,
        prefix=prefix,
    )

    # Concatenate all window blocks plus dormancy along columns. Every
    # block is indexed by the original row index so the join is
    # positional-safe and order-preserving.
    features = pd.concat([*feature_blocks, dormancy], axis="columns")

    # Restore the original row ordering so the returned features align
    # with ``frame.index`` by simple positional alignment.
    features = features.sort_index()

    # ``features`` is now indexed by the original row index of the
    # input frame. Drop the index name to keep downstream concat clean.
    features.index.name = None

    return features


def _compute_window_aggregates(
    *,
    work: pd.DataFrame,
    entity_column: str,
    window: str,
    prefix: str,
) -> pd.DataFrame:
    """Compute the per-window aggregation block for one entity role.

    Internal helper. Returns a frame indexed by ``_original_index`` and
    carrying every feature for the single window passed in.

    The aggregation strategy uses pandas' ``groupby(...).rolling(...)``
    with ``closed='left'`` so the current transaction is excluded from
    its own window. This is what enforces causality and prevents the
    model from peeking at the very transaction it is scoring.
    """
    grouped = work.groupby(entity_column, sort=False, group_keys=False)
    rolling = grouped.rolling(window, closed="left")

    # Compute every window aggregation in a single .agg() call so pandas
    # can share the underlying sliding-window computation across the
    # output columns. Pandas 2.x rejects the NamedAgg keyword pattern on
    # Rolling objects (that pattern is GroupBy-only); the dict-of-lists
    # form below is the documented Rolling.agg() contract. We rename the
    # resulting multi-level columns to flat, prefixed names immediately
    # so the rest of the function reads the same as before.
    aggregated = rolling.agg(
        {
            AMOUNT_PAID_COLUMN: ["count", "sum", "mean", "std", "max"],
            "_is_sub_threshold": ["sum"],
            "_is_round_amount": ["sum"],
        }
    )
    aggregated.columns = [
        f"{prefix}_{window}_txn_count",
        f"{prefix}_{window}_amount_sum",
        f"{prefix}_{window}_amount_mean",
        f"{prefix}_{window}_amount_std",
        f"{prefix}_{window}_amount_max",
        f"{prefix}_{window}_sub_threshold_count",
        f"{prefix}_{window}_round_amount_count",
    ]

    # The aggregated frame has a hierarchical index of
    # (entity, timestamp). Reset so we can attach the original-row
    # index column and produce a clean positional join key.
    aggregated = aggregated.reset_index(drop=False)

    # Bring the original-row index back from the work frame via a
    # positional join. The aggregated frame is the same length and in
    # the same row order as ``work`` (the rolling aggregation preserves
    # order within each group), so a column assignment from the values
    # array is equivalent to a merge on ``(_entity, Timestamp)`` and
    # avoids the join overhead.
    aggregated["_original_index"] = work["_original_index"].values

    # Derive the share features from the counts now that we have them.
    # The .replace + divide pattern avoids divide-by-zero noise on
    # entities that have no historical activity in the window.
    txn_count = aggregated[f"{prefix}_{window}_txn_count"]
    safe_denominator = txn_count.replace(0, np.nan)

    aggregated[f"{prefix}_{window}_sub_threshold_share"] = (
        aggregated[f"{prefix}_{window}_sub_threshold_count"] / safe_denominator
    ).fillna(0.0)
    aggregated[f"{prefix}_{window}_round_amount_share"] = (
        aggregated[f"{prefix}_{window}_round_amount_count"] / safe_denominator
    ).fillna(0.0)

    # Index by the original row position so the caller can concat the
    # window blocks horizontally without worrying about row alignment.
    aggregated = aggregated.set_index("_original_index")

    # Drop the join columns now that the index carries the alignment.
    aggregated = aggregated.drop(columns=[entity_column, TIMESTAMP_COLUMN])

    # First-transactions-by-entity have NaN counts (no prior history).
    # Fill the counts with zero - there genuinely was no activity. Mean,
    # std, and max remain NaN because their values are undefined on an
    # empty window; the sklearn pipeline imputes them downstream.
    count_columns = [
        f"{prefix}_{window}_txn_count",
        f"{prefix}_{window}_amount_sum",
        f"{prefix}_{window}_sub_threshold_count",
        f"{prefix}_{window}_round_amount_count",
    ]
    aggregated[count_columns] = aggregated[count_columns].fillna(0.0)

    return aggregated


def _compute_dormancy(
    *,
    work: pd.DataFrame,
    entity_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Compute the per-entity dormancy feature.

    Dormancy is the number of seconds elapsed since the entity's
    previous transaction. The first transaction for any entity has no
    "previous" and gets NaN, which the sklearn pipeline imputes with
    the column maximum so a never-before-seen entity is treated as
    maximally dormant (the prior for an unknown entity is "long quiet").
    """
    # ``work`` is already sorted by (entity, timestamp) from the caller,
    # so a simple groupby-shift gives the previous timestamp per entity.
    timestamps_as_index = work.index.to_series()
    previous_timestamp = timestamps_as_index.groupby(work[entity_column]).shift(1)

    elapsed = (timestamps_as_index - previous_timestamp).dt.total_seconds()
    elapsed.name = f"{prefix}_dormancy_seconds"

    # Restore the original row index so the dormancy block aligns with
    # the per-window blocks for the final concat.
    out = elapsed.to_frame()
    out["_original_index"] = work["_original_index"].values
    out = out.set_index("_original_index")
    return out
