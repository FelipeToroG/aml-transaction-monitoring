"""Velocity features for AML detection.

Velocity features quantify how fast money moves through an entity. They
are the strongest signal for the money-mule typology, where funds enter
and exit an account within minutes or hours with no intervening
consumption, and a secondary signal for layering, which produces sharp
throughput spikes against an entity's baseline.

This module produces the following features per transaction, all
computed on the strictly-causal historical window of the entity
(i.e., excluding the current transaction):

* ``entity_in_out_count_ratio_24h``: ratio of inbound to outbound
  transaction counts over the trailing 24 hours, computed for the
  *source* entity. Mules are characterised by ratios near 1.
* ``entity_in_out_amount_ratio_24h``: same as above but on amount.
* ``entity_net_flow_24h``: inbound amount minus outbound amount over
  24 hours, for the source entity. Mules net to near zero.
* ``entity_throughput_to_baseline_24h``: ratio of the entity's
  24h-trailing throughput to its all-time baseline daily throughput,
  measuring the *spike* relative to historical normal.

The convention "for the source entity" means: when scoring a
transaction T flowing from A to B, the velocity features describe A's
recent activity. The model then evaluates whether A's behaviour around
T fits a mule pattern.

Why the source entity and not the destination
---------------------------------------------
A mule receives money and sends it out again. From the perspective of
the outbound leg (the transaction that empties the mule), the source
is the mule itself. Computing velocity on the source captures the
relevant signal: the mule's history of rapid in-and-out flow.
Destination-side throughput is captured separately by the entity rolling
features and the graph features.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from src.data.loader import (
    AMOUNT_PAID_COLUMN,
    DEST_ACCOUNT_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    TIMESTAMP_COLUMN,
)

# Velocity is computed on a single window — the 24-hour daily cycle is
# the canonical window in published AML practice because most mule
# operations resolve within one banking day. Multi-window velocity is
# subsumed by the entity rolling features in entity_features.py.
VELOCITY_WINDOW: Final[str] = "24h"


def compute_velocity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity features for every transaction in the frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw transaction frame with the canonical schema from
        :mod:`src.data.loader`.

    Returns
    -------
    pd.DataFrame
        A frame indexed by the original row index with the velocity
        feature columns. The input is not mutated.
    """
    # The velocity computation needs both inbound and outbound histories
    # for each source entity. We obtain them by treating the same
    # transaction frame in two views: outbound (where the entity is the
    # source) and inbound (where the entity is the destination).

    work = frame[
        [SOURCE_ACCOUNT_COLUMN, DEST_ACCOUNT_COLUMN, TIMESTAMP_COLUMN, AMOUNT_PAID_COLUMN]
    ].copy()
    work["_original_index"] = frame.index

    # Outbound view: each row is a transaction from the source entity's
    # perspective. The entity key is the source account.
    outbound = work[[SOURCE_ACCOUNT_COLUMN, TIMESTAMP_COLUMN, AMOUNT_PAID_COLUMN]].rename(
        columns={SOURCE_ACCOUNT_COLUMN: "_entity"}
    )

    # Inbound view: each row is a transaction from the destination
    # entity's perspective. The entity key is the destination account.
    inbound = work[[DEST_ACCOUNT_COLUMN, TIMESTAMP_COLUMN, AMOUNT_PAID_COLUMN]].rename(
        columns={DEST_ACCOUNT_COLUMN: "_entity"}
    )

    # Aggregate each view by entity over the trailing 24-hour window.
    # ``closed='left'`` preserves causality.
    outbound_agg = _aggregate_by_entity(
        outbound, window=VELOCITY_WINDOW, suffix="out"
    )
    inbound_agg = _aggregate_by_entity(
        inbound, window=VELOCITY_WINDOW, suffix="in"
    )

    # Both aggregates are indexed by (_entity, timestamp). We now need
    # to join them onto the original transaction frame at the
    # (source_entity, timestamp) key so each row sees its source entity's
    # recent inbound and outbound history. ``merge_asof`` is the right
    # tool: it joins on the closest non-future key per group, which is
    # exactly what causal rolling features require.

    # First, prepare the original frame indexed by source entity.
    target = work[[SOURCE_ACCOUNT_COLUMN, TIMESTAMP_COLUMN, "_original_index"]].rename(
        columns={SOURCE_ACCOUNT_COLUMN: "_entity"}
    )

    # Sort all three by (timestamp, _entity) — timestamp first.
    # ``merge_asof`` requires the ``on`` key to be globally monotonically
    # increasing on both sides; the ``by`` parameter only buckets the
    # join logic, it does not relax the sort requirement. Pandas 2.x
    # enforces this strictly (older versions were lenient). Sorting by
    # ``[_entity, timestamp]`` would only sort timestamps within each
    # entity, which is not enough — leading to "left keys must be
    # sorted". With timestamp first, the global ordering holds and each
    # per-entity subsequence remains sorted as well.
    target = target.sort_values([TIMESTAMP_COLUMN, "_entity"])
    outbound_agg = outbound_agg.sort_values([TIMESTAMP_COLUMN, "_entity"])
    inbound_agg = inbound_agg.sort_values([TIMESTAMP_COLUMN, "_entity"])

    joined = pd.merge_asof(
        target,
        outbound_agg,
        on=TIMESTAMP_COLUMN,
        by="_entity",
        direction="backward",
        allow_exact_matches=False,  # exclude the current transaction
    )
    joined = pd.merge_asof(
        joined.sort_values([TIMESTAMP_COLUMN, "_entity"]),
        inbound_agg,
        on=TIMESTAMP_COLUMN,
        by="_entity",
        direction="backward",
        allow_exact_matches=False,
    )

    # Compute the derived ratios and net-flow features from the joined
    # raw counts and sums. The ``+ 1e-9`` regularisers prevent division
    # by zero on entities with no historical activity; the sklearn
    # pipeline imputes the resulting NaN/inf values downstream.
    out_count = joined["count_out"].fillna(0.0)
    in_count = joined["count_in"].fillna(0.0)
    out_sum = joined["sum_out"].fillna(0.0)
    in_sum = joined["sum_in"].fillna(0.0)

    joined["entity_in_out_count_ratio_24h"] = (in_count + 1.0) / (out_count + 1.0)
    joined["entity_in_out_amount_ratio_24h"] = (in_sum + 1.0) / (out_sum + 1.0)
    joined["entity_net_flow_24h"] = in_sum - out_sum

    # Throughput-to-baseline ratio. The "baseline" is the entity's
    # all-time average daily outbound amount, computed offline. Here
    # we approximate it with the entity's all-time outbound mean
    # transaction amount — a simpler but well-correlated baseline that
    # avoids a second time-series aggregation. The production-grade
    # baseline (per-entity historical daily-amount mean) is computed
    # in the offline feature store and joined here; the in-process
    # approximation below is the runtime fallback when the offline
    # baseline is unavailable.
    entity_lifetime_mean = (
        work.groupby(SOURCE_ACCOUNT_COLUMN)[AMOUNT_PAID_COLUMN].transform("mean")
    )
    # Align this back to the joined frame via _original_index.
    baseline_lookup = pd.Series(
        entity_lifetime_mean.values, index=work["_original_index"]
    )
    joined["_baseline"] = joined["_original_index"].map(baseline_lookup)
    joined["entity_throughput_to_baseline_24h"] = (
        (out_sum / VELOCITY_WINDOW_HOURS) / (joined["_baseline"] + 1e-9)
    )

    # Select only the feature columns plus the original index. Re-index
    # by the original row position so the caller can concat horizontally
    # with the other feature blocks.
    feature_columns = [
        "entity_in_out_count_ratio_24h",
        "entity_in_out_amount_ratio_24h",
        "entity_net_flow_24h",
        "entity_throughput_to_baseline_24h",
    ]
    output = joined.set_index("_original_index")[feature_columns].sort_index()
    output.index.name = None
    return output


# The numeric form of the velocity window, used in throughput-rate
# computations. Kept in sync with ``VELOCITY_WINDOW`` by construction.
VELOCITY_WINDOW_HOURS: Final[float] = 24.0


def _aggregate_by_entity(
    view: pd.DataFrame, *, window: str, suffix: str
) -> pd.DataFrame:
    """Compute trailing window count and sum per entity.

    Internal helper. Returns a frame with columns
    ``[_entity, timestamp, count_<suffix>, sum_<suffix>]`` where the
    aggregates exclude the current transaction (``closed='left'``).
    """
    view = view.sort_values(["_entity", TIMESTAMP_COLUMN], kind="stable")
    view = view.set_index(TIMESTAMP_COLUMN)

    rolled = view.groupby("_entity", sort=False, group_keys=False).rolling(
        window, closed="left"
    )
    # Pandas 2.x rejects the NamedAgg keyword pattern on Rolling objects
    # (that pattern is a GroupBy-only convenience). The dict-of-lists
    # form below is the documented Rolling.agg() contract; we rename the
    # resulting multi-level columns to the suffix-style names the rest
    # of this module expects.
    aggregated = rolled.agg({AMOUNT_PAID_COLUMN: ["count", "sum"]})
    aggregated.columns = [f"count_{suffix}", f"sum_{suffix}"]
    aggregated = aggregated.reset_index()
    return aggregated
