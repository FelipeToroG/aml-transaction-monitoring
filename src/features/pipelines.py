"""sklearn pipeline composition for AML feature engineering.

This module wires the per-domain feature builders (entity, velocity,
graph) together with the sklearn preprocessing primitives into a single
end-to-end pipeline. The composed object is what the model trains on
and what the API serves predictions from — there is one pipeline shape
for both, eliminating train/serve skew.

Zero-leakage guarantee
----------------------
All preprocessing — scaling, encoding, imputation — happens inside the
sklearn ``Pipeline``. The pipeline is fit once per cross-validation
fold on the training half of that fold only. Validation and test
predictions go through the pipeline's ``transform``, not ``fit_transform``.
This places scaling parameters, encoder vocabularies, and imputed
values strictly downstream of the fold boundary. Leakage is not a
matter of discipline — it is structurally impossible.

Two-stage shape
---------------
Feature construction has two stages. Stage one (``build_engineered_frame``)
computes the entity, velocity, and graph features against the raw
transaction frame. These features are causal-windowed and therefore
safe to compute *before* the train/val/test split. Stage two
(``build_feature_pipeline``) is the sklearn ``Pipeline`` that the
training driver fits on the engineered frame, refitting per fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import (
    AMOUNT_PAID_COLUMN,
    AMOUNT_RECEIVED_COLUMN,
    PAYMENT_CURRENCY_COLUMN,
    PAYMENT_FORMAT_COLUMN,
    RECEIVING_CURRENCY_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.features.entity_features import compute_entity_features
from src.features.graph_features import compute_graph_features
from src.features.velocity_features import compute_velocity_features

# Low-cardinality categorical columns that one-hot encoding handles
# well. The account identifier columns are intentionally excluded:
# their cardinality is in the millions and one-hot would explode the
# feature space. The entity rolling features summarise the per-account
# behaviour, which is the actual signal.
CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    PAYMENT_FORMAT_COLUMN,
    RECEIVING_CURRENCY_COLUMN,
    PAYMENT_CURRENCY_COLUMN,
)

# Derived temporal features extracted from the timestamp. Hour-of-day
# and day-of-week capture the cyclic patterns in legitimate commerce
# that laundering activity tends to violate (e.g., 3 AM round-amount
# transactions are anomalous in a way 3 PM ones are not).
TEMPORAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
)


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Result of engineered-feature construction.

    Frozen because a feature bundle is an immutable view of one pass
    through the feature builders. Keeping the metadata (feature column
    lists) alongside the dataframe makes the bundle self-describing in
    downstream consumers — the training driver does not have to
    re-derive column lists from inspection.

    Attributes
    ----------
    frame : pd.DataFrame
        The full engineered frame, with raw columns plus all engineered
        features.
    numerical_columns : tuple[str, ...]
        Names of the numerical feature columns the model trains on.
    categorical_columns : tuple[str, ...]
        Names of the low-cardinality categorical feature columns the
        model trains on.
    """

    frame: pd.DataFrame
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def build_engineered_frame(frame: pd.DataFrame) -> FeatureBundle:
    """Compute the full engineered feature set for a raw frame.

    Calls each feature builder in the correct order, concatenates the
    resulting feature blocks back to the original frame, and returns a
    ``FeatureBundle`` carrying both the data and the column metadata.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw transaction frame with the canonical schema.

    Returns
    -------
    FeatureBundle
        The engineered frame plus the lists of numerical and categorical
        feature columns the sklearn pipeline will consume.
    """
    # Entity features for both roles. We compute them separately so the
    # column-name prefix scheme stays unambiguous (src_* vs dst_*).
    src_entity = compute_entity_features(frame, entity_role="source")
    dst_entity = compute_entity_features(frame, entity_role="destination")

    velocity = compute_velocity_features(frame)
    graph = compute_graph_features(frame)

    # Derived temporal features. These are cheap and align row-for-row
    # with the source frame, so they live here rather than in their own
    # module.
    temporal = _compute_temporal_features(frame)

    # Concatenate all feature blocks horizontally with the raw frame.
    # Every block is indexed by the original row index, so a plain
    # join-on-index is positional-safe.
    engineered = pd.concat(
        [frame, src_entity, dst_entity, velocity, graph, temporal],
        axis="columns",
    )

    numerical_columns = (
        AMOUNT_PAID_COLUMN,
        AMOUNT_RECEIVED_COLUMN,
        *tuple(src_entity.columns),
        *tuple(dst_entity.columns),
        *tuple(velocity.columns),
        *tuple(graph.columns),
        *TEMPORAL_FEATURE_COLUMNS,
    )

    return FeatureBundle(
        frame=engineered,
        numerical_columns=numerical_columns,
        categorical_columns=CATEGORICAL_FEATURE_COLUMNS,
    )


def build_feature_pipeline(
    *,
    numerical_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
) -> Pipeline:
    """Construct the sklearn preprocessing pipeline.

    The returned pipeline is a single object the training driver
    composes with the chosen classifier. The pipeline:

    1. Imputes numerical features with the column median (robust to
       skewed AML-feature distributions like amount sums).
    2. Scales numerical features with ``StandardScaler``. Gradient-
       boosted trees do not require scaling but the unsupervised
       Isolation Forest does, and one pipeline serves both heads of
       the ensemble.
    3. One-hot encodes the low-cardinality categorical features with
       ``handle_unknown='ignore'`` so a new payment-format value at
       runtime does not break the pipeline.

    Parameters
    ----------
    numerical_columns : tuple[str, ...]
        Names of numerical columns. Typically obtained from a
        ``FeatureBundle``.
    categorical_columns : tuple[str, ...]
        Names of categorical columns. Typically obtained from a
        ``FeatureBundle``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The composed preprocessing pipeline ready to be combined with
        a classifier in the training driver.
    """
    numerical_pipeline = Pipeline(
        steps=[
            # Median imputation is the right default for AML features:
            # most numerical features are skewed by a long upper tail
            # of high-volume entities, and median is robust to that
            # skew in a way mean is not.
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Categorical missing values are filled with a constant
            # sentinel so they become a distinct category. This is
            # informative — a missing payment_format is itself a signal
            # — and prevents the one-hot encoder from silently dropping
            # rows.
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    # Column selection: we pass explicit name tuples rather than using
    # sklearn.compose.make_column_selector(dtype_include=...) because
    # an AML model is audit-defensible only if the feature set is
    # traceable. Compliance officers and MRM reviewers reading the
    # ensemble metadata need to see "the model uses these N named
    # features," not "the model uses whatever happened to be numeric
    # in the input." Explicit enumeration also fails loudly when an
    # upstream schema change adds or removes a column, where dtype-
    # based selection would silently absorb it and produce a model
    # whose feature set has quietly drifted. The performance cost of
    # the explicit form vs make_column_selector is zero; the audit
    # benefit is meaningful.
    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, list(numerical_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        # Drop the raw identifier columns that we deliberately do not
        # use as features. Setting remainder='drop' is the safe default;
        # passthrough would silently include columns that should not
        # leak into the model.
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[("features", column_transformer)])


def _compute_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical and calendar features from the timestamp.

    Internal helper. The features are:

    * ``hour_of_day``: 0–23. Combined with ``is_night`` this captures
      "off-hours" anomalies.
    * ``day_of_week``: 0 (Monday) – 6 (Sunday). Combined with
      ``is_weekend`` this captures legitimate-commerce cyclicity.
    * ``is_weekend``: 1.0 on Saturday and Sunday.
    * ``is_night``: 1.0 between 22:00 and 06:00 local time of the
      timestamp.

    Returns a frame indexed by the original row index so it composes
    with the other feature blocks via positional concat.
    """
    ts = frame[TIMESTAMP_COLUMN]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek

    output = pd.DataFrame(
        {
            "hour_of_day": hour.astype(np.float32),
            "day_of_week": dow.astype(np.float32),
            "is_weekend": (dow >= 5).astype(np.float32),
            "is_night": ((hour >= 22) | (hour < 6)).astype(np.float32),
        },
        index=frame.index,
    )
    return output
