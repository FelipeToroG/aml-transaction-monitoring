"""Data layer for the AML monitoring service.

This package owns three concerns that are deliberately co-located:

1. **Schema** (`loader`): the single source of truth for raw column
   names, dtypes, categorical-versus-numerical classification, and the
   target label. Every downstream module — the feature pipeline, the API
   request schema, the database models, the test suite — imports from
   here. A rename propagates from one place.

2. **Splits** (`splits`): temporal train/validation/test partitioning.
   AML data is a time series; random splits leak information about
   future laundering patterns into the training set. The splitter
   enforces strict chronological ordering and records the date
   boundaries for audit reproducibility.

3. **Typologies** (`typologies`): the named-pattern catalog used by
   feature engineering, the narrator, and the investigator UI. Centralising
   the typology vocabulary prevents drift between what the model detects,
   what the narrator describes, and what the investigator sees on screen.
"""

from src.data.loader import (
    CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
    NUMERICAL_COLUMNS,
    RAW_COLUMNS,
    TIMESTAMP_COLUMN,
    DataLoader,
)
from src.data.splits import TemporalSplit, temporal_train_val_test_split
from src.data.typologies import TYPOLOGIES, Typology

__all__ = [
    "CATEGORICAL_COLUMNS",
    "LABEL_COLUMN",
    "NUMERICAL_COLUMNS",
    "RAW_COLUMNS",
    "TIMESTAMP_COLUMN",
    "TYPOLOGIES",
    "DataLoader",
    "TemporalSplit",
    "Typology",
    "temporal_train_val_test_split",
]
