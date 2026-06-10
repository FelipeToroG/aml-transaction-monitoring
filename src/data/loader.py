"""Schema and dataset loading for the IBM AML HI-Small dataset.

This module is the single source of truth for the raw schema. Every
component that touches transaction data - feature engineering, the
sklearn ColumnTransformer, the FastAPI request body, the database
model, the test fixtures - imports constants from this file. A column
rename here either propagates cleanly through the codebase or fails
loudly at import time. There is no other place schema lives.

The IBM AML HI-Small dataset (Altman et al., IBM Research, 2023)
contains ~5M synthetic banking transactions with multi-agent-simulated
laundering typologies and ground-truth illicit-flow labels. It is the
standard benchmark for AML detection research and is recognised by
hiring teams in financial-crime ML.

References:
    Altman, E., Egressy, B., Blanuša, J., & Atasu, K. (2023).
    Realistic Synthetic Financial Transactions for Anti-Money
    Laundering Models. https://arxiv.org/abs/2306.16424
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

# ---------------------------------------------------------------------------
# Raw schema constants - the single source of truth
# ---------------------------------------------------------------------------
# The exact column names emitted by the IBM AML HI-Small CSV. Preserved
# verbatim (including spaces) because the upstream dataset shipped these
# names; renaming them here would break reproducibility against the
# published benchmarks.

TIMESTAMP_COLUMN: Final[str] = "Timestamp"
SOURCE_BANK_COLUMN: Final[str] = "From Bank"
SOURCE_ACCOUNT_COLUMN: Final[str] = "Account"
DEST_BANK_COLUMN: Final[str] = "To Bank"
DEST_ACCOUNT_COLUMN: Final[str] = "Account.1"
AMOUNT_RECEIVED_COLUMN: Final[str] = "Amount Received"
RECEIVING_CURRENCY_COLUMN: Final[str] = "Receiving Currency"
AMOUNT_PAID_COLUMN: Final[str] = "Amount Paid"
PAYMENT_CURRENCY_COLUMN: Final[str] = "Payment Currency"
PAYMENT_FORMAT_COLUMN: Final[str] = "Payment Format"
LABEL_COLUMN: Final[str] = "Is Laundering"

# Numerical columns: continuous quantities the model treats as floats.
NUMERICAL_COLUMNS: Final[tuple[str, ...]] = (
    AMOUNT_RECEIVED_COLUMN,
    AMOUNT_PAID_COLUMN,
)

# Categorical columns: low-cardinality identifiers and discrete payment
# attributes. Bank and account identifiers are intentionally treated as
# categorical here because their cardinality is bounded by the dataset's
# entity universe and they carry signal about counterparty risk profiles.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    SOURCE_BANK_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    DEST_BANK_COLUMN,
    DEST_ACCOUNT_COLUMN,
    RECEIVING_CURRENCY_COLUMN,
    PAYMENT_CURRENCY_COLUMN,
    PAYMENT_FORMAT_COLUMN,
)

# The complete raw column set in the order the CSV emits them. Used by
# the loader to validate schema compliance on every read and by the test
# suite to assert the loader cannot silently accept truncated files.
RAW_COLUMNS: Final[tuple[str, ...]] = (
    TIMESTAMP_COLUMN,
    SOURCE_BANK_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    DEST_BANK_COLUMN,
    DEST_ACCOUNT_COLUMN,
    AMOUNT_RECEIVED_COLUMN,
    RECEIVING_CURRENCY_COLUMN,
    AMOUNT_PAID_COLUMN,
    PAYMENT_CURRENCY_COLUMN,
    PAYMENT_FORMAT_COLUMN,
    LABEL_COLUMN,
)

# Pandas dtype mapping for the raw load. Account identifiers are read as
# strings because the dataset uses hexadecimal-style codes that pandas
# would otherwise truncate when inferring numeric types. The label is
# read as int8 to minimise memory pressure on the ~5M-row HI-Small file.
RAW_DTYPES: Final[dict[str, str]] = {
    SOURCE_BANK_COLUMN: "int32",
    SOURCE_ACCOUNT_COLUMN: "string",
    DEST_BANK_COLUMN: "int32",
    DEST_ACCOUNT_COLUMN: "string",
    AMOUNT_RECEIVED_COLUMN: "float64",
    RECEIVING_CURRENCY_COLUMN: "category",
    AMOUNT_PAID_COLUMN: "float64",
    PAYMENT_CURRENCY_COLUMN: "category",
    PAYMENT_FORMAT_COLUMN: "category",
    LABEL_COLUMN: "int8",
}

# Default location for the raw CSV. The download script
# (`scripts/download_data.sh`) places the file here. The loader will
# accept any path the caller supplies, but the default keeps the
# notebooks and training driver path-free for one-command reproduction.
DEFAULT_RAW_PATH: Final[Path] = Path("data/raw/HI-Small_Trans.csv")


@dataclass
class SchemaValidationError(Exception):
    """Raised when a loaded frame does not match the expected raw schema.

    Subclassing the dataclass yields an immutable structured error that
    carries the offending columns rather than just a message string. This
    is what failing-loudly looks like for a data contract: the caller
    gets enough structured detail to remediate without grepping logs.

    Note: ``slots=True`` is deliberately omitted on this dataclass.
    Combining ``@dataclass(slots=True)`` with ``Exception`` produces a
    runtime ``super(type, obj)`` mismatch in CPython 3.11+ because the
    slotted decorator replaces the class while ``__post_init__`` still
    closes over the original. Exceptions are rare allocations, so the
    memory benefit of ``__slots__`` is not worth the breakage.
    """

    missing_columns: tuple[str, ...] = field(default_factory=tuple)
    unexpected_columns: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Call ``Exception.__init__`` directly rather than via ``super()``.
        # The dataclass-generated ``__init__`` does not invoke the parent
        # constructor automatically; explicitly seeding the message here
        # makes the exception render its structured fields when printed
        # or logged.
        Exception.__init__(
            self,
            f"Schema mismatch: missing={list(self.missing_columns)} "
            f"unexpected={list(self.unexpected_columns)}",
        )


class DataLoader:
    """Loader for the IBM AML HI-Small dataset.

    The loader handles three concerns:

    1. **Schema validation.** Every read verifies the column set against
       `RAW_COLUMNS`. A mismatch raises `SchemaValidationError` with the
       offending columns, so a corrupted or truncated CSV fails fast at
       ingest rather than producing silently malformed features later.

    2. **Dtype enforcement.** Pandas type inference is fast but can
       quietly downgrade numeric precision or read account IDs as
       float64 (truncating leading zeros). Explicit `dtype=` arguments
       prevent that.

    3. **Timestamp normalisation.** The dataset's timestamp column ships
       as an ISO-like string. The loader parses it to `pd.Timestamp` so
       downstream temporal feature engineering and the splitter can rely
       on a real datetime index.

    The class is deliberately thin: no caching, no filtering, no feature
    construction. Those concerns belong elsewhere. Keeping the loader
    minimal makes its responsibility unambiguous in code review.

    Examples
    --------
    >>> loader = DataLoader()
    >>> frame = loader.load()
    >>> frame.columns.tolist() == list(RAW_COLUMNS)
    True
    """

    def __init__(self, raw_path: Path | str | None = None) -> None:
        # Accept either a path-like or a string; coerce to Path so all
        # downstream operations use a uniform interface. None falls back
        # to the project-default location.
        self.raw_path: Path = Path(raw_path) if raw_path is not None else DEFAULT_RAW_PATH

    def load(self, *, nrows: int | None = None) -> pd.DataFrame:
        """Load the raw transaction frame.

        Parameters
        ----------
        nrows : int | None
            If provided, reads only the first ``nrows`` rows. Useful for
            smoke tests, EDA prototyping, and CI runs that should not
            burn ~5M rows of I/O. Production training never sets this.

        Returns
        -------
        pd.DataFrame
            The validated frame with enforced dtypes and a parsed
            timestamp column.

        Raises
        ------
        FileNotFoundError
            If the CSV is absent at the configured path. The error
            message points the operator at the download script rather
            than leaving them to grep documentation.
        SchemaValidationError
            If the loaded columns deviate from `RAW_COLUMNS`.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Raw dataset not found at {self.raw_path}. "
                "Run `bash scripts/download_data.sh` to fetch the IBM AML "
                "HI-Small CSV before invoking the loader."
            )

        frame = pd.read_csv(
            self.raw_path,
            dtype=RAW_DTYPES,
            parse_dates=[TIMESTAMP_COLUMN],
            nrows=nrows,
        )

        self._validate_schema(frame)
        return frame

    @staticmethod
    def _validate_schema(frame: pd.DataFrame) -> None:
        """Assert the frame's column set matches the expected schema.

        Comparison is order-insensitive on purpose: column order in the
        upstream CSV has changed between dataset releases in the past,
        and the rest of the pipeline references columns by name. The
        check that matters is set equality, not positional equality.
        """
        observed = set(frame.columns)
        expected = set(RAW_COLUMNS)

        missing = tuple(sorted(expected - observed))
        unexpected = tuple(sorted(observed - expected))

        if missing or unexpected:
            raise SchemaValidationError(
                missing_columns=missing,
                unexpected_columns=unexpected,
            )

    def load_sample(self, n: int = 10_000, *, random_state: int = 42) -> pd.DataFrame:
        """Load a deterministic stratified sample.

        Returns a sample preserving the laundering-class proportion. The
        sampler reads the full dataset and then stratifies because the
        upstream CSV is loosely time-sorted, and a naive `nrows` head
        would systematically over-represent early-period transactions.

        Parameters
        ----------
        n : int
            Approximate sample size. The actual count may differ by one
            row per class due to integer rounding in the stratification.
        random_state : int
            Seed for the sampler. Fixed by default for reproducibility.

        Returns
        -------
        pd.DataFrame
            A sample frame with the same schema as :meth:`load`.
        """
        frame = self.load()

        # Stratify by the label so the rare positive class is preserved
        # at its empirical frequency rather than getting sampled away.
        # The fraction-per-class is what `groupby + sample` does cleanly.
        class_fraction = n / len(frame)
        return (
            frame.groupby(LABEL_COLUMN, group_keys=False)
            .apply(
                lambda group: group.sample(
                    frac=min(class_fraction, 1.0),
                    random_state=random_state,
                )
            )
            .reset_index(drop=True)
        )
