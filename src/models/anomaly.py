"""Isolation Forest anomaly scoring with calibrated output.

The unsupervised component of the AML hybrid ensemble. Isolation Forest
isolates each point through random axis-aligned partitions and uses the
average path length to isolation as a novelty score. The implementation
here wraps sklearn's ``IsolationForest`` with two additions necessary
for production use:

1. **Calibrated output**: sklearn's raw scores are unbounded real
   numbers where lower means more anomalous. We negate and percentile-
   normalise to a stable ``[0, 1]`` range where 1 means most anomalous,
   so the score composes cleanly with the supervised classifier's
   probability in the ensemble layer.

2. **Reproducibility metadata**: the fit captures the calibration
   percentiles, which are required for inference. The fitted object
   carries everything needed to reproduce a score given the same input
   — required for audit traceability of any alert the model produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(slots=True)
class AnomalyScorer:
    """Isolation Forest wrapper producing calibrated anomaly scores.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Higher values reduce variance at
        roughly linear cost.
    max_samples : int | float
        Subsample size used to build each tree, passed through to
        sklearn. An int is an absolute sample count, a float is a
        fraction of the training set. Prefer a small absolute count:
        Isolation Forest is designed for small subsamples (sklearn
        defaults to 256), which preserves the isolation property and
        keeps memory flat. A fraction of a multi-million-row fold both
        degrades detection and explodes memory (it OOM-killed the v1
        full train), so configs pass an absolute count here.
    contamination : float
        The expected positive (anomalous) rate. Used by Isolation Forest
        to set its internal decision boundary; we override the decision
        with our own calibrated thresholds downstream, so this parameter
        affects only the model's internal scoring scale.
    max_features : float
        Fraction of features sampled per tree. Sub-1.0 values are
        beneficial on the AML feature set because the long tail of
        per-window aggregates is correlated within each window.
    random_state : int
        Seed for reproducibility.

    Attributes (post-fit)
    ---------------------
    _calibration_low, _calibration_high : float
        Empirical 0.5th and 99.5th percentiles of the raw training-set
        anomaly scores. These define the linear ``[0, 1]`` mapping used
        at inference. The percentiles are stored on the instance and
        serialised with the model so inference reproduces fit-time
        calibration exactly.
    """

    n_estimators: int = 200
    max_samples: int | float = 256
    contamination: float = 0.01
    max_features: float = 1.0
    random_state: int = 42

    _model: IsolationForest | None = field(default=None, init=False, repr=False)
    _calibration_low: float = field(default=0.0, init=False, repr=False)
    _calibration_high: float = field(default=1.0, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # The underlying sklearn estimator is constructed once at init
        # time so the configured hyperparameters are visible to any
        # downstream inspection (e.g., experiment tracking dumps the
        # model's get_params at trial logging time).
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "AnomalyScorer":
        """Fit the forest and compute the calibration percentiles.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix. Shape ``(n_samples, n_features)``.

        Returns
        -------
        AnomalyScorer
            ``self`` for fluent chaining.
        """
        assert self._model is not None  # narrows for type checkers
        self._model.fit(X)

        # Calibration uses the 0.5th and 99.5th percentiles rather than
        # the absolute min/max. The wider quantile band is robust against
        # pathological outliers in the training set that would otherwise
        # collapse the [0, 1] mapping into a narrow band around the
        # outlier values, leaving the bulk of the inference distribution
        # squeezed into a tiny score range.
        raw_scores = self._raw_score(X)
        self._calibration_low = float(np.percentile(raw_scores, 0.5))
        self._calibration_high = float(np.percentile(raw_scores, 99.5))
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated anomaly scores in ``[0, 1]``.

        The output convention is ``1.0 = most anomalous, 0.0 = most
        normal``. Composes additively with the supervised classifier's
        probability in the ensemble layer.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to score. Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Calibrated scores. Shape ``(n_samples,)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "AnomalyScorer has not been fit. Call fit() before score()."
            )

        raw = self._raw_score(X)
        # Clip to the calibration range so values outside fit-time
        # extremes saturate cleanly at 0 or 1 rather than producing
        # uncontrolled out-of-range outputs.
        clipped = np.clip(raw, self._calibration_low, self._calibration_high)
        span = self._calibration_high - self._calibration_low

        if span <= 0:
            # Degenerate training data (zero-variance scores). Returning
            # a neutral 0.5 keeps the ensemble layer operational and
            # surfaces the issue via the get_metadata() output rather
            # than crashing the scoring path.
            return np.full(len(X), 0.5, dtype=np.float64)

        return (clipped - self._calibration_low) / span

    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores where 'higher = more anomalous'.

        sklearn's ``score_samples`` returns 'higher = more normal'.
        Negating produces the convention the rest of the system uses,
        eliminating an entire class of sign-error bugs at the boundary.
        """
        assert self._model is not None
        return -self._model.score_samples(X)

    def get_metadata(self) -> dict[str, Any]:
        """Return serialisable metadata for the audit log and registry.

        Embedded in the production model artifact so any alert can be
        reconstructed from the saved record.
        """
        return {
            "component": "anomaly",
            "type": "isolation_forest",
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "max_features": self.max_features,
                "random_state": self.random_state,
            },
            "calibration": {
                "low_percentile_0_5": self._calibration_low,
                "high_percentile_99_5": self._calibration_high,
                "fitted": self._fitted,
            },
        }
