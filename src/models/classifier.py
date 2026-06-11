"""Supervised classifier factory for the AML ensemble.

The supervised component of the hybrid ensemble. This module owns the
instantiation logic for every model family in the Optuna sweep, behind a
single ``build_classifier`` factory function. Centralising construction
behind one entry point keeps the training driver clean: Optuna trials
sample a family name and a hyperparameter dict, then dispatch to the
factory without needing to know each family's specific constructor
quirks.

Why a factory and not direct sklearn instantiation
--------------------------------------------------
Each family imposes its own conventions around the class-imbalance
parameter (``scale_pos_weight`` for XGBoost vs. ``class_weight`` for
sklearn-native), the random-state argument name (``random_state`` vs.
``seed``), and the way categorical features are consumed. The factory
absorbs those differences and presents a uniform interface to the
training driver.

The factory also handles probability calibration. Tree ensembles produce
poorly calibrated probabilities by default - the predicted probabilities
do not match empirical positive rates. For cost-sensitive AML scoring
this matters: the ensemble layer combines anomaly scores and supervised
probabilities at the score level, and poorly calibrated probabilities
distort the combination. The factory wraps tree-based classifiers in
``CalibratedClassifierCV`` with isotonic regression when calibration is
requested.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Final, Literal

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)


ClassifierFamily = Literal[
    "xgboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
]


# Families that benefit from explicit probability calibration. Logistic
# regression is intrinsically well-calibrated by construction and is
# excluded; gradient-boosted ensembles are systematically over-confident
# on the positive class in highly imbalanced data and are included.
_FAMILIES_REQUIRING_CALIBRATION: Final[frozenset[ClassifierFamily]] = frozenset(
    {"xgboost", "lightgbm", "random_forest"}
)


# Module-level cache for the one-shot GPU probe. ``None`` means "not yet
# probed"; the probe is expensive enough (it builds and trains a throwaway
# booster) that we only ever want to pay for it once per process. The
# AML_FORCE_CPU override is deliberately NOT folded into this cache: it is
# re-read on every call so a test or CI job can flip it without having to
# arrange for a fresh interpreter.
_GPU_PROBE_RESULT: bool | None = None

# Guards the device-selection INFO line so a full Optuna sweep, which
# constructs hundreds of XGBoost classifiers, logs the chosen device once
# rather than once per trial.
_DEVICE_LOG_EMITTED: bool = False

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _xgboost_gpu_available() -> bool:
    """Return whether XGBoost can train on a CUDA device on this machine.

    The result of the actual probe is cached at module level so the probe
    runs at most once per process. The ``AML_FORCE_CPU`` escape hatch is
    checked first and uncached: a truthy value forces CPU deterministically,
    which is how the test suite and CI pin device selection regardless of
    what hardware the runner happens to have.

    The probe doubles as the Blackwell safety net. The probe trains a tiny
    booster with ``device="cuda"`` inside a try/except and returns ``True``
    only if that succeeds. If the installed XGBoost build ships no compute
    kernel for this GPU's architecture (the RTX 5080 is sm_120, which older
    XGBoost wheels were not compiled for), the probe raises here instead of
    deep inside an Optuna trial, we log the reason and return ``False``, and
    training falls back to CPU rather than crashing mid-sweep. Logging the
    exception at INFO is what lets us tell "no GPU on this box" apart from
    "GPU present but unsupported by this XGBoost build".
    """
    if os.environ.get("AML_FORCE_CPU", "").strip().lower() in _TRUTHY:
        return False

    global _GPU_PROBE_RESULT
    if _GPU_PROBE_RESULT is None:
        _GPU_PROBE_RESULT = _probe_xgboost_gpu()
    return _GPU_PROBE_RESULT


def _probe_xgboost_gpu() -> bool:
    """Attempt a one-estimator CUDA fit; ``True`` iff it trains cleanly."""
    try:
        import numpy as np
        from xgboost import XGBClassifier

        # A handful of rows and a single tree: enough to force XGBoost to
        # actually dispatch to the CUDA kernel without paying real training
        # cost. Warnings (e.g. host-to-device data copies) are throwaway
        # noise for a probe of this size and are silenced so they do not
        # surface in CLI or test output.
        x = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            XGBClassifier(
                n_estimators=1,
                tree_method="hist",
                device="cuda",
                verbosity=0,
            ).fit(x, y)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "no usable GPU"
        # INFO, not WARNING: a CPU-only box is a perfectly normal, expected
        # configuration. The message carries the exception so an operator can
        # distinguish an absent GPU from an architecture-mismatch (sm_120).
        logger.info("XGBoost GPU probe failed; using CPU. Reason: %s", exc)
        return False


def _resolve_xgboost_device() -> str:
    """Pick ``"cuda"`` or ``"cpu"`` for XGBoost and log the choice once."""
    global _DEVICE_LOG_EMITTED
    device = "cuda" if _xgboost_gpu_available() else "cpu"
    if not _DEVICE_LOG_EMITTED:
        logger.info("XGBoost will train on device=%s", device)
        _DEVICE_LOG_EMITTED = True
    return device


def build_classifier(
    family: ClassifierFamily,
    *,
    hyperparameters: dict[str, Any],
    random_state: int = 42,
    calibrate: bool = False,
) -> BaseEstimator:
    """Construct a sklearn-compatible classifier for the given family.

    Parameters
    ----------
    family : ClassifierFamily
        One of ``xgboost``, ``lightgbm``, ``random_forest``,
        ``logistic_regression``.
    hyperparameters : dict[str, Any]
        Family-specific hyperparameters as sampled by Optuna. Keys must
        match the names in ``configs/model_config.yaml``. Unknown keys
        for a given family are passed through to the underlying
        constructor, which raises ``TypeError`` - this is intentional
        because a silently ignored hyperparameter would invalidate the
        Optuna sweep without surfacing an error.
    random_state : int
        Seed for reproducibility. Routed to the family-specific
        seed/random_state argument.
    calibrate : bool
        If ``True``, wrap the classifier in ``CalibratedClassifierCV``
        with isotonic regression. Recommended for the final selected
        model; not recommended during the Optuna sweep because
        calibration adds a 3x training cost.

    Returns
    -------
    BaseEstimator
        A fitted-by-the-caller sklearn-compatible classifier with
        ``fit``, ``predict``, and ``predict_proba``.

    Raises
    ------
    ValueError
        If ``family`` is not a recognised family name.
    """
    if family == "xgboost":
        # Imported lazily so the API container - which does not need
        # XGBoost's heavy C++ initialisation - can omit the dependency
        # if it ever drops the gradient-boosted families. The current
        # API does include xgboost in requirements-api.txt because the
        # production ensemble loads it.
        from xgboost import XGBClassifier

        # tree_method='hist' is the histogram algorithm XGBoost uses on both
        # CPU and CUDA; the device argument is what actually routes the fit
        # to the GPU. _resolve_xgboost_device() returns "cuda" only when a
        # one-shot probe confirmed this XGBoost build can train on the local
        # GPU, and "cpu" otherwise, so this line is safe on a CPU-only box
        # and on the Blackwell card alike. n_jobs=-1 uses all cores for the
        # CPU path (it is a no-op for the GPU path).
        classifier: BaseEstimator = XGBClassifier(
            tree_method="hist",
            device=_resolve_xgboost_device(),
            n_jobs=-1,
            random_state=random_state,
            eval_metric="aucpr",  # match the CV selection metric
            **hyperparameters,
        )

    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        # Deliberately CPU-only: unlike XGBoost, the pinned LightGBM wheel
        # ships no CUDA build (GPU support requires compiling from source),
        # so there is no device knob to flip here. LightGBM's verbose=-1
        # silences the per-iteration training log which otherwise floods the
        # Optuna trial output. The metric parameter mirrors the XGBoost
        # configuration for parity.
        classifier = LGBMClassifier(
            n_jobs=-1,
            random_state=random_state,
            metric="average_precision",
            verbose=-1,
            **hyperparameters,
        )

    elif family == "random_forest":
        # Deliberately CPU-only: scikit-learn's RandomForest has no GPU
        # backend at all, so only XGBoost participates in GPU acceleration.
        # Random Forest's class_weight default is None, which performs
        # poorly on imbalanced data. The configured hyperparameters
        # include class_weight, so we do not set a default here that
        # would override the Optuna choice.
        classifier = RandomForestClassifier(
            n_jobs=-1,
            random_state=random_state,
            **hyperparameters,
        )

    elif family == "logistic_regression":
        # An explicit max_iter is the convergence backstop. The search
        # space (configs/model_config.yaml) already removes the
        # solver=saga + penalty=l1 + low-C combination that hung the v0
        # sweep, so liblinear should converge well within this budget.
        # Bounding the iteration count guarantees that any residual hard
        # trial terminates with a ConvergenceWarning rather than running
        # unbounded: the default 100 is too low for the AML feature matrix
        # under L1, and an unbounded solver is what caused the v0 incident
        # (docs/INCIDENT_REPORT.md). 2000 is generous enough that a
        # warning here signals a genuinely difficult point worth ignoring
        # in selection, not a routine near-miss.
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            **hyperparameters,
        )

    else:
        raise ValueError(
            f"Unknown classifier family: {family!r}. "
            f"Valid: ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression']"
        )

    if calibrate and family in _FAMILIES_REQUIRING_CALIBRATION:
        # Isotonic calibration is non-parametric and dominant for
        # gradient-boosted ensembles on imbalanced data; Platt scaling
        # (sigmoid) underfits the characteristically S-shaped raw
        # probability distribution of these models. cv=3 balances
        # calibration quality against training cost.
        return CalibratedClassifierCV(
            estimator=classifier,
            method="isotonic",
            cv=3,
            n_jobs=-1,
        )

    return classifier


def get_classifier_info(estimator: ClassifierMixin) -> dict[str, Any]:
    """Return serialisable metadata about a fitted classifier.

    Used at ensemble-save time to embed component metadata in the
    serialised artifact. The audit trail for every alert references this
    metadata to reconstruct the exact model state that produced the
    score.
    """
    # If the classifier is wrapped in CalibratedClassifierCV the inner
    # estimator carries the actual hyperparameters. We unwrap one level.
    if isinstance(estimator, CalibratedClassifierCV):
        # After fitting, CalibratedClassifierCV exposes per-fold
        # calibrated classifiers; the first one's estimator is
        # representative of the configured base.
        base = estimator.estimator
        calibrated = True
    else:
        base = estimator
        calibrated = False

    return {
        "component": "supervised",
        "type": type(base).__name__,
        "module": type(base).__module__,
        "calibrated": calibrated,
        "hyperparameters": base.get_params() if hasattr(base, "get_params") else {},
    }
