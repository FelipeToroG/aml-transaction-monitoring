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
poorly calibrated probabilities by default — the predicted probabilities
do not match empirical positive rates. For cost-sensitive AML scoring
this matters: the ensemble layer combines anomaly scores and supervised
probabilities at the score level, and poorly calibrated probabilities
distort the combination. The factory wraps tree-based classifiers in
``CalibratedClassifierCV`` with isotonic regression when calibration is
requested.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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
        constructor, which raises ``TypeError`` — this is intentional
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
        # Imported lazily so the API container — which does not need
        # XGBoost's heavy C++ initialisation — can omit the dependency
        # if it ever drops the gradient-boosted families. The current
        # API does include xgboost in requirements-api.txt because the
        # production ensemble loads it.
        from xgboost import XGBClassifier

        # tree_method='hist' is the fastest available method on CPU
        # training and matches the documented behaviour for HI-Small.
        # n_jobs=-1 uses all available cores.
        classifier: BaseEstimator = XGBClassifier(
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            eval_metric="aucpr",  # match the CV selection metric
            **hyperparameters,
        )

    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        # LightGBM's verbose=-1 silences the per-iteration training log
        # which otherwise floods the Optuna trial output. The metric
        # parameter mirrors the XGBoost configuration for parity.
        classifier = LGBMClassifier(
            n_jobs=-1,
            random_state=random_state,
            metric="average_precision",
            verbose=-1,
            **hyperparameters,
        )

    elif family == "random_forest":
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
