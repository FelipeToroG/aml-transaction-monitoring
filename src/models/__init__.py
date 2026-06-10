"""Modeling layer: anomaly detection, supervised classification, ensemble.

Three model components live here and are composed at training time:

* ``anomaly`` - an Isolation Forest wrapper that produces a calibrated
  anomaly score in [0, 1]. The unsupervised component catches novel
  laundering typologies that did not appear in the training labels.
* ``classifier`` - a gradient-boosted supervised classifier (XGBoost is
  the default winner; LightGBM and Random Forest are evaluated for
  comparison). Produces a calibrated probability in [0, 1].
* ``ensemble`` - the production scoring object. Takes the two component
  scores and produces a single risk score using weights tuned during
  training to the cost-weighted Precision@k objective.

The training driver (``train.py``) runs the Optuna sweep across all
families, selects the best component models on cost-weighted Precision@k,
fits the ensemble layer, and serialises the production object to
``models/ensemble.pkl``. The API loads that single artifact at startup
and routes every incoming transaction through it.
"""
