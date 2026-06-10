"""Feature engineering for AML transaction scoring.

The package is partitioned into single-responsibility modules so that
each feature family can be reasoned about and tested in isolation:

* ``entity_features`` - per-entity rolling aggregates (counts, amounts,
  uniqueness measures) computed over multiple time windows.
* ``velocity_features`` - short-window throughput and in-out ratios that
  detect rapid laundering flows.
* ``graph_features`` - NetworkX-based structural features on the
  transaction graph (degree, PageRank, edge novelty).
* ``pipelines`` - sklearn ``Pipeline`` and ``ColumnTransformer``
  composition that wires everything together with zero-leakage
  preprocessing.

The composed pipeline is exposed via ``build_feature_pipeline`` and is
the only object the rest of the codebase should import. Direct imports
of individual feature functions are reserved for the test suite.
"""
