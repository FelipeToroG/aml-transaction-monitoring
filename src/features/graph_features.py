"""Graph-structural features for AML transaction scoring.

Layering and smurfing typologies leave their fingerprint on the
transaction graph rather than on any single entity's activity. An
account that participates in many short paths, that suddenly acquires
inbound edges from many unrelated sources, or that sits on a previously
unseen counterparty pair, exhibits structural patterns invisible to
per-entity rolling features. This module extracts those signals.

Computational strategy
----------------------
Building a NetworkX graph for every transaction in a 5M-row dataset is
prohibitive - that would be O(n) graph constructions plus per-graph
centrality computations. Instead, the implementation buckets
transactions into rolling 24-hour windows and constructs a graph once
per bucket. For each transaction we look up its source-entity and
destination-entity features against the most recent bucket *strictly
before* the transaction's timestamp. This preserves causality (the
bucket the transaction is scored against does not contain the
transaction itself) while keeping the total graph-construction cost
linear in the number of buckets, not the number of transactions.

Runtime serving note
--------------------
At runtime the equivalent computation maintains a rolling 24-hour
graph in memory, updated incrementally as transactions arrive. The
training-time batch implementation here produces feature semantics
identical to that runtime path; the rolling-update version is the
production-readiness milestone called out in the roadmap of the README.

Feature catalog
---------------
For each transaction, this module emits the following features (both
the source entity ``src`` and destination entity ``dst`` variants):

* ``{role}_24h_degree_in``: number of distinct counterparties the entity
  received from in the trailing 24h window
* ``{role}_24h_degree_out``: number of distinct counterparties the
  entity sent to in the trailing 24h
* ``{role}_24h_pagerank``: PageRank centrality in the 24h subgraph, a
  proxy for the entity's position in laundering chains
* ``{role}_24h_clustering``: local clustering coefficient, high values
  indicate participation in tight triangle structures

Plus a transaction-level edge feature:

* ``edge_novelty_24h``: 1.0 if the (source, destination) ordered pair
  is absent from the trailing 24h window, 0.0 otherwise. Novel edges
  are over-represented in laundering.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import numpy as np
import pandas as pd

from src.data.loader import (
    DEST_ACCOUNT_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    TIMESTAMP_COLUMN,
)

# Window over which the rolling graph is constructed. The 24-hour value
# matches the velocity window so graph and velocity features share a
# coherent temporal frame of reference for any interpretation crossing
# the two.
GRAPH_WINDOW: Final[pd.Timedelta] = pd.Timedelta("24h")

# Bucket granularity. The graph is rebuilt once per bucket, and every
# transaction in a bucket looks up its features against the *previous*
# bucket. Smaller buckets produce more graphs (more cost) and less
# staleness in the per-transaction features; larger buckets produce
# fewer graphs (less cost) and more staleness. One hour balances the
# two for the HI-Small dataset's transaction density.
GRAPH_BUCKET: Final[pd.Timedelta] = pd.Timedelta("1h")


def compute_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute graph-structural features for every transaction.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw transaction frame with the canonical schema from
        :mod:`src.data.loader`.

    Returns
    -------
    pd.DataFrame
        A frame indexed by the original row index with the graph
        feature columns. Transactions in the first bucket (no prior
        history exists) receive NaN values, imputed downstream by the
        sklearn pipeline.
    """
    work = frame[
        [SOURCE_ACCOUNT_COLUMN, DEST_ACCOUNT_COLUMN, TIMESTAMP_COLUMN]
    ].copy()
    work["_original_index"] = frame.index

    # Determine bucket boundaries. The bucket of a transaction at time T
    # is ``floor(T / GRAPH_BUCKET)``; the graph it scores against is the
    # one built from transactions in ``[T_bucket_start - GRAPH_WINDOW,
    # T_bucket_start)``. We compute and label every bucket once.
    work["_bucket"] = work[TIMESTAMP_COLUMN].dt.floor(GRAPH_BUCKET)

    # Order the bucket labels chronologically so we can iterate in
    # historical order while accumulating graph state.
    bucket_labels = sorted(work["_bucket"].unique())

    # Output container, keyed by the original row index. We seed every
    # feature column with NaN so transactions in early buckets (which
    # have no prior history) emit NaN by default; the sklearn pipeline
    # imputes these downstream.
    feature_columns = [
        "src_24h_degree_in",
        "src_24h_degree_out",
        "src_24h_pagerank",
        "src_24h_clustering",
        "dst_24h_degree_in",
        "dst_24h_degree_out",
        "dst_24h_pagerank",
        "dst_24h_clustering",
        "edge_novelty_24h",
    ]
    output = pd.DataFrame(
        data={col: np.nan for col in feature_columns},
        index=frame.index,
    )

    # Iterate through buckets in chronological order. For each bucket,
    # construct the graph of transactions in the trailing 24h window
    # ending at the bucket's start. Every transaction whose bucket is
    # this one will read its features from that graph.
    for bucket_start in bucket_labels:
        window_end = bucket_start
        window_start = bucket_start - GRAPH_WINDOW

        # Transactions in the lookback window become the graph edges.
        # ``closed='left'`` on the lower bound is enforced implicitly
        # by the strict inequality on ``window_start``; the upper
        # bound is strict-less-than ``bucket_start`` so the bucket's
        # own transactions are excluded from their own graph.
        window_mask = (
            (work[TIMESTAMP_COLUMN] >= window_start)
            & (work[TIMESTAMP_COLUMN] < window_end)
        )
        edges = work.loc[
            window_mask, [SOURCE_ACCOUNT_COLUMN, DEST_ACCOUNT_COLUMN]
        ]

        # An empty window means no graph exists; the seeded NaNs above
        # carry through and the sklearn imputer handles them.
        if len(edges) == 0:
            continue

        graph = _build_directed_graph(edges)
        pagerank = _safe_pagerank(graph)
        clustering = _safe_clustering(graph)

        # Score every transaction whose bucket label matches this one.
        # ``in_degree`` and ``out_degree`` are O(1) lookups on the
        # NetworkX DiGraph; PageRank and clustering were computed once
        # above and looked up from the precomputed dicts.
        bucket_rows = work.loc[work["_bucket"] == bucket_start]

        src_features = _lookup_node_features(
            bucket_rows[SOURCE_ACCOUNT_COLUMN], graph, pagerank, clustering
        )
        dst_features = _lookup_node_features(
            bucket_rows[DEST_ACCOUNT_COLUMN], graph, pagerank, clustering
        )

        original_idx = bucket_rows["_original_index"].values

        output.loc[original_idx, "src_24h_degree_in"] = src_features["in_degree"]
        output.loc[original_idx, "src_24h_degree_out"] = src_features["out_degree"]
        output.loc[original_idx, "src_24h_pagerank"] = src_features["pagerank"]
        output.loc[original_idx, "src_24h_clustering"] = src_features["clustering"]

        output.loc[original_idx, "dst_24h_degree_in"] = dst_features["in_degree"]
        output.loc[original_idx, "dst_24h_degree_out"] = dst_features["out_degree"]
        output.loc[original_idx, "dst_24h_pagerank"] = dst_features["pagerank"]
        output.loc[original_idx, "dst_24h_clustering"] = dst_features["clustering"]

        # Edge novelty: 1.0 if the (src, dst) ordered pair is absent
        # from the bucket's graph, 0.0 otherwise. Built as a set lookup
        # against the edges of the trailing-window graph.
        existing_edges = set(graph.edges())
        novelty = np.array(
            [
                1.0 if (s, d) not in existing_edges else 0.0
                for s, d in zip(
                    bucket_rows[SOURCE_ACCOUNT_COLUMN].values,
                    bucket_rows[DEST_ACCOUNT_COLUMN].values,
                )
            ],
            dtype=np.float32,
        )
        output.loc[original_idx, "edge_novelty_24h"] = novelty

    return output


def _build_directed_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Construct a directed multigraph and collapse parallel edges.

    Internal helper. The construction collapses parallel edges into a
    single edge with a ``weight`` attribute equal to the count. This
    preserves the throughput signal while letting PageRank and the
    other centrality calculations operate on a simple graph (where
    they are well-defined).
    """
    multigraph = nx.MultiDiGraph()
    multigraph.add_edges_from(
        zip(edges[SOURCE_ACCOUNT_COLUMN].values, edges[DEST_ACCOUNT_COLUMN].values)
    )

    simple = nx.DiGraph()
    for u, v in multigraph.edges():
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += 1.0
        else:
            simple.add_edge(u, v, weight=1.0)
    return simple


def _safe_pagerank(graph: nx.DiGraph) -> dict[str, float]:
    """Compute weighted PageRank with a fallback for tiny graphs.

    Internal helper. PageRank on graphs with fewer than two nodes is
    degenerate; we return an empty dict and let the downstream lookup
    use its NaN-imputation path.
    """
    if graph.number_of_nodes() < 2:
        return {}
    # Damping factor 0.85 is the canonical value from the original
    # PageRank paper. We use the weighted variant so high-throughput
    # edges contribute proportionally more to the centrality.
    try:
        return nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=100)
    except nx.PowerIterationFailedConvergence:
        # Pathological graphs occasionally fail to converge in 100
        # iterations. In that case we fall back to a uniform PageRank,
        # which preserves the no-information default the sklearn
        # imputer would assign anyway.
        return dict.fromkeys(graph.nodes(), 1.0 / graph.number_of_nodes())


def _safe_clustering(graph: nx.DiGraph) -> dict[str, float]:
    """Compute local clustering coefficient on the undirected projection.

    Internal helper. NetworkX's directed clustering is defined but
    rarely used in practice for transaction graphs; the undirected
    projection produces a more interpretable signal and matches the
    convention in the academic AML graph-learning literature.
    """
    if graph.number_of_nodes() < 2:
        return {}
    return nx.clustering(graph.to_undirected())


def _lookup_node_features(
    nodes: pd.Series,
    graph: nx.DiGraph,
    pagerank: dict[str, float],
    clustering: dict[str, float],
) -> dict[str, np.ndarray]:
    """Vectorised node-feature lookup against a constructed graph.

    Internal helper. Nodes absent from the graph (e.g., an entity making
    its first appearance in this window) receive NaN, which the sklearn
    pipeline imputes downstream.
    """
    in_degree = np.array(
        [graph.in_degree(n) if n in graph else np.nan for n in nodes],
        dtype=np.float32,
    )
    out_degree = np.array(
        [graph.out_degree(n) if n in graph else np.nan for n in nodes],
        dtype=np.float32,
    )
    pr = np.array(
        [pagerank.get(n, np.nan) for n in nodes],
        dtype=np.float32,
    )
    cl = np.array(
        [clustering.get(n, np.nan) for n in nodes],
        dtype=np.float32,
    )
    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "pagerank": pr,
        "clustering": cl,
    }
