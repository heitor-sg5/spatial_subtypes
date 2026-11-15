"""Tests for cluster.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from spatialsubtypes import cluster_embedding, cluster_to_counts, knn_graph, resolution_for_n_clusters


def _blobs(k=5, n=80, seed=0, spread=8.0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, spread, (k, 4))
    X = np.vstack([rng.normal(c, 1.0, (n, 4)) for c in centres])
    return X, np.repeat(np.arange(k), n)


def test_recovers_well_separated_groups():
    X, y = _blobs(3)
    assert adjusted_rand_score(y, cluster_embedding(X, 0.5, 10)) == 1.0


def test_nan_input_raises():
    with pytest.raises(ValueError, match="NaN"):
        knn_graph(np.full((20, 3), np.nan))


def test_same_seed_same_labels():
    X, _ = _blobs(4)
    assert (cluster_embedding(X, 1.0, 10, seed=3) == cluster_embedding(X, 1.0, 10, seed=3)).all()


def test_resolution_search_hits_requested_cluster_count():
    # overlapping blobs: the kNN graph is connected, so Leiden can merge them freely
    X, _ = _blobs(6, n=120, spread=1.5)
    g = knn_graph(X, 15)
    for target in (3, 6):
        _, labels = resolution_for_n_clusters(g, target)
        assert labels.max() + 1 == target


def test_cluster_to_counts_returns_every_count_and_seed():
    X, _ = _blobs(6, spread=1.5)
    out = cluster_to_counts(X, counts=(3, 6), seeds=(0, 1, 2), n_neighbors=15)
    assert set(out) == {(c, s) for c in (3, 6) for s in (0, 1, 2)}
    assert out[(6, 0)].max() + 1 == 6
