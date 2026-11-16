"""
Stage 3: Leiden clustering of the embedding.

A kNN graph is built on the embedding and partitioned with Leiden. Because
the number of clusters depends strongly on the resolution, 'cluster_to_counts'
searches for the resolution that gives a requested number of clusters.
"""

from __future__ import annotations

import igraph as ig
import leidenalg
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_graph(X: np.ndarray, n_neighbors: int = 15) -> ig.Graph:
    """Undirected kNN graph on the rows of X (exact search)."""
    X = np.asarray(X)
    if np.isnan(X).any():
        raise ValueError("X contains NaNs")
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    keep = idx != np.arange(n)[:, None]
    keep[keep.all(axis=1), -1] = False
    a = np.repeat(np.arange(n), k)
    b = idx[keep]
    pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1), axis=0)
    return ig.Graph(n=n, edges=pairs.tolist())

def leiden_labels(graph: ig.Graph, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Leiden partition of a graph as an integer label array."""
    part = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(part.membership, dtype=int)

def cluster_embedding(X: np.ndarray, resolution: float = 1.0, n_neighbors: int = 15,
                      seed: int = 0) -> np.ndarray:
    """kNN graph on X, then Leiden at a fixed resolution."""
    return leiden_labels(knn_graph(X, n_neighbors), resolution, seed)

def resolution_for_n_clusters(graph: ig.Graph, n_clusters: int, seed: int = 0,
                              max_iter: int = 16, lo: float = 1e-3, hi: float = 50.0):
    """
    Search (in log space) for the resolution that yields 'n_clusters'.
    Returns (resolution, labels). If the exact count is never reached, the
    closest count found is returned (ties go to the lower resolution).
    """
    best = None
    a, b = np.log(lo), np.log(hi)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        labels = leiden_labels(graph, float(np.exp(mid)), seed)
        k = int(labels.max()) + 1
        gap = abs(k - n_clusters)
        if best is None or gap < best[0]:
            best = (gap, float(np.exp(mid)), labels)
        if k == n_clusters:
            break
        if k < n_clusters:
            a = mid
        else:
            b = mid
    return best[1], best[2]

def cluster_to_counts(X: np.ndarray, counts=(4, 6, 8, 10), seeds=(0,), n_neighbors: int = 15):
    """
    Cluster at several target cluster counts and several Leiden seeds.
    The resolution is searched once per count (seed = first seed) and reused
    for the other seeds, so the realised count can differ slightly between
    seeds. Returns {(count, seed): labels}.
    """
    g = knn_graph(X, n_neighbors)
    out = {}
    for c in counts:
        res, labels = resolution_for_n_clusters(g, c, seed=seeds[0])
        out[(c, seeds[0])] = labels
        for s in seeds[1:]:
            out[(c, s)] = leiden_labels(g, res, s)
    return out
