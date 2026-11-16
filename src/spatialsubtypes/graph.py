"""
Stage 1: spatial neighbour graphs and distance kernels.

A graph is a list of edges with their true Euclidean lengths. Weights are
applied afterwards by 'apply_kernel', so a kernel can be swapped without
rebuilding the graph.

Only the rows you ask for ('query') are built. For subtyping, that is the
target cells; their neighbours can be any bin in the tissue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree

MODES = ("radius", "knn", "delaunay")
KERNELS = ("gaussian", "exponential", "reciprocal", "uniform")

@dataclass
class Graph:
    """
    Directed edges from 'n_rows' query cells to 'n_cols' cells.
    'row' indexes the query cells (0 .. n_rows-1), 'col' indexes all cells.
    'dist' holds the true Euclidean length of each edge.
    """

    n_rows: int
    n_cols: int
    row: np.ndarray
    col: np.ndarray
    dist: np.ndarray
    mode: str
    radius: float | None = None

    @property
    def degree(self) -> np.ndarray:
        """Number of neighbours of each query cell."""
        return np.bincount(self.row, minlength=self.n_rows)

def _as_query(query, n: int) -> np.ndarray:
    if query is None:
        return np.arange(n)
    query = np.asarray(query)
    if query.dtype == bool:
        if query.shape[0] != n:
            raise ValueError("boolean query must have one entry per cell")
        return np.flatnonzero(query)
    return query.astype(np.int64)

def _symmetrise(row, col, dist, n):
    """Keep each undirected edge once per direction (union of both directions)."""
    a = np.concatenate([row, col]).astype(np.int64)
    b = np.concatenate([col, row]).astype(np.int64)
    d = np.concatenate([dist, dist])
    key = a * n + b
    _, first = np.unique(key, return_index=True)
    return a[first], b[first], d[first]

def _delaunay_edges(coords: np.ndarray, prune_factor: float | None):
    """
    Unique undirected Delaunay edges (a < b) and their true lengths.
    Each interior edge belongs to two triangles, so edges are de-duplicated
    before lengths are computed.
    """
    n = len(coords)
    s = Delaunay(coords).simplices
    e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(axis=1)
    key = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1])
    a, b = key // n, key % n
    length = np.linalg.norm(coords[a] - coords[b], axis=1)
    if prune_factor:
        keep = length <= prune_factor * np.median(length)
        a, b, length = a[keep], b[keep], length[keep]
    return a, b, length

def build_graph(
    coords: np.ndarray,
    mode: str = "knn",
    *,
    radius: float | None = None,
    n_neighbors: int = 15,
    rings: int = 1,
    prune_factor: float | None = 3.0,
    query=None,
    symmetric: bool = False,
) -> Graph:
    """
    Build a spatial graph.

    Parameters
    ----------
    coords : (n, 2) array of coordinates, in one consistent unit (e.g. microns).
    mode : "radius", "knn" or "delaunay".
    radius : cutoff for mode="radius", in the units of 'coords'.
    n_neighbors : neighbours per cell for mode="knn" (self excluded).
    rings : for mode="delaunay", how many triangulation hops count as neighbours
        (1 = triangulation edges only).
    prune_factor : for mode="delaunay", drop edges longer than this multiple of
        the median edge length (removes links across tissue gaps).
    query : indices (or boolean mask) of the cells to build rows for. Default: all.
    symmetric : make the graph undirected (union of both directions). Only
        valid when 'query' is None.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    n = len(coords)
    q = _as_query(query, n)
    if symmetric and query is not None:
        raise ValueError("symmetric=True requires query=None")

    if mode == "radius":
        if radius is None or radius <= 0:
            raise ValueError("mode='radius' requires a positive radius")
        tree = cKDTree(coords)
        tq = tree if query is None else cKDTree(coords[q])
        pairs = tq.sparse_distance_matrix(tree, radius, output_type="ndarray")
        i, j, d = pairs["i"].astype(np.int64), pairs["j"].astype(np.int64), pairs["v"]
        keep = q[i] != j
        row, col, dist = i[keep], j[keep], d[keep]

    elif mode == "knn":
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        kk = min(n_neighbors + 1, n)
        dist_k, idx = cKDTree(coords).query(coords[q], k=kk, workers=-1)
        dist_k = dist_k.reshape(len(q), kk)
        idx = idx.reshape(len(q), kk)
        keep = idx != q[:, None]
        missed_self = keep.all(axis=1)  # ties at distance 0 can push self out
        keep[missed_self, -1] = False
        row = np.repeat(np.arange(len(q)), kk - 1)
        col, dist = idx[keep], dist_k[keep]

    else:  # delaunay
        if rings < 1:
            raise ValueError("rings must be at least 1")
        a, b, _ = _delaunay_edges(coords, prune_factor)
        adj = sp.csr_matrix(
            (np.ones(2 * len(a)), (np.concatenate([a, b]), np.concatenate([b, a]))),
            shape=(n, n),
        )
        reach = adj[q]
        for _ in range(rings - 1):
            reach = reach + reach @ adj
            reach.data = np.ones_like(reach.data)
        reach = reach.tocoo()
        keep = reach.col != q[reach.row]
        row, col = reach.row[keep].astype(np.int64), reach.col[keep].astype(np.int64)
        dist = np.linalg.norm(coords[q[row]] - coords[col], axis=1)

    if symmetric:
        row, col, dist = _symmetrise(row, col, dist, n)
    order = np.argsort(row, kind="stable")
    return Graph(
        n_rows=len(q),
        n_cols=n,
        row=row[order].astype(np.int32),
        col=col[order].astype(np.int32),
        dist=dist[order],
        mode=mode,
        radius=radius if mode == "radius" else None,
    )

def adjacency(graph: Graph) -> sp.csr_matrix:
    """Binary adjacency matrix (1 for every edge)."""
    return sp.csr_matrix(
        (np.ones(len(graph.row)), (graph.row, graph.col)),
        shape=(graph.n_rows, graph.n_cols),
    )

def apply_kernel(graph: Graph, kernel: str = "gaussian", bandwidth: float | None = None) -> sp.csr_matrix:
    """
    Turn a graph into a row-normalised weight matrix (each non-empty row sums to 1).

    kernel
        "gaussian"    exp(-2 d^2 / h^2)
        "exponential" exp(-2 d / h)
        "reciprocal"  1 / d (BANKSY's "reciprocal"; needs no bandwidth)
        "uniform"     equal weights (baseline)

    The bandwidth h is 'bandwidth' if given, else the radius for radius
    graphs, else each cell's farthest-neighbour distance. At d = h the
    Gaussian and exponential weights are both exp(-2) = 0.14.
    Cells with no neighbours get an all-zero row.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    d, row = graph.dist, graph.row

    if kernel == "uniform":
        w = np.ones_like(d)
    elif kernel == "reciprocal":
        positive = d[d > 0]
        floor = positive.min() if positive.size else 1.0
        w = 1.0 / np.maximum(d, floor) 
    else:
        if bandwidth is not None:
            h = np.full(graph.n_rows, float(bandwidth))
        elif graph.radius is not None:
            h = np.full(graph.n_rows, float(graph.radius))
        else:
            h = np.zeros(graph.n_rows)
            np.maximum.at(h, row, d)
            h[h == 0] = 1.0
        hh = h[row]
        w = np.exp(-2.0 * d**2 / hh**2) if kernel == "gaussian" else np.exp(-2.0 * d / hh)

    sums = np.bincount(row, weights=w, minlength=graph.n_rows)
    w = w / np.where(sums > 0, sums, 1.0)[row]
    return sp.csr_matrix((w, (row, graph.col)), shape=(graph.n_rows, graph.n_cols))
