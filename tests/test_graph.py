"""Tests for graph.py: builders, true edge lengths, kernels."""

from __future__ import annotations

import numpy as np
import pytest

from spatialsubtypes import adjacency, apply_kernel, build_graph

def _interior(grid, margin):
    return np.flatnonzero((grid.min(1) > margin) & (grid.max(1) < grid.max() - margin))

@pytest.mark.parametrize("mode,kw", [("radius", dict(radius=15.0)), ("knn", dict(n_neighbors=6)), ("delaunay", {})])
def test_every_mode_builds_a_nonempty_graph(two_blob_coords, mode, kw):
    g = build_graph(two_blob_coords, mode, **kw)
    assert len(g.row) > 0 and (g.dist > 0).all()
    assert g.n_rows == g.n_cols == len(two_blob_coords)

def test_delaunay_edge_lengths_are_true_euclidean_lengths(two_blob_coords):
    """Regression: interior edges used to be stored at twice their length (summed duplicates)."""
    g = build_graph(two_blob_coords, "delaunay", prune_factor=None)
    true = np.linalg.norm(two_blob_coords[g.row] - two_blob_coords[g.col], axis=1)
    assert np.allclose(g.dist, true)
    key = g.row.astype(np.int64) * g.n_cols + g.col
    assert len(np.unique(key)) == len(key), "duplicate edges"

def test_delaunay_is_symmetric(two_blob_coords):
    A = adjacency(build_graph(two_blob_coords, "delaunay"))
    assert abs(A - A.T).max() == 0

def test_delaunay_pruning_cuts_links_across_gaps():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.uniform(0, 100, (150, 2)), rng.uniform(0, 100, (150, 2)) + [1000, 0]])
    kept = build_graph(pts, "delaunay", prune_factor=3.0)
    assert kept.dist.max() < 3.0 * np.median(build_graph(pts, "delaunay", prune_factor=None).dist) + 1e-9
    assert (build_graph(pts, "delaunay", prune_factor=None).dist > 500).any()

def test_grid_neighbour_counts(grid_coords):
    q = _interior(grid_coords, 60)
    assert set(build_graph(grid_coords, "radius", radius=32.0, query=q).degree) == {48}
    assert set(build_graph(grid_coords, "knn", n_neighbors=48, query=q).degree) == {48}
    d1 = build_graph(grid_coords, "delaunay", rings=1, query=q).degree.mean()
    d3 = build_graph(grid_coords, "delaunay", rings=3, query=q).degree.mean()
    assert d1 < 8 and d3 > 30

def test_query_rows_equal_rows_of_full_graph(two_blob_coords):
    q = np.arange(0, 200, 7)
    for mode, kw in [("radius", dict(radius=12.0)), ("knn", dict(n_neighbors=5)), ("delaunay", dict(rings=2))]:
        full = apply_kernel(build_graph(two_blob_coords, mode, **kw), "gaussian")[q]
        part = apply_kernel(build_graph(two_blob_coords, mode, query=q, **kw), "gaussian")
        assert np.allclose(full.toarray(), part.toarray())

def test_knn_excludes_self_and_has_k_neighbours(two_blob_coords):
    g = build_graph(two_blob_coords, "knn", n_neighbors=7)
    assert (g.degree == 7).all() and (g.row != g.col).all()

def test_knn_with_duplicate_coordinates_still_excludes_self():
    pts = np.array([[0.0, 0], [0, 0], [0, 0], [5, 5], [6, 5], [9, 9]])
    g = build_graph(pts, "knn", n_neighbors=2)
    assert (g.degree == 2).all() and (g.row != g.col).all()

def test_symmetric_knn_is_symmetric(two_blob_coords):
    A = adjacency(build_graph(two_blob_coords, "knn", n_neighbors=5, symmetric=True))
    assert abs(A - A.T).max() == 0

def test_radius_requires_radius(two_blob_coords):
    with pytest.raises(ValueError):
        build_graph(two_blob_coords, "radius")

@pytest.mark.parametrize("kernel", ["gaussian", "exponential", "reciprocal", "uniform"])
def test_kernel_rows_sum_to_one(two_blob_coords, kernel):
    W = apply_kernel(build_graph(two_blob_coords, "knn", n_neighbors=8), kernel)
    assert np.allclose(np.asarray(W.sum(1)).ravel(), 1.0)
    assert W.data.min() > 0

def test_kernels_weight_nearer_cells_more_but_uniform_does_not():
    pts = np.array([[0.0, 0], [1, 0], [2, 0], [4, 0]])
    g = build_graph(pts, "knn", n_neighbors=3, query=[0])
    order = np.argsort(g.dist)
    for kernel in ("gaussian", "exponential", "reciprocal"):
        w = apply_kernel(g, kernel).toarray()[0, g.col[order]]
        assert (np.diff(w) < 0).all(), kernel
    w = apply_kernel(g, "uniform").toarray()[0, g.col]
    assert np.allclose(w, 1 / 3)

def test_reciprocal_weights_are_inverse_distance():
    pts = np.array([[0.0, 0], [1, 0], [2, 0]])
    g = build_graph(pts, "knn", n_neighbors=2, query=[0])
    w = apply_kernel(g, "reciprocal").toarray()[0]
    assert np.isclose(w[1] / w[2], 2.0)

def test_gaussian_and_exponential_agree_at_the_bandwidth():
    """Both kernels are exp(-2) at d = h, before normalisation."""
    pts = np.array([[0.0, 0], [10, 0], [5, 0]])
    g = build_graph(pts, "knn", n_neighbors=2, query=[0])
    h = g.dist.max()
    for kernel in ("gaussian", "exponential"):
        W = apply_kernel(g, kernel, bandwidth=h).toarray()[0]
        far, near = W[1], W[2]
        raw_far = np.exp(-2)
        raw_near = np.exp(-2 * 0.25) if kernel == "gaussian" else np.exp(-2 * 0.5)
        assert np.isclose(far / near, raw_far / raw_near)

def test_coincident_cells_do_not_create_infinite_weights():
    pts = np.array([[0.0, 0], [0, 0], [3, 0], [0, 4]])
    g = build_graph(pts, "knn", n_neighbors=3, query=[0])
    for kernel in ("gaussian", "exponential", "reciprocal"):
        assert np.isfinite(apply_kernel(g, kernel).data).all()

def test_isolated_cells_get_empty_rows():
    pts = np.array([[0.0, 0], [1, 0], [100, 100]])
    g = build_graph(pts, "radius", radius=5.0)
    W = apply_kernel(g, "gaussian")
    assert W[2].nnz == 0 and g.degree[2] == 0
