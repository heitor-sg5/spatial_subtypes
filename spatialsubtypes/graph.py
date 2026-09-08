"""
Part 1 — spatial neighbor graph construction.

Builds a weighted adjacency matrix (W) over cells based on physical 
(spatial) coordinates, which is used downstream by both augment.py
and validate.py.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.spatial import Delaunay

def build_spatial_graph(
    adata: AnnData,
    spatial_key: str = "spatial",
    mode: str = "delaunay", 
    radius: float | None = None,
    n_neighbors: int = 6,
    decay: str | None = None,
    decay_scale: float | None = None,
    include_self: bool = False,
    key_added: str = "spatial",
) -> sp.csr_matrix:
    """
    Build a weighted spatial adjacency matrix over cells in adata.

    Parameters
    adata : AnnData
        Must contain 2D spatial coordinates in `adata.obsm[spatial_key]`.
    spatial_key : str
        Key in `adata.obsm` holding coordinates.
    mode : {"radius", "knn", "delaunay"}
        How candidate neighbors are selected:
        - "radius":   all cells within `radius` of each other.
        - "knn":      each cell's `n_neighbors` nearest cells.
        - "delaunay": Delaunay triangulation edges.
    radius : float, optional
        Distance cutoff in the same units as the spatial coordinates
        (e.g. microns). Used if mode="radius" and as the
        default bandwidth reference for decay_scale.
    n_neighbors : int
        Number of neighbors per cell if mode="knn".
    decay : {None, "gaussian", "exponential"}
        Reweighting applied to selected edges based on physical distance.
        - None:           binary weights (1.0 for all selected edges).
        - "gaussian":     weight = exp(-(d^2) / (2 * decay_scale^2))
        - "exponential":  weight = exp(-d / decay_scale)
    decay_scale : float, optional
        Bandwidth for the decay kernel. Defaults to `radius / 2` if
        `radius` is set, else the median observed neighbor distance.
    include_self : bool
        Whether to keep self-loops (diagonal = max weight). Default
        False; augment.py adds self-contribution explicitly and
        separately, so leaving this False avoids double-counting.
    key_added : str
        Results are stored in `adata.obsp[f"{key_added}_connectivities"]`
        (weighted, post-decay) and `adata.obsp[f"{key_added}_distances"]`
        (raw distances for selected edges), using the same slot naming
        convention as squidpy so downstream baseline comparisons
        (squidpy neighborhood enrichment, etc.) can reuse this graph
        without recomputation.

    Returns
    W : scipy.sparse.csr_matrix, shape (n_cells, n_cells)
        Symmetric weighted adjacency matrix. Binary (0/1) if decay is
        None, continuous in (0, 1) on selected edges otherwise.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"'{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key])
    n = coords.shape[0]

    if mode not in {"radius", "knn", "delaunay"}:
        raise ValueError(f"mode must be 'radius', 'knn', or 'delaunay', got {mode!r}")
    if decay not in {None, "gaussian", "exponential"}:
        raise ValueError(f"decay must be None, 'gaussian', or 'exponential', got {decay!r}")

    # ---- Select candidate edges + raw distances ----
    if mode == "radius":
        if radius is None:
            raise ValueError("radius must be set when mode='radius'")
        D = radius_neighbors_graph(
            coords, radius=radius, mode="distance", include_self=include_self
        )
        D = D.maximum(D.T)  # symmetrize

    elif mode == "knn":
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        dist, idx = nn.kneighbors(coords)
        if not include_self:
            dist, idx = dist[:, 1:], idx[:, 1:]
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.ravel()
        vals = dist.ravel()
        D = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        D = D.maximum(D.T)  # symmetrize

    else:  # (delaunay)
        tri = Delaunay(coords)
        rows, cols = [], []
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    rows.append(simplex[i])
                    cols.append(simplex[j])
                    rows.append(simplex[j])
                    cols.append(simplex[i])
        rows, cols = np.array(rows), np.array(cols)
        vals = np.linalg.norm(coords[rows] - coords[cols], axis=1)
        D = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        D.sum_duplicates()

    D.eliminate_zeros() if not include_self else None

    # ---- Decay reweighting ----
    if decay is None:
        W = D.copy()
        W.data = np.ones_like(W.data)
    else:
        if decay_scale is None:
            if mode == "radius" and radius is not None:
                decay_scale = radius / 2.0
            else:
                decay_scale = float(np.median(D.data)) if D.nnz > 0 else 1.0

        W = D.copy()
        if decay == "gaussian":
            W.data = np.exp(-(W.data**2) / (2 * decay_scale**2))
        else:  # (exponential)
            W.data = np.exp(-W.data / decay_scale)

    W = W.maximum(W.T)
    W.eliminate_zeros()

    adata.obsp[f"{key_added}_connectivities"] = W
    adata.obsp[f"{key_added}_distances"] = D
    adata.uns[f"{key_added}_graph_params"] = {
        "mode": mode,
        "radius": radius,
        "n_neighbors": n_neighbors,
        "decay": decay,
        "decay_scale": decay_scale,
    }

    return W