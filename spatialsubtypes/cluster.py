"""
Part 3 — Leiden clustering.

Takes an embedding, returns labels. If augment_features was called with
target_mask, pass the already-subsetted X_pca returned by that function.
"""

from __future__ import annotations

import igraph as ig
import leidenalg
import numpy as np
from anndata import AnnData
from sklearn.neighbors import kneighbors_graph

def cluster_leiden(
    X: np.ndarray,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    random_state: int = 0,
    key_added: str | None = None,
    adata_out: AnnData | None = None,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    kNN graph (built via sklearn) + Leiden clustering (via
    leidenalg/igraph directly).

    Parameters
    X : np.ndarray, shape (n_cells, n_components)
        Embedding to cluster on (already restricted to target cells).
        Must not contain NaNs; slice the full-length obsm array
        with target mask before calling (`X_full[target_mask]`) or
        just use the array `augment_features` already returned directly.
    resolution : float
        Leiden resolution parameter. Higher = more, smaller clusters.
    n_neighbors : int
        Number of neighbors for the expression-space kNN graph Leiden
        runs on.
    random_state : int
        For reproducibility of Leiden's optimization.
    key_added : str, optional
        If `adata_out` is also given, results are additionally written
        to `adata_out.obs[key_added]` (respecting `target_mask` if set,
        i.e. non-target cells get "NA" rather than a spurious label).
    adata_out : AnnData, optional
        Convenience: if provided along with `key_added`, cluster labels
        are written back onto this object's `.obs`.
    target_mask : np.ndarray of bool, optional
        Only used if `adata_out` is given, to correctly place labels
        back into the full-length `.obs` column.

    Returns
    labels : np.ndarray, shape (n_cells,)
        Cluster assignment as a string array (e.g. "0", "1", ...),
        matching scanpy's convention so downstream categorical handling
    """
    if np.isnan(X).any():
        raise ValueError(
            "X contains NaNs. If this came from augment_features with a "
            "target_mask, pass the returned array directly (already "
            "subsetted) rather than the full-length adata.obsm entry."
        )

    n_cells = X.shape[0]
    k = min(n_neighbors, n_cells - 1)

    # Exact kNN via sklearn
    knn = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    knn = knn.maximum(knn.T) # symmetrize

    rows, cols = knn.nonzero()
    undirected_mask = rows < cols  # dedupe symmetric pairs into single edges
    edges = list(zip(rows[undirected_mask].tolist(), cols[undirected_mask].tolist()))

    g = ig.Graph(n=n_cells, edges=edges)

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state,
    )
    labels = np.array([str(c) for c in partition.membership])

    if adata_out is not None and key_added is not None:
        col = np.full(adata_out.n_obs, "NA", dtype=object)
        if target_mask is not None:
            col[target_mask] = labels
        else:
            col[:] = labels
        adata_out.obs[key_added] = col

    return labels