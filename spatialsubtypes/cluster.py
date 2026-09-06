"""
Part 3 — Leiden clustering.

Takes an embedding, returns labels. If augment_features was called with 
target_mask, pass the already-subsetted X_pca returned by that function.
"""

import numpy as np
import scanpy as sc
from anndata import AnnData

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
    Standard kNN graph + Leiden clustering.

    Parameters
    X : np.ndarray, shape (n_cells, n_components)
        Embedding to cluster on (already restricted to target cells). 
        Must not contain NaNs;  slice the full-length obsm array 
        with target mask before calling (`X_full[target_mask]`) or 
        just use the array `augment_features` already returned directly.
    resolution : float
        Leiden resolution parameter. Higher = more, smaller clusters.
    n_neighbors : int
        Number of neighbors for the expression-space kNN graph Leiden
        runs on.
    random_state : int
        For reproducibility of both the kNN graph and Leiden.
    key_added : str, optional
        If `adata_out` is also given, results are additionally written
        to `adata_out.obs[key_added]` (respecting `target_mask` if set,
        i.e. non-target cells get NaN rather than a spurious label).
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

    tmp = AnnData(X=np.zeros((X.shape[0], 1)))  # scanpy needs an AnnData shell
    tmp.obsm["X_input"] = X
    sc.pp.neighbors(tmp, use_rep="X_input", n_neighbors=n_neighbors, random_state=random_state)
    sc.tl.leiden(
        tmp,
        resolution=resolution,
        random_state=random_state,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    labels = tmp.obs["leiden"].to_numpy().astype(str)

    if adata_out is not None and key_added is not None:
        col = np.full(adata_out.n_obs, "NA", dtype=object)
        if target_mask is not None:
            col[target_mask] = labels
        else:
            col[:] = labels
        adata_out.obs[key_added] = col

    return labels