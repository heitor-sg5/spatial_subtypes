"""
Paer 2 — neighbor-feature augmentation.

Using the spatial adjacency matrix W built by graph.py produces
an embedding that blends each cell's own expression with a decay
-weighted average of its spatial neighbors' expression.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _get_expression_matrix(adata: AnnData, layer: str | None, use_hvg: bool) -> np.ndarray:
    X = adata.layers[layer] if layer is not None else adata.X
    if use_hvg:
        if "highly_variable" not in adata.var:
            raise KeyError(
                "adata.var['highly_variable'] not found. Run HVG selection "
                "(e.g. scanpy.pp.highly_variable_genes) before augment_features, "
                "or pass use_hvg=False to use all genes."
            )
        # Subset to HVGs before any further processing, to save memory and speed.
        X = X[:, adata.var["highly_variable"].to_numpy()]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    return X

def _row_normalize(W: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Row-stochastic normalization so neighbor aggregation is a weighted
    mean.

    Returns the normalized matrix and a boolean mask of isolated cells
    (zero-degree rows) that need a fallback.
    """
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    isolated = row_sums == 0
    safe_sums = np.where(isolated, 1.0, row_sums)
    D_inv = sp.diags(1.0 / safe_sums)
    W_norm = D_inv @ W
    return W_norm.tocsr(), isolated

def augment_features(
    adata: AnnData,
    W: sp.csr_matrix,
    layer: str | None = None,
    use_hvg: bool = True,
    lam: float = 0.3,
    n_pcs: int = 50,
    include_gradient: bool = False,
    random_state: int = 0,
    key_added: str = "augmented",
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the neighbor-augmented PCA embedding used for clustering.

    Parameters
    adata : AnnData
        Expression data; assumed already normalized/log1p-transformed
        (standard scanpy preprocessing) before this step.
    W : scipy.sparse matrix
        Spatial adjacency matrix from `build_spatial_graph`. Any weight
        scheme (binary or decay-weighted).
    layer : str, optional
        Use `adata.layers[layer]` instead of `adata.X`.
    use_hvg : bool
        Restrict to `adata.var['highly_variable']` genes. Recommended
        True; set False only for small custom panels.
    lam : float in [0, 1]
        Add weight; lam=0 reduces to standard own-expression-only PCA
        (i.e. the expression-only Leiden baseline, once clustered),
        lam=1 is neighborhood-only. Block-wise scaling is what makes lam
        directly comparable across datasets of different expression scale.
    n_pcs : int
        Number of principal components to return.
    include_gradient : bool
        If True, adds a third feature block capturing local expression
        heterogeneity (a weighted within-neighborhood standard deviation
        per gene), capturing "am I near an expression gradient/edge"
        rather than just "what is my neighborhood's average". When
        enabled, lam is split evenly between the mean neighbor block and 
        the gradient block.
    random_state : int
        PCA random state.
    key_added : str
        The PCA output is stored at `adata.obsm[f"X_{key_added}_pca"]`
        (rows outside `target_mask` are NaN-padded). Note: only the
        final PCA embedding is stored, NOT the raw concatenated
        feature blocks -- storing those at full-tissue scale is
        expensive for no benefit, since nothing downstream reads them
        back (use this function's return value, X_pca, directly).
    target_mask : np.ndarray of bool, shape (n_cells,), optional
        Marks which cells to compute output embeddings for (e.g. the
        cell type being subtyped). IMPORTANT: `adata` and `W` should
        still represent the FULL TISSUE, not a pre-filtered subset.
        If None, all cells in `adata` are treated as the target.

    Returns
    X_pca : np.ndarray, shape (n_target_cells, n_pcs)
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    own_full = _get_expression_matrix(adata, layer, use_hvg)
    W_norm, isolated_full = _row_normalize(W)
    if isolated_full.any():
        n_target_isolated = isolated_full.sum() if target_mask is None else (isolated_full & target_mask).sum()
        if n_target_isolated > 0:
            warnings.warn(
                f"{n_target_isolated} target cell(s) have zero spatial neighbors "
                f"under the current graph (isolated nodes). Their neighborhood "
                f"block falls back to their own expression. Consider a larger "
                f"radius/n_neighbors if this affects a large fraction of cells.",
                stacklevel=2,
            )

    # Neighbor aggregation always uses the FULL TISSUE graph.
    neighbor_mean_full = W_norm @ own_full
    neighbor_mean_full[isolated_full] = own_full[isolated_full]  # fallback for isolated cells

    # Now restrict to the target population for everything downstream.
    if target_mask is not None:
        own = own_full[target_mask]
        neighbor_mean = neighbor_mean_full[target_mask]
    else:
        own = own_full
        neighbor_mean = neighbor_mean_full

    scaler = StandardScaler()
    own_scaled = scaler.fit_transform(own)
    neighbor_scaled = StandardScaler().fit_transform(neighbor_mean)

    if include_gradient:
        # Weighted within-neighborhood variance per gene: E[(x_j - nbr_mean_i)^2 | j in N(i)]
        sq_diff_full = W_norm @ (own_full**2) - neighbor_mean_full**2
        sq_diff_full = np.clip(sq_diff_full, a_min=0.0, a_max=None)  # guard float error
        gradient_full = np.sqrt(sq_diff_full)
        gradient_full[isolated_full] = 0.0
        gradient = gradient_full[target_mask] if target_mask is not None else gradient_full
        gradient_scaled = StandardScaler().fit_transform(gradient)

        own_w = np.sqrt(1 - lam)
        nbr_w = np.sqrt(lam / 2)
        grad_w = np.sqrt(lam / 2)
        blocks = [own_w * own_scaled, nbr_w * neighbor_scaled, grad_w * gradient_scaled]
    else:
        own_w = np.sqrt(1 - lam)
        nbr_w = np.sqrt(lam)
        blocks = [own_w * own_scaled, nbr_w * neighbor_scaled]

    X_concat = np.concatenate(blocks, axis=1)

    n_pcs_eff = min(n_pcs, X_concat.shape[0] - 1, X_concat.shape[1])
    pca = PCA(n_components=n_pcs_eff, random_state=random_state)
    X_pca = pca.fit_transform(X_concat)

    if target_mask is None:
        adata.obsm[f"X_{key_added}_pca"] = X_pca
    else:
        # Store the PCA output in adata.obsm with NaN padding for non-target rows.
        full_pca = np.full((adata.n_obs, X_pca.shape[1]), np.nan, dtype=np.float32)
        full_pca[target_mask] = X_pca
        adata.obsm[f"X_{key_added}_pca"] = full_pca

    adata.uns[f"{key_added}_params"] = {
        "lam": lam,
        "n_pcs": n_pcs_eff,
        "include_gradient": include_gradient,
        "use_hvg": use_hvg,
        "target_mask_used": target_mask is not None,
        "n_target_cells": int(target_mask.sum()) if target_mask is not None else adata.n_obs,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    return X_pca