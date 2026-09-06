"""
Tests for augment.py over the FULL TISSUE graph.
"""

import numpy as np
import pytest
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from spatialsubtypes.augment import augment_features
from spatialsubtypes.graph import build_spatial_graph

def test_lam0_equals_plain_pca(two_blob_coords):
    """
    augment_features(lam=0) should be numerically equivalent to
    standard PCA on z-scored expression.
    """
    rng = np.random.default_rng(0)
    n = len(two_blob_coords)
    X = rng.normal(0, 1, (n, 10)).astype(np.float32)
    adata = AnnData(X=X, obsm={"spatial": two_blob_coords})
    adata.var["highly_variable"] = True

    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")
    X_pca = augment_features(adata, W, lam=0.0, n_pcs=5, key_added="regress")

    X_manual = PCA(n_components=5, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    corr = abs(np.corrcoef(X_pca[:, 0], X_manual[:, 0])[0, 1])
    assert corr > 0.9999

def test_lambda_sweep_improves_separability_under_weak_expression_signal():
    """
    With a weak own-expression signal (0.3 SD mean shift) but a strong
    spatial split, silhouette against the true spatial group should
    increase as lam increases from 0 to 0.9.
    """
    rng = np.random.default_rng(0)
    n_per_group = 100
    coords = np.vstack(
        [rng.normal([0, 0], 8, (n_per_group, 2)), rng.normal([200, 200], 8, (n_per_group, 2))]
    )
    n_genes = 20
    base = rng.normal(0, 1, (n_per_group * 2, n_genes))
    shift = np.zeros_like(base)
    shift[:n_per_group, :5] += 0.3
    shift[n_per_group:, :5] -= 0.3
    X = (base + shift).astype(np.float32)

    adata = AnnData(X=X, obsm={"spatial": coords})
    adata.var["highly_variable"] = True
    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")

    niche_label = np.array([0] * n_per_group + [1] * n_per_group)
    sils = {}
    for lam in [0.0, 0.3, 0.6, 0.9]:
        X_pca = augment_features(adata, W, lam=lam, n_pcs=5, key_added=f"sweep{lam}")
        sils[lam] = silhouette_score(X_pca, niche_label)

    assert sils[0.9] > sils[0.6] > sils[0.3] > sils[0.0]
    assert sils[0.0] < 0.1  # near-zero: own expression alone barely separates the groups

def test_target_mask_output_shape_and_nan_placement(macrophage_in_tissue):
    """
    Output should be restricted to target cells; obsm storage for
    non-target rows should be NaN.
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]
    n_target = target_mask.sum()

    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")
    X_pca = augment_features(adata, W, lam=0.7, n_pcs=5, target_mask=target_mask, key_added="macro")

    assert X_pca.shape[0] == n_target
    stored = adata.obsm["X_macro_pca"]
    assert stored.shape[0] == adata.n_obs
    assert not np.isnan(stored[target_mask]).any()
    assert np.isnan(stored[~target_mask]).all()

def test_target_mask_uses_full_tissue_neighbors(macrophage_in_tissue):
    """
    Target cells have identical own expression regardless of location,
    so any spatial separation in the output must come from neighbor
    aggregation over the surrounding non-target cells. Silhouette
    against the true spatial niche should be clearly positive despite
    zero own-expression signal.
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]
    true_niche = macrophage_in_tissue["true_niche"]

    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")
    X_pca = augment_features(adata, W, lam=0.7, n_pcs=5, target_mask=target_mask, key_added="macro")

    sil = silhouette_score(X_pca, true_niche)
    assert sil > 0.3

def test_isolated_cells_warn(macrophage_in_tissue):
    """
    A radius small enough to isolate at least one target cell should
    raise a UserWarning.
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]
    W = build_spatial_graph(adata, mode="radius", radius=0.01, decay="gaussian")
    with pytest.warns(UserWarning, match="zero spatial neighbors"):
        augment_features(adata, W, lam=0.5, n_pcs=5, target_mask=target_mask, key_added="isolated")

def test_lam_out_of_range_raises(macrophage_in_tissue):
    """
    lam < 0 or lam > 1 should raise a ValueError.
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]
    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")
    with pytest.raises(ValueError):
        augment_features(adata, W, lam=1.5, target_mask=target_mask)