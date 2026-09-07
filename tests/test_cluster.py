"""
Tests for cluster.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from spatialsubtypes.augment import augment_features
from spatialsubtypes.cluster import cluster_leiden
from spatialsubtypes.graph import build_spatial_graph

def test_recovers_known_spatial_groups(macrophage_in_tissue):
    """
    Full graph -> augment -> cluster chain on the macrophage_in_tissue
    fixture should recover the true 30/30 spatial split with perfect
    ARI (=1).
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]
    true_niche = macrophage_in_tissue["true_niche"]

    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")
    X_pca = augment_features(adata, W, lam=0.7, n_pcs=5, target_mask=target_mask, key_added="macro")
    labels = cluster_leiden(X_pca, resolution=0.5, n_neighbors=10)

    assert len(set(labels)) == 2
    assert adjusted_rand_score(true_niche, labels) == 1.0

def test_nan_input_raises():
    """
    cluster_leiden should refuse NaN-containing input with a clear
    error.
    """
    X_with_nan = np.full((20, 5), np.nan)
    with pytest.raises(ValueError, match="NaN"):
        cluster_leiden(X_with_nan)

def test_adata_out_writeback_respects_target_mask(macrophage_in_tissue):
    """
    When adata_out/target_mask are given, labels should land only on
    target rows; non-target rows should get the "NA" placeholder.
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]
    W = build_spatial_graph(adata, mode="radius", radius=20.0, decay="gaussian")
    X_pca = augment_features(adata, W, lam=0.7, n_pcs=5, target_mask=target_mask, key_added="macro")

    cluster_leiden(
        X_pca, resolution=0.5, n_neighbors=10,
        adata_out=adata, key_added="subtype", target_mask=target_mask,
    )

    assert (adata.obs.loc[~target_mask, "subtype"] == "NA").all()
    assert (adata.obs.loc[target_mask, "subtype"] != "NA").all()