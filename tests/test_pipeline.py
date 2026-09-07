"""
Tests for pipeline.py covering the full find_spatial_subtypes() chain.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score

from spatialsubtypes import find_spatial_subtypes

def test_end_to_end_recovery(macrophage_in_tissue):
    """
    Full graph -> augment -> cluster -> validate chain on the macrophage_in_tissue 
    fixture should recover the true 30/30 spatial split with perfect ARI (=1).
    """
    adata = macrophage_in_tissue["adata"]
    true_niche = macrophage_in_tissue["true_niche"]

    adata = find_spatial_subtypes(
        adata,
        cell_type_key="cell_type", target_type="macrophage",
        graph_kwargs=dict(mode="radius", radius=20.0, decay="gaussian"),
        augment_kwargs=dict(lam=0.7, n_pcs=5),
        cluster_kwargs=dict(resolution=0.5, n_neighbors=10),
        validate_kwargs=dict(action="drop", n_perm=1000),
    )

    mask = (adata.obs["cell_type"] == "macrophage").to_numpy()
    raw = adata.obs.loc[mask, "spatial_subtype"].to_numpy()
    valid = adata.obs.loc[mask, "spatial_subtype_valid"].to_numpy()

    assert adjusted_rand_score(true_niche, raw) == 1.0
    assert "unresolved" not in valid  # both real clusters should survive validation
    assert (adata.obs.loc[~mask, "spatial_subtype"] == "NA").all()

def test_end_to_end_overSplitting_gets_trimmed():
    """
    Forces Leiden to over-split a single weak-signal population via an
    aggressive resolution, then checks that validation reduces the
    cluster count and produces a non-trivial "unresolved" group.
    """
    rng = np.random.default_rng(3)
    n_target = 90
    coords_target = np.vstack([rng.normal([0, 0], 5, (30, 2)), rng.normal([200, 200], 5, (60, 2))])
    coords_other = np.vstack([rng.normal([0, 0], 5, (75, 2)), rng.normal([200, 200], 5, (75, 2))])
    coords = np.vstack([coords_target, coords_other])

    n_genes = 10
    expr_target = rng.normal(0, 1, (n_target, n_genes)) * 0.1  # weak own-expression signal
    expr_other = rng.normal(0, 1, (150, n_genes))
    expr_other[:75, 0] += 3.0
    expr_other[75:, 0] -= 3.0
    X = np.vstack([expr_target, expr_other]).astype(np.float32)
    cell_type = np.array(["macrophage"] * n_target + ["other"] * 150)

    adata = AnnData(X=X, obsm={"spatial": coords})
    adata.obs["cell_type"] = cell_type
    adata.var["highly_variable"] = True

    adata = find_spatial_subtypes(
        adata,
        cell_type_key="cell_type", target_type="macrophage",
        graph_kwargs=dict(mode="radius", radius=20.0, decay="gaussian"),
        augment_kwargs=dict(lam=0.7, n_pcs=5),
        cluster_kwargs=dict(resolution=2.0, n_neighbors=10),  # deliberately aggressive
        validate_kwargs=dict(action="drop", n_perm=500),
    )

    mask = (adata.obs["cell_type"] == "macrophage").to_numpy()
    raw = adata.obs.loc[mask, "spatial_subtype"]
    valid = adata.obs.loc[mask, "spatial_subtype_valid"]

    n_raw_clusters = raw.nunique()
    n_valid_clusters = valid[valid != "unresolved"].nunique()

    assert n_raw_clusters >= 5  # confirms over-splitting actually happened
    assert n_valid_clusters < n_raw_clusters
    assert (valid == "unresolved").sum() > 0

def test_cell_type_key_and_target_type_must_be_given_together(macrophage_in_tissue):
    """
    Omitting either cell_type_key or target_type should raise a ValueError.
    """
    adata = macrophage_in_tissue["adata"]
    with pytest.raises(ValueError):
        find_spatial_subtypes(adata, cell_type_key="cell_type", target_type=None)

def test_missing_cell_type_warns(macrophage_in_tissue):
    """
    Omitting cell_type_key/target_type entirely should still run
    (treating the whole adata as target) but must give a warning.
    """
    adata = macrophage_in_tissue["adata"][macrophage_in_tissue["target_mask"]].copy()
    with pytest.warns(UserWarning, match="target population"):
        find_spatial_subtypes(
            adata,
            graph_kwargs=dict(mode="radius", radius=20.0, decay="gaussian"),
            augment_kwargs=dict(lam=0.3, n_pcs=5),
            run_validation=False,
        )