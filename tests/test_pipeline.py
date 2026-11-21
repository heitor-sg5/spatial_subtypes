"""End-to-end tests of find_spatial_subtypes."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score

from spatialsubtypes import find_spatial_subtypes

KW = dict(mode="radius", radius=20.0, kernel="gaussian", n_hvg=10, n_pcs=5, cluster_neighbors=10,
          validation_neighbors=6, validation_kwargs=dict(n_perm=300, min_cluster_size=10))

def test_end_to_end_recovers_the_two_subtypes(macrophage_subtypes_in_tissue):
    fix = macrophage_subtypes_in_tissue
    ad = fix["adata"]
    r = find_spatial_subtypes(ad, cell_type_key="cell_type", target_type="macrophage",
                              lam=0.7, resolution=0.5, **KW)
    assert adjusted_rand_score(fix["true_niche"], r.labels_raw) == 1.0
    assert adjusted_rand_score(fix["true_niche"], r.labels) == 1.0 and r.merges == []
    assert (ad.obs.loc[~fix["target_mask"], "spatial_subtype"] == "NA").all()
    assert (ad.obs.loc[fix["target_mask"], "spatial_subtype_validated"] != "NA").all()

def test_niche_without_transcriptional_difference_is_clustered_but_not_kept(macrophage_in_tissue):
    """
    Cells that only differ by their surroundings are separated by the clustering, then
    merged by validation, because a subtype must also differ in its own expression.
    """
    ad = macrophage_in_tissue["adata"]
    r = find_spatial_subtypes(ad, cell_type_key="cell_type", target_type="macrophage",
                              lam=0.7, resolution=0.5, **KW)
    assert adjusted_rand_score(macrophage_in_tissue["true_niche"], r.labels_raw) == 1.0
    assert len(set(r.labels)) == 1
    assert r.merges and r.merges[0][2] == "not distinct in own expression"

def test_composition_feature_runs_end_to_end(macrophage_in_tissue):
    ad = macrophage_in_tissue["adata"].copy()
    r = find_spatial_subtypes(ad, cell_type_key="cell_type", target_type="macrophage",
                              neighbour_feature="composition", lam=0.5, resolution=0.5, **KW)
    assert len(r.labels) == macrophage_in_tissue["target_mask"].sum()

@pytest.mark.parametrize("mode,extra", [("knn", dict(n_neighbors=10)), ("delaunay", dict(rings=2))])
@pytest.mark.parametrize("kernel", ["gaussian", "exponential", "reciprocal", "uniform"])
def test_every_graph_and_kernel_combination_runs(macrophage_in_tissue, mode, extra, kernel):
    ad = macrophage_in_tissue["adata"].copy()
    kw = {**KW, "mode": mode, "kernel": kernel, **extra}
    r = find_spatial_subtypes(ad, target_mask=macrophage_in_tissue["target_mask"], lam=0.5,
                              resolution=0.5, **kw)
    assert len(r.labels) == 60

def test_n_clusters_is_honoured(macrophage_in_tissue):
    ad = macrophage_in_tissue["adata"].copy()
    r = find_spatial_subtypes(ad, target_mask=macrophage_in_tissue["target_mask"], lam=0.5,
                              n_clusters=3, validate=False, **{k: v for k, v in KW.items() if k != "validation_kwargs"})
    assert len(set(r.labels_raw)) == 3 and r.stats is None

def test_noise_only_target_population_yields_no_subtypes():
    rng = np.random.default_rng(3)
    n = 1200
    ad = AnnData(X=(rng.normal(size=(n, 30)) + 5).astype(np.float32),
                 obsm={"spatial": rng.uniform(0, 500, (n, 2))})
    ad.obs["ct"] = "m"
    r = find_spatial_subtypes(ad, cell_type_key="ct", target_type="m", mode="knn", n_neighbors=10,
                              lam=0.6, resolution=1.0, n_hvg=30, n_pcs=15,
                              validation_kwargs=dict(n_perm=200))
    assert len(set(r.labels_raw)) >= 5 and len(set(r.labels)) == 1

def test_argument_errors(macrophage_in_tissue):
    ad = macrophage_in_tissue["adata"]
    with pytest.raises(ValueError):
        find_spatial_subtypes(ad)
    with pytest.raises(ValueError):
        find_spatial_subtypes(ad, target_mask=macrophage_in_tissue["target_mask"], cell_type_key="cell_type",
                              target_type="macrophage")
    with pytest.raises(KeyError):
        find_spatial_subtypes(ad, cell_type_key="cell_type", target_type="macrophage", spatial_key="nope")
