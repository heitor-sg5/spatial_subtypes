"""Tests for validate.py."""

from __future__ import annotations

import numpy as np
import pytest

from spatialsubtypes import adjacency, build_graph, validate_subtypes
from spatialsubtypes.validate import bh_adjust

def _W(coords, k=8):
    return adjacency(build_graph(coords, "knn", n_neighbors=k, symmetric=True))

def test_true_split_survives(true_split_scenario):
    s = true_split_scenario
    r = validate_subtypes(s["labels"], _W(s["coords"]), s["embedding"], n_perm=500, min_cluster_size=10)
    assert len(set(r.labels)) == 2 and r.merges == [] and r.stats["passes"].all()

def test_arbitrary_split_inside_one_region_is_merged(fake_split_scenario):
    s = fake_split_scenario
    r = validate_subtypes(s["labels"], _W(s["coords"]), s["embedding"], n_perm=500, min_cluster_size=10)
    assert len(set(r.labels)) == 2
    sizes = sorted(np.unique(r.labels, return_counts=True)[1])
    assert sizes == [30, 60], "B1 and B2 should merge; A should stay"

def test_no_cell_is_dropped_and_labels_stay_aligned(fake_split_scenario):
    s = fake_split_scenario
    r = validate_subtypes(s["labels"], _W(s["coords"]), s["embedding"], n_perm=200, min_cluster_size=10)
    assert len(r.labels) == len(s["labels"])
    assert (r.labels[:30] == r.labels[0]).all() and r.labels[0] != r.labels[-1]

def test_singleton_cluster_is_merged_not_reported_significant(true_split_scenario):
    s = true_split_scenario
    labels = s["labels"].copy(); labels[40] = "solo"
    r = validate_subtypes(labels, _W(s["coords"]), s["embedding"], n_perm=200, min_cluster_size=10)
    assert "solo" not in set(r.labels) and len(r.labels) == 90
    assert any(reason == "too small" for _, _, reason in r.merges)

def test_single_cluster_input_returns_cleanly(true_split_scenario):
    s = true_split_scenario
    r = validate_subtypes(np.array(["a"] * 90), _W(s["coords"]), s["embedding"], n_perm=50)
    assert set(r.labels) == {"0"} and len(r.stats) == 1 and not r.stats["passes"].iloc[0]

def test_pure_noise_clusters_do_not_survive():
    from spatialsubtypes import apply_kernel, embed, neighbour_block, prepare_inputs, cluster_embedding
    from anndata import AnnData
    rng = np.random.default_rng(1)
    n = 1500
    coords = rng.uniform(0, 600, (n, 2))
    ad = AnnData(X=(rng.normal(size=(n, 40)) + 5).astype(np.float32), obsm={"spatial": coords})
    mask = np.ones(n, bool)
    inp = prepare_inputs(ad, mask, n_hvg=40)
    W = apply_kernel(build_graph(coords, "knn", n_neighbors=10, query=inp.target_idx), "gaussian")
    nbr = neighbour_block(inp, W)
    own_emb = embed(inp.own, None, 0.0, 15)
    for lam in (0.5, 0.7):
        emb = embed(inp.own, nbr, lam, 15)
        labels = cluster_embedding(emb, 1.0, 15)
        assert len(set(labels)) >= 6, "test needs a real over-clustering to be meaningful"
        r = validate_subtypes(labels, _W(coords, 10), own_emb, n_perm=200)
        assert len(set(r.labels)) <= 1, f"lam={lam}: {len(set(r.labels))} noise clusters survived"

def test_shape_mismatch_raises(fake_split_scenario):
    s = fake_split_scenario
    with pytest.raises(ValueError):
        validate_subtypes(s["labels"][:-1], _W(s["coords"]), s["embedding"])

def test_bh_adjust_matches_hand_computation():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(bh_adjust(p), [0.02, 0.04, 0.04, 0.02])
    assert np.isnan(bh_adjust(np.array([np.nan, 0.5]))[0])
