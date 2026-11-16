"""Tests for features.py: HVGs, blocks, lambda mixing."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from spatialsubtypes import apply_kernel, build_graph, embed, neighbour_block, prepare_inputs, select_hvg
from spatialsubtypes.features import neighbour_matrix

def _blocks(fix, feature="expression", scaling="input_std", **kw):
    ad, mask = fix["adata"], fix["target_mask"]
    inp = prepare_inputs(ad, mask, n_hvg=10, composition_key="cell_type" if feature == "composition" else None)
    W = apply_kernel(build_graph(ad.obsm["spatial"], "radius", radius=20.0, query=inp.target_idx), "gaussian")
    return inp, W, neighbour_block(inp, W, feature, scaling)

def test_select_hvg_finds_planted_variable_genes():
    """Variance rises with mean in real data; planted genes are extra-variable for their mean."""
    rng = np.random.default_rng(0)
    g, planted = 3000, 100
    mu = rng.lognormal(-1, 1, g)
    sd = 0.4 * np.sqrt(mu)
    sd[:planted] *= 3
    X = (rng.standard_normal((800, g)) * sd + mu).clip(0)
    picked = set(select_hvg(sp.csr_matrix(X), planted))
    assert len(picked & set(range(planted))) >= 90

def test_select_hvg_returns_everything_when_asked_for_more_than_exist():
    X = np.random.default_rng(0).random((50, 8)) + 0.1
    assert len(select_hvg(X, 100)) == 8

def test_own_block_has_unit_total_variance_and_neighbour_block_less(macrophage_in_tissue):
    inp, W, nbr = _blocks(macrophage_in_tissue)
    assert np.isclose(inp.own.var(axis=0).sum(), 1.0, atol=1e-3)
    assert 0 < nbr.var(axis=0).sum() < 1.0, "BANKSY-like scale: averaging shrinks the neighbour block"
    assert np.allclose(nbr.mean(axis=0), 0, atol=1e-5)

def test_unit_variance_option_rescales_the_neighbour_block(macrophage_in_tissue):
    _, _, nbr = _blocks(macrophage_in_tissue, scaling="unit_variance")
    assert np.isclose(nbr.var(axis=0).sum(), 1.0, atol=1e-3)

@pytest.mark.parametrize("scaling", ["input_std", "unit_variance", "per_gene"])
def test_all_scalings_run(macrophage_in_tissue, scaling):
    _, _, nbr = _blocks(macrophage_in_tissue, scaling=scaling)
    assert np.isfinite(nbr).all()

def test_lam_zero_is_plain_pca_of_own_expression(macrophage_in_tissue):
    inp, W, nbr = _blocks(macrophage_in_tissue)
    a = embed(inp.own, nbr, 0.0, n_pcs=5)
    b = PCA(5, random_state=0).fit_transform(inp.own)
    assert abs(np.corrcoef(a[:, 0], b[:, 0])[0, 1]) > 0.9999

def test_lam_out_of_range_raises(macrophage_in_tissue):
    inp, W, nbr = _blocks(macrophage_in_tissue)
    with pytest.raises(ValueError):
        embed(inp.own, nbr, 1.5)

def test_neighbours_come_from_the_whole_tissue(macrophage_in_tissue):
    """Targets have no own niche signal, so separation must come from non-target neighbours."""
    inp, W, nbr = _blocks(macrophage_in_tissue)
    emb = embed(inp.own, nbr, 0.9, n_pcs=5)
    assert silhouette_score(emb, macrophage_in_tissue["true_niche"]) > 0.3
    assert silhouette_score(embed(inp.own, nbr, 0.0, n_pcs=5), macrophage_in_tissue["true_niche"]) < 0.1

def test_increasing_lambda_increases_niche_separability():
    rng = np.random.default_rng(0)
    n = 100
    coords = np.vstack([rng.normal([0, 0], 8, (n, 2)), rng.normal([200, 200], 8, (n, 2))])
    X = rng.normal(0, 1, (2 * n, 20)); X[:n, :5] += 0.3; X[n:, :5] -= 0.3
    ad = AnnData(X=(X + 5).astype(np.float32), obsm={"spatial": coords})
    mask = np.ones(2 * n, bool)
    inp = prepare_inputs(ad, mask, n_hvg=20)
    W = apply_kernel(build_graph(coords, "radius", radius=20.0, query=inp.target_idx), "gaussian")
    nbr = neighbour_block(inp, W)
    niche = np.array([0] * n + [1] * n)
    sil = [silhouette_score(embed(inp.own, nbr, lam, 5), niche) for lam in (0.0, 0.3, 0.6, 0.9)]
    assert sil[3] > sil[2] > sil[1] > sil[0]

def test_isolated_cells_get_the_average_neighbourhood_not_their_own_expression(macrophage_in_tissue):
    ad, mask = macrophage_in_tissue["adata"], macrophage_in_tissue["target_mask"]
    inp = prepare_inputs(ad, mask, n_hvg=10)
    W = apply_kernel(build_graph(ad.obsm["spatial"], "radius", radius=0.001, query=inp.target_idx), "gaussian")
    nbr = neighbour_block(inp, W)
    assert np.allclose(nbr, 0)

def test_composition_rows_are_fractions_summing_to_one(macrophage_in_tissue):
    ad, mask = macrophage_in_tissue["adata"], macrophage_in_tissue["target_mask"]
    inp = prepare_inputs(ad, mask, n_hvg=10, composition_key="cell_type")
    W = apply_kernel(build_graph(ad.obsm["spatial"], "radius", radius=20.0, query=inp.target_idx), "gaussian")
    raw = neighbour_matrix(inp, W, "composition", "per_gene")
    assert raw.shape == (mask.sum(), 2)
    assert np.allclose(raw.sum(axis=1), 1.0) and (raw >= 0).all()

def test_composition_detects_niches_that_differ_in_cell_types():
    rng = np.random.default_rng(1)
    n = 150
    coords = np.vstack([rng.normal([0, 0], 8, (n, 2)), rng.normal([200, 200], 8, (n, 2))])
    types = np.array(["T"] * 60 + ["M"] * 30 + ["CAF"] * 60 + ["T"] * 60 + ["M"] * 30 + ["Epi"] * 60, dtype=object)
    types[:60] = "T"; types[60:90] = "M"; types[90:150] = "CAF"
    types[150:210] = "Epi"; types[210:240] = "M"; types[240:300] = "T"
    ad = AnnData(X=rng.random((2 * n, 6)).astype(np.float32) + 1, obsm={"spatial": coords})
    ad.obs["ct"] = types
    mask = types == "M"
    inp = prepare_inputs(ad, mask, n_hvg=6, composition_key="ct")
    W = apply_kernel(build_graph(coords, "radius", radius=25.0, query=inp.target_idx), "gaussian")
    emb = embed(inp.own, neighbour_block(inp, W, "composition"), 0.95, 3)
    niche = (np.flatnonzero(mask) >= n).astype(int)
    assert silhouette_score(emb, niche) > 0.5

def test_unlabelled_cells_form_an_unassigned_category(macrophage_in_tissue):
    ad, mask = macrophage_in_tissue["adata"], macrophage_in_tissue["target_mask"]
    ad = ad.copy()
    ad.obs["ct"] = ad.obs["cell_type"].astype(object)
    ad.obs.loc[ad.obs_names[100:120], "ct"] = np.nan
    inp = prepare_inputs(ad, mask, n_hvg=10, composition_key="ct")
    assert "unassigned" in inp.categories

def test_composition_without_labels_raises(macrophage_in_tissue):
    ad, mask = macrophage_in_tissue["adata"], macrophage_in_tissue["target_mask"]
    inp = prepare_inputs(ad, mask, n_hvg=10)
    W = apply_kernel(build_graph(ad.obsm["spatial"], "radius", radius=20.0, query=inp.target_idx), "gaussian")
    with pytest.raises(ValueError):
        neighbour_block(inp, W, "composition")
