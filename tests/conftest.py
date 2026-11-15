"""Shared synthetic fixtures. Everything here is simulated; no real data is needed."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


@pytest.fixture
def two_blob_coords():
    """200 points in two well-separated spatial blobs (SD 5, centres 280 apart)."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal([0, 0], 5, (100, 2)), rng.normal([200, 200], 5, (100, 2))])


@pytest.fixture
def grid_coords():
    """60 x 60 grid of bins 8 units apart (like Visium HD 8 um bins)."""
    return np.stack(np.meshgrid(np.arange(60), np.arange(60)), -1).reshape(-1, 2) * 8.0


@pytest.fixture
def macrophage_in_tissue():
    """60 target cells split 30/30 across two distant regions, among 200 other cells.

    Target cells have identical own-expression statistics in both regions, so a
    successful split must come from the neighbours. The other cells carry a
    marker gene (gene 0) that differs by region.
    """
    rng = np.random.default_rng(0)
    n_target, n_a, n_b, n_genes = 60, 100, 100, 10
    coords = np.vstack([
        rng.normal([0, 0], 5, (n_target // 2, 2)), rng.normal([200, 200], 5, (n_target // 2, 2)),
        rng.normal([0, 0], 5, (n_a, 2)), rng.normal([200, 200], 5, (n_b, 2)),
    ])
    expr_t = rng.normal(0, 1, (n_target, n_genes))
    expr_a = rng.normal(0, 1, (n_a, n_genes)); expr_a[:, 0] += 3.0
    expr_b = rng.normal(0, 1, (n_b, n_genes)); expr_b[:, 0] -= 3.0
    X = np.clip(np.vstack([expr_t, expr_a, expr_b]) + 4, 0, None).astype(np.float32)
    adata = AnnData(X=X, obsm={"spatial": coords})
    adata.obs["cell_type"] = ["macrophage"] * n_target + ["other"] * (n_a + n_b)
    mask = (adata.obs["cell_type"] == "macrophage").to_numpy()
    niche = np.array([0] * (n_target // 2) + [1] * (n_target // 2))
    return {"adata": adata, "target_mask": mask, "true_niche": niche}


@pytest.fixture
def fake_split_scenario():
    """90 target-only cells: 30 in a distant region "A"; 60 in region "B" split at random into B1/B2."""
    rng = np.random.default_rng(0)
    coords = np.vstack([rng.normal([0, 0], 5, (30, 2)), rng.normal([200, 200], 5, (60, 2))])
    split = np.random.default_rng(1).choice(["B1", "B2"], size=60)
    labels = np.array(["A"] * 30 + list(split))
    emb = np.zeros((90, 3)); emb[:30] += [5, 0, 0]; emb[30:] += rng.normal(0, 0.1, (60, 3))
    emb[:30] += rng.normal(0, 0.1, (30, 3))
    return {"coords": coords, "labels": labels, "embedding": emb}


@pytest.fixture
def true_split_scenario():
    """Same layout as fake_split_scenario, but labels and expression genuinely differ by region."""
    rng = np.random.default_rng(0)
    coords = np.vstack([rng.normal([0, 0], 5, (30, 2)), rng.normal([200, 200], 5, (60, 2))])
    labels = np.array(["A"] * 30 + ["B"] * 60)
    emb = np.zeros((90, 3)); emb[:30] += [5, 0, 0]; emb[30:] += [0, 5, 0]
    emb += rng.normal(0, 0.3, (90, 3))
    return {"coords": coords, "labels": labels, "embedding": emb}


@pytest.fixture
def macrophage_subtypes_in_tissue(macrophage_in_tissue):
    """Like macrophage_in_tissue, but the two target groups also differ a little in their own
    expression (genes 1-3 shifted by +/-1.5), as real transcriptional subtypes would."""
    ad = macrophage_in_tissue["adata"].copy()
    niche = macrophage_in_tissue["true_niche"]
    X = np.asarray(ad.X).copy()
    X[: len(niche), 1:4] += np.where(niche[:, None] == 1, 1.5, -1.5)
    ad.X = X.astype(np.float32)
    return {**macrophage_in_tissue, "adata": ad}
