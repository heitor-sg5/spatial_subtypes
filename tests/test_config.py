"""
Shared fixtures for the spatialsubtypes testing.
"""

import numpy as np
import pytest
from anndata import AnnData

@pytest.fixture
def two_blob_coords():
    """
    The most basic scenario: 200 points split into two well-separated
    spatial clusters (~280 units apart, blob SD=5).

    Returns a numpy.ndarray:
        coords : (200, 2) array of x/y coordinates
    """
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal([0, 0], 5, size=(100, 2)), rng.normal([200, 200], 5, size=(100, 2))]
    )
    return coords

@pytest.fixture
def macrophage_in_tissue():
    """
    Full tissue scenario with 60 target ("macrophage") cells split
    30/30 across two distant spatial regions, embedded among 200
    "other" cells. Target cells have (noise) identical own
    expression regardless of location, such that any successful spatial
    separation must come from neighbor signal, not target-cell identity.
    "Other" cells carry a real marker gradient (gene 0) that differs by
    region, giving the neighbor-aggregation step something real to pick
    up on.

    Returns a dict with:
        adata        : AnnData, full tissue (260 cells)
        target_mask  : bool array, True for the 60 macrophages
        true_niche   : 0/1 array for the 60 macrophages (30 per region)
    """
    rng = np.random.default_rng(0)
    n_target = 60
    n_other_near, n_other_far = 100, 100

    coords_target = np.vstack(
        [rng.normal([0, 0], 5, (n_target // 2, 2)), rng.normal([200, 200], 5, (n_target // 2, 2))]
    )
    coords_other_a = rng.normal([0, 0], 5, (n_other_near, 2))
    coords_other_b = rng.normal([200, 200], 5, (n_other_far, 2))
    coords = np.vstack([coords_target, coords_other_a, coords_other_b])

    n_genes = 10
    expr_target = rng.normal(0, 1, (n_target, n_genes))  # no spatial signal in own expression
    expr_other_a = rng.normal(0, 1, (n_other_near, n_genes))
    expr_other_a[:, 0] += 3.0
    expr_other_b = rng.normal(0, 1, (n_other_far, n_genes))
    expr_other_b[:, 0] -= 3.0

    X = np.vstack([expr_target, expr_other_a, expr_other_b]).astype(np.float32)
    cell_type = np.array(["macrophage"] * n_target + ["other"] * (n_other_near + n_other_far))

    adata = AnnData(X=X, obsm={"spatial": coords})
    adata.obs["cell_type"] = cell_type
    adata.var["highly_variable"] = True

    target_mask = (adata.obs["cell_type"] == "macrophage").to_numpy()
    true_niche = np.array([0] * (n_target // 2) + [1] * (n_target // 2))

    return {"adata": adata, "target_mask": target_mask, "true_niche": true_niche}

@pytest.fixture
def fake_split_scenario():
    """
    Ambiguous scenario with 90 target-only cells: 30 cells near a 
    real, distant region ("A"); 60 cells in a second dense region, 
    arbitrarily split (no spatial basis) into "B1"/"B2".

    Returns a dict with:
        coords          : (90, 2) array of x/y coordinates
        labels          : (90,) array of "A"/"B1"/"B2" labels
        fake_expression : (90, 3) array of expression values, with no
                            spatial signal in the 60 B cells
    """
    rng = np.random.default_rng(0)
    coords_a = rng.normal([0, 0], 5, (30, 2))
    coords_b = rng.normal([200, 200], 5, (60, 2))
    coords = np.vstack([coords_a, coords_b])

    split_rng = np.random.default_rng(1)
    fake_split = split_rng.choice(["B1", "B2"], size=60)
    labels = np.array(["A"] * 30 + list(fake_split))

    fake_expression = np.zeros((90, 3))
    fake_expression[:30] += [5, 0, 0]
    fake_expression[30:] += rng.normal(0, 0.1, (60, 3))

    return {"coords": coords, "labels": labels, "fake_expression": fake_expression}

@pytest.fixture
def true_split_scenario():
    """
    True split scenario with the same coordinate layout as fake split, 
    but labels and expression are both genuinely split by region.

    Returns a dict with:
        coords     : (90, 2) array of x/y coordinates
        labels     : (90,) array of "A"/"B" labels
        expression : (90, 3) array of expression values, with spatial signal
                     in the 60 B cells
    """
    rng = np.random.default_rng(0)
    coords_a = rng.normal([0, 0], 5, (30, 2))
    coords_b = rng.normal([200, 200], 5, (60, 2))
    coords = np.vstack([coords_a, coords_b])

    labels = np.array(["A"] * 30 + ["B"] * 60)
    expression = np.zeros((90, 3))
    expression[:30] += [5, 0, 0]
    expression[30:] += [0, 5, 0]

    return {"coords": coords, "labels": labels, "expression": expression}