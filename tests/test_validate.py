"""
Tests for validate.py covering the global and pairwise modes, and the drop/merge actions.
"""

import numpy as np
import pytest
from anndata import AnnData

from spatialsubtypes.graph import build_spatial_graph
from spatialsubtypes.validate import restrict_to_target, spatial_validate

def _build_W(coords, **kwargs):
    """
    Helper to build a spatial graph from just coordinates.
    """
    adata = AnnData(X=np.zeros((len(coords), 1)), obsm={"spatial": coords})
    return build_spatial_graph(adata, **kwargs)

def test_global_positive_control(true_split_scenario):
    """
    Labels that match the true spatial regions should come back
    strongly significant under the global test.
    """
    W = _build_W(true_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    _, stats = spatial_validate(
        true_split_scenario["labels"], W, test_mode="global", action="flag", n_perm=1000
    )
    assert (stats["p_adj"] < 0.01).all()

def test_global_negative_control(true_split_scenario):
    """
    Same cluster sizes, but labels randomly scrambled across the full
    spatial extent, should not be significant under the global test.
    """
    rng = np.random.default_rng(42)
    scrambled = rng.permutation(true_split_scenario["labels"])
    W = _build_W(true_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    _, stats = spatial_validate(scrambled, W, test_mode="global", action="flag", n_perm=1000)
    assert (stats["p_adj"] > 0.05).all()

def test_global_mode_has_known_blind_spot(fake_split_scenario):
    """
    Under one-vs-rest null, an arbitrary split (B1/B2) confined entirely to
    one dense spatial region comes back significant, indistinguishable
    from the genuinely separate region (A), since both are being
    compared against a null that includes the whole population.
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    _, stats = spatial_validate(
        fake_split_scenario["labels"], W, test_mode="global", action="flag", n_perm=1000
    )
    b_rows = stats[stats["cluster"].isin(["B1", "B2"])]
    assert (b_rows["p_adj"] < 0.05).all()

def test_pairwise_mode_rejects_fake_split(fake_split_scenario):
    """
    Under test_mode='pairwise', the arbitrary B1/B2 split should
    correctly come back non-significant.
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    _, stats = spatial_validate(
        fake_split_scenario["labels"], W, test_mode="pairwise", action="flag",
        reference_expression=fake_split_scenario["fake_expression"], n_perm=1000,
    )
    b_rows = stats[stats["cluster"].isin(["B1", "B2"])]
    assert (b_rows["p_adj"] > 0.05).all()

def test_pairwise_mode_retains_real_split(true_split_scenario):
    """
    Check the pairwise fix must not cost sensitivity on a genuine spatial split.
    """
    W = _build_W(true_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    _, stats = spatial_validate(
        true_split_scenario["labels"], W, test_mode="pairwise", action="flag",
        reference_expression=true_split_scenario["expression"], n_perm=1000,
    )
    assert (stats["p_adj"] < 0.01).all()

def test_drop_produces_unresolved(fake_split_scenario):
    """
    The arbitrary B1/B2 split should be dropped to "unresolved" under
    action="drop".
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    labels_final, _ = spatial_validate(
        fake_split_scenario["labels"], W, test_mode="pairwise", action="drop",
        reference_expression=fake_split_scenario["fake_expression"], n_perm=1000,
    )
    assert set(np.unique(labels_final)) == {"A", "unresolved"}
    assert (labels_final == "unresolved").sum() == 60

def test_merge_collapses_mutual_failures_to_one_label(fake_split_scenario):
    """
    B1 and B2 are mutual nearest-siblings and both fail the
    test, union-find should merge them into a single label, 
    compared to a naive independent reassignment which would
    swap their labels.
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    labels_final, _ = spatial_validate(
        fake_split_scenario["labels"], W, test_mode="pairwise", action="merge",
        reference_expression=fake_split_scenario["fake_expression"], n_perm=1000,
    )
    unique_labels, counts = np.unique(labels_final, return_counts=True)
    assert len(unique_labels) == 2, f"expected B1+B2 merged into one label, got {dict(zip(unique_labels, counts))}"
    assert set(counts) == {30, 60}

def test_merge_requires_reference_expression(fake_split_scenario):
    """
    action="merge" requires reference_expression to be given.
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    with pytest.raises(ValueError):
        spatial_validate(fake_split_scenario["labels"], W, action="merge", test_mode="global")

def test_pairwise_requires_reference_expression(fake_split_scenario):
    """
    test_mode="pairwise" requires reference_expression to be given.
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    with pytest.raises(ValueError):
        spatial_validate(fake_split_scenario["labels"], W, test_mode="pairwise")

def test_restrict_to_target_preserves_edge_weights(macrophage_in_tissue):
    """
    restrict_to_target should be equivalent to building the graph
    directly on just the target cells' coordinates.
    """
    adata = macrophage_in_tissue["adata"]
    target_mask = macrophage_in_tissue["target_mask"]

    W_full = build_spatial_graph(adata.copy(), mode="radius", radius=20.0, decay="gaussian")
    W_restricted = restrict_to_target(W_full, target_mask)

    adata_target_only = adata[target_mask].copy()
    W_direct = build_spatial_graph(adata_target_only, mode="radius", radius=20.0, decay="gaussian")

    assert W_restricted.shape == W_direct.shape
    assert np.allclose(W_restricted.toarray(), W_direct.toarray())

def test_w_shape_mismatch_raises(fake_split_scenario):
    """
    W and labels must have the same number of rows; otherwise a ValueError is raised.
    """
    W = _build_W(fake_split_scenario["coords"], mode="radius", radius=20.0, decay="gaussian")
    bad_labels = fake_split_scenario["labels"][:-1]  # wrong length
    with pytest.raises(ValueError):
        spatial_validate(bad_labels, W, test_mode="global")