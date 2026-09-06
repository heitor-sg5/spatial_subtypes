"""
Tests for graph.py covering the 9 mode x decay combinations.
"""

import numpy as np
import pytest
from anndata import AnnData

from spatialsubtypes.graph import build_spatial_graph

@pytest.mark.parametrize("mode", ["radius", "knn", "delaunay"])
@pytest.mark.parametrize("decay", [None, "gaussian", "exponential"])
def test_graph_symmetric_and_valid_weight_range(two_blob_coords, mode, decay):
    """
    Every mode x decay combination should produce a symmetric matrix
    with binary weights (exactly 1.0) if decay is None, or continuous
    weights in (0, 1] if decay is set.
    """
    adata = AnnData(X=np.zeros((len(two_blob_coords), 1)), obsm={"spatial": two_blob_coords})
    kwargs = {"mode": mode, "decay": decay}
    if mode == "radius":
        kwargs["radius"] = 15.0
    elif mode == "knn":
        kwargs["n_neighbors"] = 6

    W = build_spatial_graph(adata, **kwargs)

    assert W.nnz > 0, f"mode={mode} decay={decay} produced an empty graph"
    assert abs(W - W.T).max() == 0, f"mode={mode} decay={decay} is not symmetric"

    if decay is None:
        assert np.allclose(W.data, 1.0)
    else:
        assert W.data.min() > 0.0
        assert W.data.max() <= 1.0 + 1e-9

def test_missing_spatial_key_raises(two_blob_coords):
    """
    Calling with a spatial_key that isn't in adata.obsm should raise a KeyError.
    """
    adata = AnnData(X=np.zeros((len(two_blob_coords), 1)))  # no obsm['spatial']
    with pytest.raises(KeyError):
        build_spatial_graph(adata, spatial_key="spatial")

def test_radius_mode_requires_radius(two_blob_coords):
    """
    mode="radius" with radius=None should raise a ValueError.
    """
    adata = AnnData(X=np.zeros((len(two_blob_coords), 1)), obsm={"spatial": two_blob_coords})
    with pytest.raises(ValueError):
        build_spatial_graph(adata, mode="radius", radius=None)