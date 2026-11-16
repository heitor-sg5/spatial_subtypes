"""
Single entry point: 'find_spatial_subtypes'.

A convenience wrapper around the four stages. Each stage function can still
be called on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cluster import cluster_embedding, knn_graph, resolution_for_n_clusters
from .features import Inputs, embed, neighbour_block, prepare_inputs
from .graph import adjacency, apply_kernel, build_graph
from .validate import validate_subtypes

@dataclass
class SubtypeResult:
    """Output of 'find_spatial_subtypes'. All arrays cover the target cells only."""

    target_idx: np.ndarray  # positions of the target cells in the full tissue
    labels_raw: np.ndarray  # Leiden clusters before validation
    labels: np.ndarray  # validated subtypes (== labels_raw if validate=False)
    embedding: np.ndarray  # lam-mixed PCA embedding that was clustered
    stats: pd.DataFrame | None = None  # validation table (None if validate=False)
    merges: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

def find_spatial_subtypes(
    adata,
    *,
    target_mask=None,
    cell_type_key: str | None = None,
    target_type: str | None = None,
    spatial_key: str = "spatial",
    mode: str = "knn",
    radius: float | None = None,
    n_neighbors: int = 15,
    rings: int = 1,
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    neighbour_feature: str = "expression",
    composition_key: str | None = None,
    nbr_scaling: str = "input_std",
    lam: float = 0.2,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    resolution: float = 1.0,
    n_clusters: int | None = None,
    cluster_neighbors: int = 15,
    validate: bool = True,
    validation_neighbors: int = 10,
    validation_kwargs: dict | None = None,
    layer: str | None = None,
    inputs: Inputs | None = None,
    seed: int = 0,
    key_added: str | None = "spatial_subtype",
) -> SubtypeResult:
    """
    Subtype one cell population using its spatial surroundings.

    Pass the FULL-TISSUE 'adata' (every cell type): the neighbours of a target
    cell can be any cell. Select the targets with 'target_mask' or with
    'cell_type_key' + 'target_type'.

    'inputs' accepts the output of 'prepare_inputs' to reuse gene selection and
    the own block across runs (it must have been built for the same target cells).

    Defaults (kNN with 15 neighbours, Gaussian kernel, lam = 0.2) mirror
    BANKSY's cell-typing setup; the benchmarks in this repo explore better settings.

    Writes 'adata.obs[key_added]' (raw clusters) and
    'adata.obs[key_added + "_validated"]' (subtypes), "NA" outside the target
    cells, unless 'key_added' is None.
    """
    if target_mask is not None:
        if cell_type_key is not None or target_type is not None:
            raise ValueError("give either target_mask or cell_type_key + target_type, not both")
        target_mask = np.asarray(target_mask, dtype=bool)
    elif cell_type_key is not None and target_type is not None:
        target_mask = (adata.obs[cell_type_key] == target_type).to_numpy()
    else:
        raise ValueError("give target_mask, or both cell_type_key and target_type")
    if target_mask.sum() == 0:
        raise ValueError("the target selection is empty")
    if neighbour_feature == "composition" and composition_key is None:
        composition_key = cell_type_key
    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not in adata.obsm; available: {list(adata.obsm)}")

    if inputs is None:
        inputs = prepare_inputs(adata, target_mask, n_hvg=n_hvg, layer=layer,
                                composition_key=composition_key if neighbour_feature == "composition" else None)
    elif not np.array_equal(inputs.target_idx, np.flatnonzero(target_mask)):
        raise ValueError("inputs were prepared for a different target selection")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    idx = inputs.target_idx

    graph = build_graph(coords, mode, radius=radius, n_neighbors=n_neighbors, rings=rings, query=idx)
    n_isolated = int((graph.degree == 0).sum())
    W = apply_kernel(graph, kernel, bandwidth)
    nbr = neighbour_block(inputs, W, neighbour_feature, nbr_scaling)
    emb = embed(inputs.own, nbr, lam, n_pcs, seed)

    g = knn_graph(emb, cluster_neighbors)
    if n_clusters is not None:
        _, labels_raw = resolution_for_n_clusters(g, n_clusters, seed)
    else:
        labels_raw = cluster_embedding(emb, resolution, cluster_neighbors, seed)
    labels_raw = labels_raw.astype(str)

    stats, merges, labels = None, [], labels_raw
    if validate:
        own_emb = emb if lam == 0 else embed(inputs.own, None, 0.0, n_pcs, seed)
        Wv = adjacency(build_graph(coords[idx], "knn", n_neighbors=validation_neighbors, symmetric=True))
        res = validate_subtypes(labels_raw, Wv, own_emb, seed=seed, **(validation_kwargs or {}))
        labels, stats, merges = res.labels, res.stats, res.merges

    if key_added is not None:
        for name, vals in ((key_added, labels_raw), (key_added + "_validated", labels)):
            col = np.full(adata.n_obs, "NA", dtype=object)
            col[idx] = vals
            adata.obs[name] = pd.Categorical(col)

    params = dict(mode=mode, radius=radius, n_neighbors=n_neighbors, rings=rings, kernel=kernel,
                  neighbour_feature=neighbour_feature, nbr_scaling=nbr_scaling, lam=lam,
                  n_hvg=n_hvg, n_pcs=n_pcs, resolution=resolution, n_clusters=n_clusters,
                  mean_degree=float(graph.degree.mean()), n_isolated=n_isolated, seed=seed)
    return SubtypeResult(idx, labels_raw, labels, emb, stats, merges, params)
