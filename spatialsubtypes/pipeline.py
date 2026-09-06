"""
pipeline.py — single entrypoint.

Every stage function (build_spatial_graph, augment_features,
cluster_leiden, spatial_validate) remains independently callable.
This module is a convenience wrapper, not a parallel implementation.
"""

import warnings
import numpy as np
from anndata import AnnData

from .augment import augment_features
from .cluster import cluster_leiden
from .graph import build_spatial_graph
from .validate import restrict_to_target, spatial_validate

def find_spatial_subtypes(
    adata: AnnData,
    spatial_key: str = "spatial",
    cell_type_key: str | None = None,
    target_type: str | None = None,
    graph_kwargs: dict | None = None,
    augment_kwargs: dict | None = None,
    cluster_kwargs: dict | None = None,
    validate_kwargs: dict | None = None,
    validation_graph_kwargs: dict | None = None,
    run_validation: bool = True,
    key_added: str = "spatial_subtype",
) -> AnnData:
    """
    Full pipeline: build the full-tissue spatial graph, compute a
    neighbor-augmented embedding for the target cell type, cluster it,
    and (by default) validate candidate subtypes by permutation.

    IMPORTANT: pass the FULL-TISSUE `adata` (all cell types), not a
    pre-subsetted one. Use `cell_type_key`/`target_type` to specify the
    target population.

    Parameters
    adata : AnnData
        Full-tissue object with spatial coordinates in
        `adata.obsm[spatial_key]` and (if subsetting) a cell-type
        column in `adata.obs[cell_type_key]`.
    spatial_key : str
        obsm key for coordinates.
    cell_type_key, target_type : str, optional
        If both given, restricts output (not the graph) to
        `adata.obs[cell_type_key] == target_type`. If neither is
        given, the entire `adata` is treated as the target population.
    graph_kwargs : dict, optional
        Passed to `build_spatial_graph` for the full-tissue augmentation
        graph. E.g. {"mode": "radius", "radius": 50, "decay": "gaussian"}.
    augment_kwargs : dict, optional
        Passed to `augment_features`. E.g. {"lam": 0.3, "n_pcs": 50}.
    cluster_kwargs : dict, optional
        Passed to `cluster_leiden`. E.g. {"resolution": 1.0}.
    validate_kwargs : dict, optional
        Passed to `spatial_validate`. E.g. {"action": "drop", "alpha": 0.05}.
        `reference_expression` is supplied automatically unless explicitly 
        overridden.
    validation_graph_kwargs : dict, optional
        If given, a SEPARATE target-only spatial graph is built via
        `build_spatial_graph` on the target-cell subset, using these
        kwargs, for use in validation instead of restricting the
        augmentation graph. Recommended when `graph_kwargs["mode"] ==
        "knn"`, since a full-tissue kNN graph restricted to target-only
        pairs can be sparse/disconnected. If None, validation reuses the 
        augmentation graph restricted to target-target pairs via `restrict_to_target`.
    run_validation : bool
        If False, skips Part 4 entirely; `key_added` will only contain
        raw Leiden labels.
    key_added : str
        Base name for `adata.obs` columns:
            f"{key_added}"        raw Leiden labels (target cells only,
                                    "NA" elsewhere)
            f"{key_added}_valid"  post-validation labels (only if
                                    run_validation=True)
        And `adata.uns[f"{key_added}_stats"]` holds the validation
        DataFrame if run_validation=True.

    Returns
    adata : AnnData
        The same object, modified in place (and returned for chaining).
    """
    graph_kwargs = dict(graph_kwargs or {})
    augment_kwargs = dict(augment_kwargs or {})
    cluster_kwargs = dict(cluster_kwargs or {})
    validate_kwargs = dict(validate_kwargs or {})

    if (cell_type_key is None) != (target_type is None):
        raise ValueError("cell_type_key and target_type must be given together, or not at all")

    if cell_type_key is not None:
        target_mask = (adata.obs[cell_type_key] == target_type).to_numpy()
        if target_mask.sum() == 0:
            raise ValueError(f"No cells found with {cell_type_key} == {target_type!r}")
    else:
        warnings.warn(
            "No cell_type_key/target_type given. Treating the ENTIRE adata as the "
            "target population. Make sure this is intentional",
            stacklevel=2,
        )
        target_mask = np.ones(adata.n_obs, dtype=bool)

    # Part 1: full-tissue spatial graph (any cell type can be a neighbor)
    W_full = build_spatial_graph(adata, spatial_key=spatial_key, **graph_kwargs)

    # Reference (expression-only) embedding
    ref_kwargs = {k: v for k, v in augment_kwargs.items() if k not in ("lam", "key_added")}
    reference_expression = augment_features(
        adata, W_full, lam=0.0, target_mask=target_mask, key_added=f"{key_added}_reference", **ref_kwargs
    )

    # Part 2: neighbor-augmented embedding at the requested lam
    X_aug = augment_features(
        adata, W_full, target_mask=target_mask, key_added=key_added, **augment_kwargs
    )

    # Part 3: cluster
    labels = cluster_leiden(X_aug, **cluster_kwargs)

    col = np.full(adata.n_obs, "NA", dtype=object)
    col[target_mask] = labels
    adata.obs[key_added] = col

    if not run_validation:
        return adata

    # Part 4: validation graph (target-target only)
    if validation_graph_kwargs is not None:
        adata_target = adata[target_mask].copy()
        W_val = build_spatial_graph(adata_target, spatial_key=spatial_key, **validation_graph_kwargs)
    else:
        W_val = restrict_to_target(W_full, target_mask)

    validate_kwargs.setdefault("reference_expression", reference_expression)
    labels_final, stats = spatial_validate(labels, W_val, **validate_kwargs)

    col_valid = np.full(adata.n_obs, "NA", dtype=object)
    col_valid[target_mask] = labels_final
    adata.obs[f"{key_added}_valid"] = col_valid
    adata.uns[f"{key_added}_stats"] = stats

    return adata