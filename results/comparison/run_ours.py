"""
Run our method on the full tissue dataset and evaluate 
against other methods.
"""

from __future__ import annotations

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    SPATIAL_KEY, evaluate_against_ground_truth, get_target_mask, load_full_tissue,
)

GRAPH_KWARGS = dict(mode="delaunay", decay=None)
LAM = 0.3
RESOLUTION = 1.0
VALIDATE_KWARGS = dict(action="drop", test_mode="pairwise", n_perm=1000)

def run_ours(adata=None, target_mask=None):
    """
    Returns (labels_valid, adata_target, eval_result, raw_labels).
    labels_valid uses "unresolved" for dropped clusters
    """
    if adata is None:
        adata = load_full_tissue()
    if target_mask is None:
        target_mask = get_target_mask(adata)

    a = find_spatial_subtypes(
        adata.copy(),
        spatial_key=SPATIAL_KEY,
        target_mask=target_mask,
        graph_kwargs=GRAPH_KWARGS,
        augment_kwargs=dict(lam=LAM, n_pcs=50),
        cluster_kwargs=dict(resolution=RESOLUTION),
        validate_kwargs=VALIDATE_KWARGS,
        key_added="ours",
    )
    raw_labels = a.obs.loc[target_mask, "ours"].to_numpy()
    labels_valid = a.obs.loc[target_mask, "ours_valid"].to_numpy()
    adata_target = a[target_mask]

    eval_result = evaluate_against_ground_truth(labels_valid, adata_target)
    return labels_valid, adata_target, eval_result, raw_labels

def main():
    labels_valid, adata_target, eval_result, raw_labels = run_ours()
    n_raw = len(set(raw_labels))
    n_valid = len(set(labels_valid) - {"unresolved"})
    print(f"Ours: {n_raw} raw clusters -> {n_valid} validated clusters")
    print(f"ARI={eval_result['ari']:.3f}  coverage={eval_result['coverage']:.2f}  "
          f"n_evaluated={eval_result['n_evaluated']}/{eval_result['n_ground_truth']}")
    print("\nContingency table (predicted x ground truth):")
    print(eval_result["contingency"])

if __name__ == "__main__":
    main()