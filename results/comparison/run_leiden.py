"""
Run plain Leiden clustering (expression only) on the full tissue 
dataset and evaluate against other methods.
"""

from __future__ import annotations

from spatialsubtypes import find_spatial_subtypes
from results.shared import SPATIAL_KEY, evaluate_against_ground_truth, get_target_mask, load_full_tissue

RESOLUTION = 1.0 

def run_plain_leiden(adata=None, target_mask=None):
    """Returns (labels, adata_target, eval_result)."""
    if adata is None:
        adata = load_full_tissue()
    if target_mask is None:
        target_mask = get_target_mask(adata)

    a = find_spatial_subtypes(
        adata.copy(),
        spatial_key=SPATIAL_KEY,
        target_mask=target_mask,
        graph_kwargs=dict(mode="delaunay", decay=None),
        augment_kwargs=dict(lam=0.0, n_pcs=50),
        cluster_kwargs=dict(resolution=RESOLUTION),
        run_validation=False,
        key_added="plain_leiden",
    )
    labels = a.obs.loc[target_mask, "plain_leiden"].to_numpy()
    adata_target = a[target_mask]
    eval_result = evaluate_against_ground_truth(labels, adata_target)
    return labels, adata_target, eval_result

def main():
    labels, adata_target, eval_result = run_plain_leiden()
    print(f"Plain Leiden (expression only): {len(set(labels))} clusters")
    print(f"ARI={eval_result['ari']:.3f}  coverage={eval_result['coverage']:.2f}")
    print("\nContingency table (predicted x ground truth):")
    print(eval_result["contingency"])

if __name__ == "__main__":
    main()