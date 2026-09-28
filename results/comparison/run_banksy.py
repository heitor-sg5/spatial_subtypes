"""
Runs BANKSY on the full tissue dataset, evaluates against ground truth, and prints results.
"""

from __future__ import annotations

import os

import numpy as np

from results.shared import (
    FIGURES_ROOT, SPATIAL_KEY, evaluate_against_ground_truth, get_target_mask, load_full_tissue,
)

BANKSY_LAMBDA = 0.2
BANKSY_RESOLUTION = 1.0
BANKSY_NUM_NEIGHBOURS = 15
BANKSY_MAX_M = 1

def run_banksy(adata=None, target_mask=None):
    """Returns (labels, adata_target, eval_result)."""
    try:
        from banksy.initialize_banksy import initialize_banksy
        from banksy.run_banksy import run_banksy_multiparam
        from banksy_utils.color_lists import spagcn_color
    except ImportError as e:
        raise ImportError(
            "pybanksy is not installed. Run: pip install pybanksy\n"
            "(official package from prabhakarlab/Banksy_py)"
        ) from e

    if adata is None:
        adata = load_full_tissue()
    if target_mask is None:
        target_mask = get_target_mask(adata)

    adata_target = adata[target_mask].copy()

    # BANKSY's coord_keys expects separate x/y columns in .obs
    coords = adata_target.obsm[SPATIAL_KEY]
    adata_target.obs["x"] = coords[:, 0]
    adata_target.obs["y"] = coords[:, 1]
    coord_keys = ("x", "y", SPATIAL_KEY)

    banksy_dict = initialize_banksy(
        adata_target,
        coord_keys,
        num_neighbours=BANKSY_NUM_NEIGHBOURS,
        nbr_weight_decay="scaled_gaussian",
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )

    out_dir = os.path.join(FIGURES_ROOT, "comparison", "banksy_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    color_list = spagcn_color * 10

    results_df = run_banksy_multiparam(
        adata_target,
        banksy_dict,
        lambda_list=[BANKSY_LAMBDA],
        resolutions=[BANKSY_RESOLUTION],
        color_list=color_list,
        max_m=BANKSY_MAX_M,
        filepath=out_dir,
        key=coord_keys,
        annotation_key=None,  
        add_nonspatial=False,  
        savefig=False,
    )

    params_name = results_df.index[0]
    raw_labels = results_df.loc[params_name, "labels"]
    if hasattr(raw_labels, "dense"):
        raw_labels = raw_labels.dense
    labels = np.asarray(raw_labels).astype(str)

    if len(labels) != adata_target.n_obs:
        raise ValueError(
            f"Extracted labels length ({len(labels)}) does not match target "
            f"population ({adata_target.n_obs})."
        )

    eval_result = evaluate_against_ground_truth(labels, adata_target)
    return labels, adata_target, eval_result

def main():
    labels, adata_target, eval_result = run_banksy()
    print(f"BANKSY: {len(set(labels))} clusters")
    print(f"ARI={eval_result['ari']:.3f}  coverage={eval_result['coverage']:.2f}")
    print("\nContingency table (predicted x ground truth):")
    print(eval_result["contingency"])

if __name__ == "__main__":
    main()