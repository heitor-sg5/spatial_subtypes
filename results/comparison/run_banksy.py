"""
Runs BANKSY on the full tissue dataset and evaluates 
against other methods.
"""

from __future__ import annotations

import numpy as np

from results.shared import SPATIAL_KEY, evaluate_against_ground_truth, get_target_mask, load_full_tissue

BANKSY_LAMBDA = 0.2
BANKSY_RESOLUTION = 1.0
BANKSY_NUM_NEIGHBOURS = 15
LABEL_COLUMN_HINT = "labels" 

def run_banksy(adata=None, target_mask=None):
    """
    Returns (labels, adata_target, eval_result).
    """
    try:
        from banksy.initialize_banksy import initialize_banksy
        from banksy.run_banksy import run_banksy_multiparam
    except ImportError as e:
        raise ImportError(
            "pybanksy is not installed. Run: pip install pybanksy\n"
        ) from e

    if adata is None:
        adata = load_full_tissue()
    if target_mask is None:
        target_mask = get_target_mask(adata)

    adata_target = adata[target_mask].copy()
    coords = adata_target.obsm[SPATIAL_KEY]
    adata_target.obs["x"] = coords[:, 0]
    adata_target.obs["y"] = coords[:, 1]
    coord_keys = ("x", "y", SPATIAL_KEY)

    banksy_dict = initialize_banksy(
        adata_target,
        coord_keys,
        num_neighbours=BANKSY_NUM_NEIGHBOURS,
        nbr_weight_decay="scaled_gaussian",
    )

    results_df = run_banksy_multiparam(
        adata_target,
        banksy_dict,
        lambda_list=[BANKSY_LAMBDA],
        resolutions=[BANKSY_RESOLUTION],
    )

    print("run_banksy_multiparam returned columns:", list(results_df.columns))
    label_cols = [c for c in results_df.columns if LABEL_COLUMN_HINT in c.lower()]
    if len(label_cols) == 0:
        raise KeyError(
            f"No column containing '{LABEL_COLUMN_HINT}' found in results_df. "
            f"Actual columns: {list(results_df.columns)}. Update LABEL_COLUMN_HINT "
            f"or extract labels manually from this DataFrame."
        )
    label_col = label_cols[0]
    print(f"Using column '{label_col}' as cluster labels.")

    labels = results_df[label_col].to_numpy().astype(str)
    if len(labels) != adata_target.n_obs:
        raise ValueError(
            f"Extracted labels length ({len(labels)}) does not match target "
            f"population ({adata_target.n_obs}). Results_df may be indexed "
            f"differently than adata_target; check row order/alignment manually."
        )

    eval_result = evaluate_against_ground_truth(labels, adata_target)
    return labels, adata_target, eval_result

def main():
    labels, adata_target, eval_result = run_banksy()
    print(f"\nBANKSY: {len(set(labels))} clusters")
    print(f"ARI={eval_result['ari']:.3f}  coverage={eval_result['coverage']:.2f}")
    print("\nContingency table (predicted x ground truth):")
    print(eval_result["contingency"])

if __name__ == "__main__":
    main()