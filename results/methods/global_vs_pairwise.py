"""
Runs global vs pairwise validation on the same raw labels, saves 
CSVs of results, and plots a side-by-side bar chart of -log10(p_adj) 
per cluster for each test mode.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from spatialsubtypes import augment_features, build_spatial_graph, cluster_leiden
from spatialsubtypes.validate import restrict_to_target, spatial_validate
from results.shared import (
    CELL_TYPE_KEY, FIGURES_ROOT, SPATIAL_KEY, TARGET_TYPE, load_full_tissue, savefig,
)

GRAPH_KWARGS = dict(mode="radius", radius=50.0, decay="gaussian")
LAM = 0.3
RESOLUTION = 1.5
N_PERM = 1000

def main():
    """
    Runs global vs pairwise validation, saves as .CSV and plots a comparison bar chart.
    """
    adata = load_full_tissue()
    target_mask = (adata.obs[CELL_TYPE_KEY] == TARGET_TYPE).to_numpy()

    W_full = build_spatial_graph(adata, spatial_key=SPATIAL_KEY, **GRAPH_KWARGS)
    reference_expression = augment_features(
        adata, W_full, lam=0.0, target_mask=target_mask, key_added="ref"
    )
    X_aug = augment_features(adata, W_full, lam=LAM, target_mask=target_mask, key_added="main")
    raw_labels = cluster_leiden(X_aug, resolution=RESOLUTION)
    print(f"Raw clustering: {len(set(raw_labels))} clusters")

    W_target = restrict_to_target(W_full, target_mask)

    _, stats_global = spatial_validate(
        raw_labels, W_target, test_mode="global", action="flag", n_perm=N_PERM,
    )
    _, stats_pairwise = spatial_validate(
        raw_labels, W_target, test_mode="pairwise", action="flag", n_perm=N_PERM,
        reference_expression=reference_expression,
    )

    n_sig_global = int(stats_global["significant"].sum())
    n_sig_pairwise = int(stats_pairwise["significant"].sum())
    print(f"\nGlobal test:   {n_sig_global}/{len(stats_global)} clusters significant")
    print(f"Pairwise test: {n_sig_pairwise}/{len(stats_pairwise)} clusters significant")

    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)
    stats_global.to_csv(os.path.join(out_dir, "global_vs_pairwise_global.csv"), index=False)
    stats_pairwise.to_csv(os.path.join(out_dir, "global_vs_pairwise_pairwise.csv"), index=False)

    # ---- -log10(p_adj) per cluster, global vs pairwise side by side ----
    merged = stats_global[["cluster", "p_adj"]].merge(
        stats_pairwise[["cluster", "p_adj"]], on="cluster", suffixes=("_global", "_pairwise")
    )
    merged["-log10_p_global"] = -np.log10(merged["p_adj_global"].clip(lower=1e-4))
    merged["-log10_p_pairwise"] = -np.log10(merged["p_adj_pairwise"].clip(lower=1e-4))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width / 2, merged["-log10_p_global"], width, label="global")
    ax.bar(x + width / 2, merged["-log10_p_pairwise"], width, label="pairwise")
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="alpha=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["cluster"], rotation=45, ha="right")
    ax.set_ylabel("-log10(p_adj)")
    ax.set_title(f"Validation test mode comparison (resolution={RESOLUTION})")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "global_vs_pairwise", "methods")
    plt.close(fig)

    print("\n" + merged.to_string(index=False))

if __name__ == "__main__":
    main()