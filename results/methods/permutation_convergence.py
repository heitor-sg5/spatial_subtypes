"""
Fixes the number of permutations used in spatial_validate and checks 
whether the adjusted p-values for the highlighted clusters are stable 
across a range of n_perm values. 
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from spatialsubtypes import augment_features, build_spatial_graph, cluster_leiden
from spatialsubtypes.validate import restrict_to_target, spatial_validate
from results.shared import FIGURES_ROOT, SPATIAL_KEY, get_target_mask, load_full_tissue, savefig

GRAPH_KWARGS = dict(mode="delaunay", decay=None)
LAM = 0.3
RESOLUTION = 1.5
N_PERM_VALUES = [100, 250, 500, 1000, 2500, 5000]
ALPHA = 0.05

def main():
    adata = load_full_tissue()
    target_mask = get_target_mask(adata)
    print(f"Target population: {target_mask.sum()} macrophage bins")

    W_full = build_spatial_graph(adata, spatial_key=SPATIAL_KEY, **GRAPH_KWARGS)
    reference_expression = augment_features(adata, W_full, lam=0.0, target_mask=target_mask, key_added="ref")
    X_aug = augment_features(adata, W_full, lam=LAM, target_mask=target_mask, key_added="main")
    labels = cluster_leiden(X_aug, resolution=RESOLUTION)
    n_clusters = len(set(labels))
    print(f"Fixed raw clustering: {n_clusters} clusters")

    W_target = restrict_to_target(W_full, target_mask)

    rows = []
    for n_perm in N_PERM_VALUES:
        print(f"\n--- n_perm={n_perm} ---")
        _, stats = spatial_validate(
            labels, W_target, test_mode="pairwise", action="flag", n_perm=n_perm,
            reference_expression=reference_expression, random_state=0,
        )
        for _, row in stats.iterrows():
            rows.append({"n_perm": n_perm, "cluster": str(row["cluster"]), "p_adj": row["p_adj"]})
        print(stats[["cluster", "p_adj"]].to_string(index=False))

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab20")

    for i, (cluster, group) in enumerate(df.groupby("cluster")):
        group = group.sort_values("n_perm")
        ax.plot(
            group["n_perm"], group["p_adj"], marker="o", color=cmap(i % 20),
            linewidth=1.5, label=f"cluster {cluster}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("n_perm (log scale)")
    ax.set_ylabel("p_adj")
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1, label=f"alpha={ALPHA}")
    ax.set_title(f"p_adj stability vs. number of permutations")
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    savefig(fig, "permutation_convergence", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()