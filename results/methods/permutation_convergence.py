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
HIGHLIGHT_CLUSTERS = ["0", "14", "18", "20", "21"]

def main():
    adata = load_full_tissue()
    target_mask = get_target_mask(adata)
    print(f"Target population: {target_mask.sum()} macrophage bins")

    W_full = build_spatial_graph(adata, spatial_key=SPATIAL_KEY, **GRAPH_KWARGS)
    reference_expression = augment_features(adata, W_full, lam=0.0, target_mask=target_mask, key_added="ref")
    X_aug = augment_features(adata, W_full, lam=LAM, target_mask=target_mask, key_added="main")
    labels = cluster_leiden(X_aug, resolution=RESOLUTION)
    print(f"Fixed raw clustering: {len(set(labels))} clusters (reused across all n_perm values below)")

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
        highlighted = stats[stats["cluster"].astype(str).isin(HIGHLIGHT_CLUSTERS)]
        print(highlighted[["cluster", "p_adj"]].to_string(index=False))

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "permutation_convergence.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    for cluster, group in df.groupby("cluster"):
        if cluster not in HIGHLIGHT_CLUSTERS:
            ax.plot(group["n_perm"], group["p_adj"], color="lightgray", linewidth=1, zorder=1)

    cmap = plt.get_cmap("tab10")
    for i, cluster in enumerate(HIGHLIGHT_CLUSTERS):
        group = df[df["cluster"] == cluster]
        if len(group) == 0:
            continue
        ax.plot(
            group["n_perm"], group["p_adj"], marker="o", color=cmap(i),
            linewidth=2, zorder=2, label=f"cluster {cluster}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("n_perm (log scale)")
    ax.set_ylabel("p_adj")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="alpha=0.05")
    ax.set_title("p_adj stability vs. number of permutations\n(gray = all other clusters, for context)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "permutation_convergence", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()