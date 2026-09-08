"""
Run the global vs pairwise validation test comparison on the full tissue, 
using the finalized default parameters (lam=0.3, resolution=1.5). 
"""

from __future__ import annotations
 
import os
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from spatialsubtypes import augment_features, build_spatial_graph, cluster_leiden
from spatialsubtypes.validate import restrict_to_target, spatial_validate
from results.shared import FIGURES_ROOT, SPATIAL_KEY, get_target_mask, load_full_tissue, savefig
 
GRAPH_KWARGS = dict(mode="delaunay", decay=None)
LAM = 0.3
RESOLUTION = 1.5 
N_PERM = 1000
ALPHA = 0.05
 
def main():
    adata = load_full_tissue()
    target_mask = get_target_mask(adata)
    print(f"Target population: {target_mask.sum()} macrophage bins")
 
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
 
    merged = stats_global[["cluster", "p_adj"]].merge(
        stats_pairwise[["cluster", "p_adj"]], on="cluster", suffixes=("_global", "_pairwise")
    )
    merged["sig_global"] = merged["p_adj_global"] < ALPHA
    merged["sig_pairwise"] = merged["p_adj_pairwise"] < ALPHA
 
    def category(row):
        if row["sig_global"] and row["sig_pairwise"]:
            return "agree: significant"
        if not row["sig_global"] and not row["sig_pairwise"]:
            return "agree: not significant"
        if row["sig_global"] and not row["sig_pairwise"]:
            return "pairwise downgrades"
        return "global downgrades"
 
    merged["category"] = merged.apply(category, axis=1)
    print("\n" + merged[["cluster", "p_adj_global", "p_adj_pairwise", "category"]].to_string(index=False))
 
    n_disagree = (merged["category"] == "pairwise downgrades").sum()
    print(f"\n{n_disagree}/{len(merged)} clusters: significant under global, "
          f"NOT significant under pairwise")
 
    merged["neglog10_global"] = -np.log10(merged["p_adj_global"].clip(lower=1e-4))
    merged["neglog10_pairwise"] = -np.log10(merged["p_adj_pairwise"].clip(lower=1e-4))
    alpha_line = -np.log10(ALPHA)
    axis_max = max(merged["neglog10_global"].max(), merged["neglog10_pairwise"].max()) * 1.15
 
    fig, ax = plt.subplots(figsize=(6.5, 6))
 
    ax.axvspan(alpha_line, axis_max, color="tab:blue", alpha=0.04, zorder=0)
    ax.axhspan(alpha_line, axis_max, color="tab:blue", alpha=0.04, zorder=0)
 
    colors = {
        "agree: significant": "tab:blue",
        "agree: not significant": "tab:gray",
        "pairwise downgrades": "tab:red",
        "global downgrades": "tab:purple",
    }
    for cat, group in merged.groupby("category"):
        ax.scatter(
            group["neglog10_global"], group["neglog10_pairwise"],
            color=colors[cat], label=f"{cat} (n={len(group)})", s=50, zorder=3, edgecolor="white",
        )
 
    ax.axvline(alpha_line, color="black", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(alpha_line, color="black", linestyle="--", linewidth=1, zorder=1)
    ax.plot([0, axis_max], [0, axis_max], color="lightgray", linewidth=1, zorder=1, label="y = x (perfect agreement)")
 
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_xlabel("-log10(p_adj), global test")
    ax.set_ylabel("-log10(p_adj), pairwise test")
    ax.set_title(
        f"Global vs. pairwise validation (resolution={RESOLUTION})\n"
        f"dashed lines = alpha={ALPHA}; shaded region = significant"
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "global_vs_pairwise", "methods")
    plt.close(fig)


if __name__ == "__main__":
    main()