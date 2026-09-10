"""
Run all methods (ours, plain Leiden, BANKSY) and compare their results.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from results.comparison.run_ours import run_ours
from results.comparison.run_leiden import run_plain_leiden
from results.shared import FIGURES_ROOT, GROUND_TRUTH_KEY, SPATIAL_KEY, get_target_mask, load_full_tissue, savefig

def _scatter_by_label(ax, coords, labels, title, unresolved_values=("unresolved", "NA")):
    labels = pd.Series(labels)
    is_unresolved = labels.isin(unresolved_values) | labels.isna()
    ax.scatter(coords[is_unresolved, 0], coords[is_unresolved, 1], c="lightgrey", s=4)
    categories = sorted(labels[~is_unresolved].unique())
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(categories):
        mask = (labels == cat).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, color=cmap(i % 10))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("spatial x (um)")
    ax.set_aspect("equal")

def main():
    adata = load_full_tissue()
    target_mask = get_target_mask(adata)
    print(f"Target population: {target_mask.sum()} macrophage bins\n")

    results = {}

    print("=== Ours ===")
    labels_ours, adata_target, eval_ours, _ = run_ours(adata, target_mask)
    results["Ours"] = {"labels": labels_ours, "eval": eval_ours}
    print(f"ARI={eval_ours['ari']:.3f}  coverage={eval_ours['coverage']:.2f}\n")

    print("=== Plain Leiden ===")
    labels_plain, _, eval_plain = run_plain_leiden(adata, target_mask)
    results["Plain Leiden"] = {"labels": labels_plain, "eval": eval_plain}
    print(f"ARI={eval_plain['ari']:.3f}  coverage={eval_plain['coverage']:.2f}\n")

    print("=== BANKSY ===")
    try:
        from results.comparison.run_banksy import run_banksy
        labels_banksy, _, eval_banksy = run_banksy(adata, target_mask)
        results["BANKSY"] = {"labels": labels_banksy, "eval": eval_banksy}
        print(f"ARI={eval_banksy['ari']:.3f}  coverage={eval_banksy['coverage']:.2f}\n")
    except ImportError as e:
        print(f"Skipping BANKSY: {e}\n")

    # ---- Bar chart: ARI across methods ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = list(results.keys())
    aris = [results[m]["eval"]["ari"] for m in methods]
    ax.bar(methods, aris, color=["tab:blue", "tab:gray", "tab:orange"][: len(methods)])
    ax.set_ylabel("ARI vs. Oliveira et al. ground truth")
    ax.set_title("Method comparison")
    fig.tight_layout()
    savefig(fig, "compare_all_ari", "comparison")
    plt.close(fig)

    for m in methods:
        print(f"{m}: ARI={results[m]['eval']['ari']:.3f}")

    # ---- Spatial map: side by side ----
    coords = adata_target.obsm[SPATIAL_KEY]
    ground_truth = adata_target.obs[GROUND_TRUTH_KEY].to_numpy()
    n_panels = len(methods) + 1

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    for ax, m in zip(axes, methods):
        _scatter_by_label(ax, coords, results[m]["labels"], m)
    _scatter_by_label(axes[-1], coords, ground_truth, "Oliveira et al. ground truth", unresolved_values=())
    axes[0].set_ylabel("spatial y (um)")

    fig.tight_layout()
    out_dir = os.path.join(FIGURES_ROOT, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    savefig(fig, "compare_all_spatial_map", "comparison")
    plt.close(fig)

if __name__ == "__main__":
    main()