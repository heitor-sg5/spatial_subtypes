"""
Isolates the effect of graph construction on result quality:
for each of the 9 mode x decay combinations, builds the graph, 
augments (fixed lam), clusters (fixed resolution), and scores 
labels against Oliveira's ground truth.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    CELL_TYPE_KEY, FIGURES_ROOT, SPATIAL_KEY, TARGET_TYPE,
    evaluate_against_ground_truth, load_full_tissue, savefig,
)

MODES = ["radius", "knn", "delaunay"]
DECAYS = [None, "gaussian", "exponential"]
RADIUS = 50.0
N_NEIGHBORS = 6
LAM = 0.3
RESOLUTION = 1.0

def graph_kwargs_for(mode: str, decay: str | None) -> dict:
    """
    Returns a dict of kwargs to pass to `find_spatial_subtypes`.  
    """
    kwargs = {"mode": mode, "decay": decay}
    if mode == "radius":
        kwargs["radius"] = RADIUS
    elif mode == "knn":
        kwargs["n_neighbors"] = N_NEIGHBORS
    return kwargs

def main():
    """
    Runs the mode x decay grid search, saves a CSV of results, and plots a grouped bar chart.
    """
    adata = load_full_tissue()
    target_mask = (adata.obs[CELL_TYPE_KEY] == TARGET_TYPE).to_numpy()
    print(f"Target population: {target_mask.sum()} macrophage bins")

    rows = []
    for mode in MODES:
        for decay in DECAYS:
            key = f"grid_{mode}_{decay}"
            print(f"\n--- mode={mode}, decay={decay} ---")
            a = find_spatial_subtypes(
                adata.copy(),
                spatial_key=SPATIAL_KEY,
                cell_type_key=CELL_TYPE_KEY, target_type=TARGET_TYPE,
                graph_kwargs=graph_kwargs_for(mode, decay),
                augment_kwargs=dict(lam=LAM, n_pcs=50),
                cluster_kwargs=dict(resolution=RESOLUTION),
                run_validation=False,
                key_added=key,
            )
            labels_target = a.obs.loc[target_mask, key].to_numpy()
            adata_target = a[target_mask]
            result = evaluate_against_ground_truth(labels_target, adata_target)

            n_clusters = len(set(labels_target))
            print(f"  n_clusters={n_clusters}  ARI={result['ari']:.3f}  coverage={result['coverage']:.2f}")
            rows.append({
                "mode": mode, "decay": str(decay), "n_clusters": n_clusters,
                "ari": result["ari"], "coverage": result["coverage"],
            })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "mode_decay_grid.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table: {out_csv}")
    print(df.to_string(index=False))

    # ---- Grouped bar chart: ARI by mode, grouped by decay ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(MODES))
    width = 0.25
    for i, decay in enumerate(DECAYS):
        vals = [df[(df["mode"] == m) & (df["decay"] == str(decay))]["ari"].iloc[0] for m in MODES]
        ax.bar(x + (i - 1) * width, vals, width, label=f"decay={decay}")
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_ylabel("ARI vs. ground truth")
    ax.set_title("Graph mode x decay: agreement with Oliveira et al. subtypes")
    ax.legend()
    savefig(fig, "mode_decay_grid", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()