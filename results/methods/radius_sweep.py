"""
Radius sweep: for each radius in RADII, builds the graph, augments (fixed lam),
clusters (fixed resolution), and scores labels against Oliveira's ground truth.
"""

from __future__ import annotations

import gc
import os

import matplotlib.pyplot as plt
import pandas as pd

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    FIGURES_ROOT, SPATIAL_KEY,
    evaluate_against_ground_truth, get_target_mask, load_full_tissue, savefig,
)

RADII = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]
DECAY = "gaussian"
LAM = 0.3
RESOLUTION = 1.0

def main():
    adata_base = load_full_tissue()
    target_mask = get_target_mask(adata_base)
    print(f"Target population: {target_mask.sum()} macrophage bins")

    rows = []
    for radius in RADII:
        key = f"radius_sweep_{radius}"
        print(f"\n--- radius={radius} ---")
        a = find_spatial_subtypes(
            adata_base.copy(),
            spatial_key=SPATIAL_KEY,
            target_mask=target_mask,
            graph_kwargs=dict(mode="radius", radius=radius, decay=DECAY),
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
            "radius": radius, "n_clusters": n_clusters,
            "ari": result["ari"], "coverage": result["coverage"],
        })

        del a, labels_target, adata_target, result
        gc.collect()

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(df["radius"], df["ari"], marker="o")
    ax1.set_xlabel("radius (um)")
    ax1.set_ylabel("ARI vs. ground truth")
    ax1.set_title("Radius magnitude vs. separability")

    ax2.plot(df["radius"], df["n_clusters"], marker="o", color="tab:orange")
    ax2.set_xlabel("radius (um)")
    ax2.set_ylabel("number of raw clusters")
    ax2.set_title("Radius magnitude vs. fragmentation")

    fig.tight_layout()
    savefig(fig, "radius_sweep", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()