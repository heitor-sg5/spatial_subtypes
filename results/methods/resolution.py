"""
Sweeps Leiden resolution and tracks raw vs. validated cluster count.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    CELL_TYPE_KEY, FIGURES_ROOT, SPATIAL_KEY, TARGET_TYPE, load_full_tissue, savefig,
)

RESOLUTIONS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
GRAPH_KWARGS = dict(mode="radius", radius=50.0, decay="gaussian")
LAM = 0.3

def main():
    """
    Sweeps Leiden resolution, saves a CSV of results, and plots a line chart of raw vs. validated cluster count.
    """
    adata = load_full_tissue()
    target_mask = (adata.obs[CELL_TYPE_KEY] == TARGET_TYPE).to_numpy()

    rows = []
    for res in RESOLUTIONS:
        key = f"res_{res}"
        print(f"\n--- resolution={res} ---")
        a = find_spatial_subtypes(
            adata.copy(),
            spatial_key=SPATIAL_KEY,
            cell_type_key=CELL_TYPE_KEY, target_type=TARGET_TYPE,
            graph_kwargs=GRAPH_KWARGS,
            augment_kwargs=dict(lam=LAM, n_pcs=50),
            cluster_kwargs=dict(resolution=res),
            validate_kwargs=dict(action="drop", n_perm=1000),
            key_added=key,
        )
        raw = a.obs.loc[target_mask, key]
        valid = a.obs.loc[target_mask, f"{key}_valid"]

        n_raw = raw.nunique()
        n_valid = valid[valid != "unresolved"].nunique()
        n_unresolved = int((valid == "unresolved").sum())
        print(f"  raw clusters={n_raw}  validated clusters={n_valid}  unresolved cells={n_unresolved}")

        rows.append({
            "resolution": res, "n_raw_clusters": n_raw,
            "n_valid_clusters": n_valid, "n_unresolved_cells": n_unresolved,
        })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "resolution_robustness.csv"), index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["resolution"], df["n_raw_clusters"], marker="o", label="raw clusters")
    ax.plot(df["resolution"], df["n_valid_clusters"], marker="o", label="validated clusters")
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("number of clusters")
    ax.set_title("Cluster count vs. resolution: raw vs. validated")
    ax.legend()
    savefig(fig, "resolution_robustness", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()