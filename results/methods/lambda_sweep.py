"""
Lambda sweep for the augmentation graph. For each lam in LAMBDAS, builds the
graph, augments (this lam), clusters (fixed resolution), and scores labels
against Oliveira's ground truth.
"""

from __future__ import annotations

import gc
import os

import matplotlib.pyplot as plt
import pandas as pd

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    FIGURES_ROOT, SPATIAL_KEY, evaluate_against_ground_truth,
    get_target_mask, load_full_tissue, savefig,
)

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
GRAPH_KWARGS = dict(mode="delaunay", decay=None)
RESOLUTION = 1.0

def main():
    adata_base = load_full_tissue()
    target_mask = get_target_mask(adata_base)
    print(f"Target population: {target_mask.sum()} macrophage bins")

    rows = []
    for lam in LAMBDAS:
        key = f"lam_sweep_{lam}"
        print(f"\n--- lam={lam} ---")
        a = find_spatial_subtypes(
            adata_base.copy(),
            spatial_key=SPATIAL_KEY,
            target_mask=target_mask,
            graph_kwargs=GRAPH_KWARGS,
            augment_kwargs=dict(lam=lam, n_pcs=50),
            cluster_kwargs=dict(resolution=RESOLUTION),
            validate_kwargs=dict(action="drop", n_perm=1000),
            key_added=key,
        )
        raw = a.obs.loc[target_mask, key].to_numpy()
        valid = a.obs.loc[target_mask, f"{key}_valid"].to_numpy()
        adata_target = a[target_mask]

        raw_eval = evaluate_against_ground_truth(raw, adata_target)
        valid_eval = evaluate_against_ground_truth(valid, adata_target, exclude_labels=("unresolved", "NA"))

        n_raw = len(set(raw))
        n_valid = len(set(valid) - {"unresolved"})
        print(f"  raw: n_clusters={n_raw} ARI={raw_eval['ari']:.3f}")
        print(f"  valid: n_clusters={n_valid} ARI={valid_eval['ari']:.3f} coverage={valid_eval['coverage']:.2f}")

        rows.append({
            "lam": lam, "n_raw_clusters": n_raw, "n_valid_clusters": n_valid,
            "raw_ari": raw_eval["ari"], "valid_ari": valid_eval["ari"],
            "valid_coverage": valid_eval["coverage"],
        })

        del a, raw, valid, adata_target
        gc.collect()

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(df["lam"], df["raw_ari"], marker="o", label="raw")
    ax1.plot(df["lam"], df["valid_ari"], marker="o", label="validated")
    ax1.set_xlabel("lam")
    ax1.set_ylabel("ARI vs. ground truth")
    ax1.set_title("Separability vs. lam (provisional: delaunay, decay=None)")
    ax1.legend()

    ax2.plot(df["lam"], df["n_raw_clusters"], marker="o", label="raw clusters")
    ax2.plot(df["lam"], df["n_valid_clusters"], marker="o", label="validated clusters")
    ax2.set_xlabel("lam")
    ax2.set_ylabel("number of clusters")
    ax2.set_title("Cluster count vs. lam")
    ax2.legend()

    fig.tight_layout()
    savefig(fig, "lambda_sweep", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()