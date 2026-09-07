"""
Sweeps lam from 0 (expression-only) to 1 (neighborhood-only) and tracks
ARI against ground truth and validated cluster count.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    CELL_TYPE_KEY, FIGURES_ROOT, SPATIAL_KEY, TARGET_TYPE,
    evaluate_against_ground_truth, load_full_tissue, savefig,
)

LAMBDAS = [0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
GRAPH_KWARGS = dict(mode="radius", radius=50.0, decay="gaussian")
RESOLUTION = 1.0

def main():
    """
    Lam sweep: runs the lam sweep, saves a CSV of results, and plots two line charts
    """
    adata = load_full_tissue()
    target_mask = (adata.obs[CELL_TYPE_KEY] == TARGET_TYPE).to_numpy()

    rows = []
    for lam in LAMBDAS:
        key = f"sweep_lam{lam}"
        print(f"\n--- lam={lam} ---")
        a = find_spatial_subtypes(
            adata.copy(),
            spatial_key=SPATIAL_KEY,
            cell_type_key=CELL_TYPE_KEY, target_type=TARGET_TYPE,
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

    df = pd.DataFrame(rows)
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "lambda_sweep.csv"), index=False)
    print("\n" + df.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(df["lam"], df["raw_ari"], marker="o", label="raw")
    ax1.plot(df["lam"], df["valid_ari"], marker="o", label="validated")
    ax1.set_xlabel("lam")
    ax1.set_ylabel("ARI vs. ground truth")
    ax1.set_title("Separability vs. lam")
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