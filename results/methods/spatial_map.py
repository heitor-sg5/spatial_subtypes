"""
Three panels of macrophage bins plotted at their actual tissue 
coordinates (spatial_um), colored by: (1) raw Leiden labels, (2) 
validated labels ("unresolved" shown in grey), and (3) Oliveira's
ground-truth MacrophageSubtype (grey where unlabeled).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from spatialsubtypes import find_spatial_subtypes
from results.shared import (
    CELL_TYPE_KEY, FIGURES_ROOT, GROUND_TRUTH_KEY, SPATIAL_KEY, TARGET_TYPE,
    load_full_tissue, savefig,
)

GRAPH_KWARGS = dict(mode="radius", radius=50.0, decay="gaussian")
LAM = 0.3
RESOLUTION = 1.0

def _scatter_by_label(ax, coords, labels, title, unresolved_values=("unresolved", "NA")):
    """
    Scatter plot of spatial coordinates colored by cluster label, with unresolved clusters in grey.
    """
    labels = pd.Series(labels)
    is_unresolved = labels.isin(unresolved_values) | labels.isna()

    ax.scatter(
        coords[is_unresolved, 0], coords[is_unresolved, 1],
        c="lightgrey", s=4, label="unresolved / unlabeled",
    )
    categories = sorted(labels[~is_unresolved].unique())
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(categories):
        mask = (labels == cat).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, color=cmap(i % 10), label=str(cat))

    ax.set_title(title)
    ax.set_xlabel("spatial x (um)")
    ax.set_ylabel("spatial y (um)")
    ax.set_aspect("equal")
    ax.legend(markerscale=3, fontsize=7, loc="best")

def main():
    """
    Plots the spatial map of macrophage bins colored by raw Leiden labels, validated labels, and ground truth.
    """
    adata = load_full_tissue()
    target_mask = (adata.obs[CELL_TYPE_KEY] == TARGET_TYPE).to_numpy()

    adata = find_spatial_subtypes(
        adata,
        spatial_key=SPATIAL_KEY,
        cell_type_key=CELL_TYPE_KEY, target_type=TARGET_TYPE,
        graph_kwargs=GRAPH_KWARGS,
        augment_kwargs=dict(lam=LAM, n_pcs=50),
        cluster_kwargs=dict(resolution=RESOLUTION),
        validate_kwargs=dict(action="drop", n_perm=1000),
        key_added="map",
    )

    coords = adata.obsm[SPATIAL_KEY][target_mask]
    raw = adata.obs.loc[target_mask, "map"].to_numpy()
    valid = adata.obs.loc[target_mask, "map_valid"].to_numpy()
    ground_truth = adata.obs.loc[target_mask, GROUND_TRUTH_KEY].to_numpy()

    print(f"Plotting {target_mask.sum()} macrophage bins")
    print(f"Raw: {len(set(raw))} clusters | Validated: {len(set(valid) - {'unresolved'})} clusters")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _scatter_by_label(axes[0], coords, raw, "Raw Leiden labels", unresolved_values=("NA",))
    _scatter_by_label(axes[1], coords, valid, "Validated labels")
    _scatter_by_label(axes[2], coords, ground_truth, "Oliveira et al. ground truth", unresolved_values=())

    fig.tight_layout()
    out_dir = os.path.join(FIGURES_ROOT, "methods")
    os.makedirs(out_dir, exist_ok=True)
    savefig(fig, "spatial_map", "methods")
    plt.close(fig)

if __name__ == "__main__":
    main()