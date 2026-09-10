"""
Run squidpy's neighborhood enrichment analysis on the full 
tissue dataset and evaluate against other methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import squidpy as sq

from results.comparison.run_ours import run_ours
from results.shared import SPATIAL_KEY

def run_squidpy_posthoc(adata_target=None, raw_labels=None):
    """
    Returns (raw_labels, zscore_df), plus the enrichment z-score matrix
    """
    if adata_target is None or raw_labels is None:
        _, adata_target, _, raw_labels = run_ours()

    adata_target = adata_target.copy()
    adata_target.obs["cluster"] = pd.Categorical(raw_labels)

    sq.gr.spatial_neighbors(adata_target, spatial_key=SPATIAL_KEY, coord_type="generic")
    sq.gr.nhood_enrichment(adata_target, cluster_key="cluster", seed=0)

    zscore = adata_target.uns["cluster_nhood_enrichment"]["zscore"]
    categories = adata_target.obs["cluster"].cat.categories.tolist()
    zscore_df = pd.DataFrame(zscore, index=categories, columns=categories)

    return raw_labels, zscore_df

def main():
    raw_labels, zscore_df = run_squidpy_posthoc()
    n_clusters = len(set(raw_labels))
    print(f"Raw clustering: {n_clusters} clusters (unchanged by squidpy -- descriptive only)")

    diag = np.diag(zscore_df.values)
    print("\nSelf-enrichment z-score per cluster (higher = more spatially clumped with itself):")
    for cat, z in zip(zscore_df.index, diag):
        print(f"  cluster {cat}: z={z:.2f}")

if __name__ == "__main__":
    main()