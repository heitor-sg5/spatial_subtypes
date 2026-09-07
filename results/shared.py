"""
Shared utilities for results/methods/.py and results/comparison/.py.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

# ---- Paths ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_TISSUE_PATH = os.path.join(REPO_ROOT, "data", "VisiumHD_P2CRC_8um_Annotated.h5ad")
MACROPHAGE_SUBSET_PATH = os.path.join(REPO_ROOT, "data", "P2CRC_macrophages_8um.h5ad")
FIGURES_ROOT = os.path.join(REPO_ROOT, "results", "figures")

# ---- Fixed data conventions ----
SPATIAL_KEY = "spatial_um"
CELL_TYPE_KEY = "DeconvolutionLabel2"
TARGET_TYPE = "Macrophage"
GROUND_TRUTH_KEY = "MacrophageSubtype"

def load_full_tissue():
    """
    Full-tissue AnnData.
    """
    adata = sc.read_h5ad(FULL_TISSUE_PATH)
    if SPATIAL_KEY not in adata.obsm:
        raise KeyError(
            f"Expected obsm['{SPATIAL_KEY}'] not found. Available: {list(adata.obsm.keys())}"
        )
    if "highly_variable" not in adata.var:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    return adata

def evaluate_against_ground_truth(labels: np.ndarray, adata_target, exclude_labels=("unresolved", "NA")):
    """
    Compares a candidate label array against Oliveira et al.'s ground-truth
    MacrophageSubtype column, restricted to the subset of cells that
    actually have a ground-truth label (most macrophages won't, since
    Oliveira's annotation is periphery-restricted).

    Parameters
    labels : np.ndarray, shape (n_target_cells,)
        Candidate labels for the target (macrophage) population, in the
        same cell order as adata_target.
    adata_target : AnnData
        The target-cell-only AnnData used to produce `labels` must contain 
        the ground-truth labels.
    exclude_labels : tuple of str
        Label values to treat as "no prediction" and exclude from
        scoring.

    Returns
    dict with:
        n_evaluated       : cells with both a ground-truth label and a
                             non-excluded candidate label
        n_ground_truth    : cells with a ground-truth label at all
        coverage          : n_evaluated / n_ground_truth (how much of
                             the ground-truth set this method actually
                             produced a usable prediction for.
        ari               : adjusted Rand index on the evaluated subset
        contingency       : pd.DataFrame crosstab of predicted vs.
                             ground-truth labels, for a qualitative
                             look
    """
    if GROUND_TRUTH_KEY not in adata_target.obs:
        raise KeyError(f"{GROUND_TRUTH_KEY} not found in adata_target.obs")
    if len(labels) != adata_target.n_obs:
        raise ValueError(
            f"labels length ({len(labels)}) does not match adata_target.n_obs ({adata_target.n_obs})"
        )

    gt = adata_target.obs[GROUND_TRUTH_KEY].to_numpy()
    has_gt = pd.notna(gt)
    n_ground_truth = int(has_gt.sum())

    labels = np.asarray(labels, dtype=object)
    has_prediction = ~np.isin(labels, exclude_labels)

    eval_mask = has_gt & has_prediction
    n_evaluated = int(eval_mask.sum())

    if n_evaluated == 0:
        return {
            "n_evaluated": 0, "n_ground_truth": n_ground_truth, "coverage": 0.0,
            "ari": float("nan"), "contingency": pd.DataFrame(),
        }

    ari = adjusted_rand_score(gt[eval_mask], labels[eval_mask])
    contingency = pd.crosstab(
        pd.Series(labels[eval_mask], name="predicted"),
        pd.Series(gt[eval_mask], name="ground_truth"),
    )

    return {
        "n_evaluated": n_evaluated,
        "n_ground_truth": n_ground_truth,
        "coverage": n_evaluated / n_ground_truth if n_ground_truth > 0 else 0.0,
        "ari": ari,
        "contingency": contingency,
    }

def savefig(fig, name: str, subdir: str):
    """
    Save a figure into results/figures/<subdir>/
    """
    out_dir = os.path.join(FIGURES_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {path}")
    return path

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)