"""
Stage 2: feature blocks and the lambda-mixed embedding.

Two blocks describe each target cell:
- own block: the cell's own expression (z-scored per gene);
- neighbour block: either the kernel-weighted mean expression of its
  neighbours ("expression") or the cell-type fractions among its neighbours
  ("composition").

Following BANKSY, genes are standardised before neighbours are averaged, and
both blocks are divided by the square root of their gene count. The own block
then has total variance 1, and the neighbour block has total variance below 1
(averaging removes noise but keeps smooth, niche-level variation). Rescaling 
the neighbour block to unit total variance (scaling="unit_variance") lets it 
swamp the own block even at small lam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

FEATURES = ("expression", "composition")
SCALINGS = ("input_std", "unit_variance", "per_gene")

def _col_moments(X: sp.spmatrix, chunk: int = 50_000):
    """Column mean and (population) variance of a sparse matrix, in row chunks."""
    n, g = X.shape
    s = np.zeros(g)
    ss = np.zeros(g)
    for start in range(0, n, chunk):
        block = X[start : start + chunk]
        s += np.asarray(block.sum(axis=0)).ravel()
        ss += np.asarray(block.multiply(block).sum(axis=0)).ravel()
    mean = s / n
    return mean, np.maximum(ss / n - mean**2, 0.0)

def select_hvg(X, n_top: int = 2000, degree: int = 3) -> np.ndarray:
    """
    Indices of highly variable genes of a log-normalised matrix.

    Fits a robust trend of log variance against log mean (cubic polynomial,
    refitted after discarding the 15% most variable genes) and keeps the
    'n_top' genes with the largest residual. Genes with zero mean or
    variance are never selected.
    """
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    mean, var = _col_moments(X)
    ok = (mean > 0) & (var > 0)
    if n_top >= ok.sum():
        return np.flatnonzero(ok)
    lm, lv = np.log(mean[ok]), np.log(var[ok])
    keep = np.ones(len(lm), dtype=bool)
    for _ in range(3):
        coef = np.polyfit(lm[keep], lv[keep], degree)
        resid = lv - np.polyval(coef, lm)
        keep = resid < np.quantile(resid, 0.85)
    score = np.full(len(mean), -np.inf)
    score[ok] = resid
    return np.sort(np.argsort(-score)[:n_top])

@dataclass
class Inputs:
    """Everything the feature blocks need, computed once per dataset."""

    target_idx: np.ndarray  # indices of target cells in the full tissue
    own: np.ndarray  # (n_target, g_own) unit-total-variance own block
    X_nbr: sp.csr_matrix  # (n_cells, g_nbr) tissue-wide HVG expression
    nbr_std: np.ndarray  # per-gene std of X_nbr over the whole tissue
    onehot: sp.csr_matrix | None = None  # (n_cells, n_types) cell-type indicators
    onehot_std: np.ndarray | None = None  # per-type std of the indicators over the whole tissue
    categories: list | None = None
    own_genes: np.ndarray | None = None
    nbr_genes: np.ndarray | None = None
    cache: dict = field(default_factory=dict)

def prepare_inputs(
    adata,
    target_mask,
    *,
    n_hvg: int = 2000,
    layer: str | None = None,
    composition_key: str | None = None,
    own_genes: np.ndarray | None = None,
    nbr_genes: np.ndarray | None = None,
) -> Inputs:
    """
    Select genes and build the own block.

    'adata.X' (or 'layer') must be log-normalised expression for the FULL
    tissue; the target cells are picked with 'target_mask' afterwards.
    'composition_key' is an 'adata.obs' column of cell-type labels; cells
    with a missing label become their own "unassigned" category.
    """
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape[0] != adata.n_obs:
        raise ValueError("target_mask must have one entry per cell")
    target_idx = np.flatnonzero(target_mask)
    if len(target_idx) < 3:
        raise ValueError("fewer than 3 target cells")

    X = adata.layers[layer] if layer is not None else adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    X = X.astype(np.float32)

    Xt = X[target_idx]
    if own_genes is None:
        own_genes = select_hvg(Xt, n_hvg)
    if nbr_genes is None:
        nbr_genes = select_hvg(X, n_hvg)

    own = Xt[:, own_genes].toarray().astype(np.float64)
    sd = own.std(axis=0)
    own = (own - own.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    own /= np.sqrt(own.shape[1])

    X_nbr = X[:, nbr_genes].tocsr()
    _, var = _col_moments(X_nbr)
    nbr_std = np.sqrt(var)

    onehot = categories = onehot_std = None
    if composition_key is not None:
        labels = adata.obs[composition_key].astype(object).where(
            adata.obs[composition_key].notna(), "unassigned"
        )
        cat = pd.Categorical(labels.astype(str))
        categories = list(cat.categories)
        onehot = sp.csr_matrix(
            (np.ones(len(cat), dtype=np.float32), (np.arange(len(cat)), cat.codes)),
            shape=(len(cat), len(categories)),
        )
        frac = np.asarray(onehot.mean(axis=0)).ravel()
        onehot_std = np.sqrt(frac * (1 - frac))

    return Inputs(
        target_idx=target_idx,
        own=own.astype(np.float32),
        X_nbr=X_nbr,
        nbr_std=nbr_std,
        onehot=onehot,
        onehot_std=onehot_std,
        categories=categories,
        own_genes=own_genes,
        nbr_genes=nbr_genes,
    )

def _scaled(inputs: Inputs, name: str, M: sp.csr_matrix, std: np.ndarray) -> sp.csr_matrix:
    """Column-scale M by 1 / std (cached on 'inputs')."""
    if name not in inputs.cache:
        inv = 1.0 / np.where(std > 0, std, 1.0)
        inputs.cache[name] = (M @ sp.diags(inv.astype(np.float32))).tocsr()
    return inputs.cache[name]

def neighbour_matrix(inputs: Inputs, W: sp.csr_matrix, feature: str = "expression",
                     scaling: str = "input_std") -> np.ndarray:
    """
    Weighted neighbour average of the chosen feature, before centring.

    With scaling "input_std" or "unit_variance" each gene (or cell-type
    indicator) is divided by its tissue-wide std before averaging, which is
    equivalent to averaging z-scored values as BANKSY does. With "per_gene"
    the raw values are averaged (composition rows then sum to 1).
    """
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}, got {scaling!r}")
    standardise = scaling != "per_gene"
    if feature == "expression":
        M = _scaled(inputs, "X_nbr_std", inputs.X_nbr, inputs.nbr_std) if standardise else inputs.X_nbr
    elif feature == "composition":
        if inputs.onehot is None:
            raise ValueError("feature='composition' needs prepare_inputs(composition_key=...)")
        M = _scaled(inputs, "onehot_std", inputs.onehot, inputs.onehot_std) if standardise else inputs.onehot
    else:
        raise ValueError(f"feature must be one of {FEATURES}, got {feature!r}")
    out = W @ M
    return out.toarray() if sp.issparse(out) else np.asarray(out)

def neighbour_block(inputs: Inputs, W: sp.csr_matrix, feature: str = "expression",
                    scaling: str = "input_std") -> np.ndarray:
    """
    Neighbour block for the target cells.

    W is the (n_target, n_cells) row-normalised weight matrix from
    'apply_kernel'. Cells with no neighbours get the average neighbourhood
    (zero after centring), not their own expression.

    scaling
        "input_std"      standardise genes, average, centre, divide by sqrt(n features).
                         Total variance is below 1.
        "unit_variance"  as above, then rescale the block to total variance 1.
        "per_gene"       average raw values, z-score every output column, divide by
                         sqrt(n features). Inflates columns whose neighbourhood mean
                         is mostly noise; kept for comparison.
    """
    if W.shape[0] != len(inputs.target_idx):
        raise ValueError("W must have one row per target cell")
    nbr = neighbour_matrix(inputs, W, feature, scaling).astype(np.float64)

    isolated = np.asarray(W.getnnz(axis=1)).ravel() == 0
    if isolated.any():
        fill = nbr[~isolated].mean(axis=0) if (~isolated).any() else 0.0
        nbr[isolated] = fill
    nbr -= nbr.mean(axis=0)

    if scaling == "per_gene":
        sd = nbr.std(axis=0)
        nbr = nbr / np.where(sd > 0, sd, 1.0)
    nbr = nbr / np.sqrt(nbr.shape[1])
    if scaling == "unit_variance":
        total = nbr.var(axis=0).sum()
        nbr = nbr / np.sqrt(total) if total > 0 else nbr
    return nbr.astype(np.float32)

def embed(own: np.ndarray, nbr: np.ndarray | None, lam: float, n_pcs: int = 50,
          seed: int = 0) -> np.ndarray:
    """
    PCA of [sqrt(1 - lam) * own, sqrt(lam) * nbr].
    lam = 0 is plain expression PCA; lam = 1 uses the neighbourhood only.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if lam == 0.0 or nbr is None:
        X = own
    elif lam == 1.0:
        X = nbr
    else:
        X = np.concatenate([np.sqrt(1.0 - lam) * own, np.sqrt(lam) * nbr], axis=1)
    k = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    return PCA(n_components=k, random_state=seed).fit_transform(X).astype(np.float32)
