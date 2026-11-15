"""
Stage 4: validation of candidate subtypes.

A cluster is kept as a subtype only if, against its nearest sibling cluster,
it passes BOTH checks:

1. Spatially coherent: its cells are more tightly connected on a target-only
   spatial graph than a random re-split of the same pooled cells (one-sided
   permutation test, Benjamini-Hochberg corrected across clusters).
2. Distinct in its own expression: a cross-validated classifier on the
   own-expression embedding separates it from the sibling (AUC >= threshold).

Why both: clusters built from neighbour-averaged features are spatially
clumped even when the data are pure noise, so check 1 alone is not enough.
Check 2 on its own is circular at lam = 0 (the clusters were built from that
very expression), where check 1 is the safeguard instead.

Failing clusters are merged into their sibling, smallest first, and the
remaining clusters are re-tested until all pass. No cell is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class ValidationResult:
    labels: np.ndarray          # final labels, "0", "1", ... ordered by size
    stats: pd.DataFrame         # one row per final cluster
    merges: list                # (merged_cluster, into_cluster, reason), original names


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs stay NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    if q.size == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * q.size / (np.arange(q.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(q.size)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _perm_pvalue(W: sp.csr_matrix, x: np.ndarray, n_perm: int, rng, batch: int = 200) -> float:
    """P(random subset of the same size is at least as tightly connected as x)."""
    n, n_c = len(x), int(x.sum())
    observed = float(x @ (W @ x))
    ge, done = 0, 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        R = np.zeros((n, b))
        for j in range(b):
            R[rng.choice(n, n_c, replace=False), j] = 1.0
        stat = np.einsum("ij,ij->j", R, W @ R)
        ge += int((stat >= observed - 1e-9).sum())
        done += b
    return (ge + 1) / (n_perm + 1)


def _pair_auc(emb: np.ndarray, y: np.ndarray, cv: int, max_pool: int, seed: int) -> float:
    """Cross-validated AUC of a logistic classifier separating y=1 from y=0."""
    rng = np.random.default_rng(seed)
    if len(y) > max_pool:
        keep = np.sort(rng.choice(len(y), max_pool, replace=False))
        emb, y = emb[keep], y[keep]
    folds = min(cv, int(y.sum()), int((1 - y).sum()))
    if folds < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, emb, y, cv=skf, scoring="roc_auc").mean())


def _nearest_sibling(clusters, labels, emb):
    centroids = {c: emb[labels == c].mean(axis=0) for c in clusters}
    sib = {}
    for c in clusters:
        others = [o for o in clusters if o != c]
        d = [np.linalg.norm(centroids[c] - centroids[o]) for o in others]
        sib[c] = others[int(np.argmin(d))]
    return sib


def _test_round(labels, W, emb, n_perm, alpha, min_cluster_size, auc_threshold, cv, max_pool, seed):
    clusters = list(np.unique(labels))
    sib = _nearest_sibling(clusters, labels, emb)
    rng = np.random.default_rng(seed)
    rows = []
    for c in clusters:
        n_c = int((labels == c).sum())
        row = {"cluster": c, "n_cells": n_c, "sibling": sib[c], "observed": np.nan,
               "p_value": np.nan, "auc": np.nan, "too_small": n_c < min_cluster_size}
        if not row["too_small"]:
            pool = np.flatnonzero(np.isin(labels, [c, sib[c]]))
            x = (labels[pool] == c).astype(float)
            Wp = W[pool][:, pool]
            row["observed"] = float(x @ (Wp @ x)) / max(n_c * (n_c - 1), 1)
            row["p_value"] = _perm_pvalue(Wp, x, n_perm, rng)
            row["auc"] = _pair_auc(emb[pool], x.astype(int), cv, max_pool, seed)
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats["p_adj"] = bh_adjust(stats["p_value"].to_numpy())
    stats["spatial_ok"] = stats["p_adj"] < alpha
    stats["expression_ok"] = stats["auc"] >= auc_threshold
    stats["passes"] = stats["spatial_ok"] & stats["expression_ok"] & ~stats["too_small"]
    return stats


def _reason(row) -> str:
    if row["too_small"]:
        return "too small"
    parts = []
    if not row["spatial_ok"]:
        parts.append("not spatially coherent")
    if not row["expression_ok"]:
        parts.append("not distinct in own expression")
    return " and ".join(parts)


def validate_subtypes(
    labels,
    W: sp.csr_matrix,
    embedding: np.ndarray,
    *,
    n_perm: int = 1000,
    alpha: float = 0.05,
    min_cluster_size: int = 20,
    auc_threshold: float = 0.8,
    cv: int = 5,
    max_pool: int = 5000,
    seed: int = 0,
) -> ValidationResult:
    """Merge candidate clusters that fail the spatial or expression check.

    Parameters
    ----------
    labels : cluster label per target cell.
    W : (n_target, n_target) symmetric, non-negative target-to-target spatial
        graph (``adjacency(build_graph(coords_target, "knn", symmetric=True))``).
    embedding : own-expression embedding of the target cells (``embed(own, None, 0)``).
        It is used for the sibling search and for check 2, so it must not
        contain neighbourhood information.
    """
    labels = np.asarray(labels).astype(str).copy()
    n = len(labels)
    if W.shape != (n, n):
        raise ValueError(f"W shape {W.shape} does not match {n} labels")
    if embedding.shape[0] != n:
        raise ValueError("embedding must have one row per label")

    merges = []
    while True:
        clusters = np.unique(labels)
        if len(clusters) < 2:
            stats = pd.DataFrame([{"cluster": clusters[0], "n_cells": n, "sibling": None,
                                   "observed": np.nan, "p_value": np.nan, "auc": np.nan,
                                   "p_adj": np.nan, "spatial_ok": False, "expression_ok": False,
                                   "too_small": False, "passes": False}])
            break
        stats = _test_round(labels, W, embedding, n_perm, alpha, min_cluster_size,
                            auc_threshold, cv, max_pool, seed)
        failing = stats[~stats["passes"]]
        if failing.empty:
            break
        worst = failing.sort_values(["n_cells", "cluster"]).iloc[0]
        merges.append((worst["cluster"], worst["sibling"], _reason(worst)))
        labels[labels == worst["cluster"]] = worst["sibling"]

    order = (pd.Series(labels).value_counts().index.tolist())
    rename = {old: str(i) for i, old in enumerate(order)}
    stats = stats.copy()
    stats["cluster"] = stats["cluster"].map(rename)
    stats["sibling"] = stats["sibling"].map(rename)
    stats = stats.sort_values("cluster", key=lambda s: s.astype(int)).reset_index(drop=True)
    return ValidationResult(np.array([rename[v] for v in labels]), stats, merges)
