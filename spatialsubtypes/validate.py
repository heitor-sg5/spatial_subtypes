"""
Part 4 — permutation-based spatial validation.

For each candidate cluster, tests whether its cells are more spatially
aggregated than expected if the same labels were randomly scattered 
across the target population's own spatial positions.

IMPORTANT: the spatial graph used here should represent proximity 
among target cells only (e.g. macrophage-to-macrophage distance),
not the full-tissue graph. Use `restrict_to_target` below to derive 
the second from the first, or build a fresh target-only graph via 
graph.build_spatial_graph on a target-only AnnData.

CAVEAT: if the augmentation graph was built with mode="knn" on the
full tissue, a target cell's k nearest neighbors may all be non-target
cells, so `restrict_to_target` can yield a very sparse or disconnected
target-target graph. For validation specifically, mode="radius" (or a
freshly-built target-only knn graph) is recommended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

def restrict_to_target(W: sp.csr_matrix, target_mask: np.ndarray) -> sp.csr_matrix:
    """
    Subset a full-tissue spatial adjacency matrix down to the
    target-cell x target-cell block.
    """
    idx = np.where(target_mask)[0]
    return W[idx][:, idx].tocsr()

def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (BH) FDR correction, no external dependency."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj

def _cluster_statistic(W: sp.csr_matrix, one_hot: np.ndarray) -> np.ndarray:
    """
    Mean within-cluster edge weight per cluster, for every cluster at
    once. one_hot: (n_cells, k) indicator matrix. Returns (k,) array.

    same_cluster_weight_c = 1_c^T @ W @ 1_c (sum of W[i,j] over all
    ordered pairs i,j both in cluster c; zero for non-edges).
    Normalized by n_c * (n_c - 1), the number of ordered same-cluster
    pairs, so the statistic is a genuine weighted mean, comparable
    across clusters of different sizes.
    """
    n_c = one_hot.sum(axis=0)
    M = one_hot.T @ (W @ one_hot)  # (k, k)
    same_weight = np.diag(M)
    denom = n_c * (n_c - 1)
    denom = np.where(denom == 0, np.nan, denom)  # undefined
    return same_weight / denom

def _morans_i_statistic(W: sp.csr_matrix, one_hot: np.ndarray) -> np.ndarray:
    """
    Per-cluster Moran's I treating each cluster's one-hot indicator as
    the variable of interest. Offered as an alternative to the
    edge-weight statistic.
    """
    n = one_hot.shape[0]
    S0 = W.sum()
    k = one_hot.shape[1]
    out = np.empty(k)
    for c in range(k):
        x = one_hot[:, c].astype(np.float64)
        xbar = x.mean()
        xc = x - xbar
        num = n * (xc @ (W @ xc))
        den = S0 * (xc @ xc)
        out[c] = num / den if den != 0 else np.nan
    return out

_STATISTICS = {"edge_weight": _cluster_statistic, "morans_i": _morans_i_statistic}

def _nearest_sibling(clusters: np.ndarray, labels: np.ndarray, reference_expression: np.ndarray) -> dict:
    """
    For each cluster, find its nearest OTHER cluster by centroid
    distance in reference_expression.
    """
    centroids = {c: reference_expression[labels == c].mean(axis=0) for c in clusters}
    nearest = {}
    for c in clusters:
        others = [o for o in clusters if o != c]
        dists = [np.linalg.norm(centroids[c] - centroids[o]) for o in others]
        nearest[c] = others[int(np.argmin(dists))]
    return nearest

def _pairwise_validate(
    labels: np.ndarray,
    W: sp.csr_matrix,
    reference_expression: np.ndarray,
    statistic: str,
    n_perm: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Local test: for each cluster, pool it with its nearest sibling and
    ask whether the observed label split is more spatially segregated
    than a random re-split of that SAME pooled set of cells.
    """
    clusters = np.unique(labels)
    stat_fn = _STATISTICS[statistic]
    nearest = _nearest_sibling(clusters, labels, reference_expression)
    rng = np.random.default_rng(random_state)

    rows = []
    for c in clusters:
        c2 = nearest[c]
        pool_mask = np.isin(labels, [c, c2])
        pool_idx = np.where(pool_mask)[0]
        W_pool = W[pool_idx][:, pool_idx]
        pool_labels = labels[pool_mask]

        one_hot = np.stack([(pool_labels == c).astype(float), (pool_labels == c2).astype(float)], axis=1)
        observed = stat_fn(W_pool, one_hot)[0]  # statistic for cluster c within this pool

        n_pool = len(pool_labels)
        null = np.empty(n_perm)
        for p in range(n_perm):
            perm_idx = rng.permutation(n_pool)
            null[p] = stat_fn(W_pool, one_hot[perm_idx])[0]

        p_raw = (np.sum(null >= observed) + 1) / (n_perm + 1)
        rows.append({"cluster": c, "nearest_sibling": c2, "n_cells": int((labels == c).sum()),
                      "observed": observed, "p_value": p_raw})

    stats = pd.DataFrame(rows)
    stats["p_adj"] = _bh_correct(stats["p_value"].to_numpy())
    return stats

def spatial_validate(
    labels: np.ndarray,
    W: sp.csr_matrix,
    statistic: str = "edge_weight",
    test_mode: str = "auto",
    reference_expression: np.ndarray | None = None,
    n_perm: int = 1000,
    alpha: float = 0.05,
    action: str = "drop",
    random_state: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Permutation test of spatial localization per candidate subtype,
    with an optional relabeling decision fed back into the output.

    "pairwise" (recommended default): for each cluster, pools it with 
        its nearest sibling (by expression centroid) and tests whether 
        the observed split is more spatially segregated than a random 
        re-split. Rejects arbitrary splits within an already-localized 
        shared region.

    "global": one-vs-rest test against the full target population.
        Doesn't need reference_expression, but only answers
        "does this cluster occupy a non-random region of the tissue
        overall".

    Parameters
    labels : np.ndarray, shape (n_target_cells,)
        Candidate cluster labels for the target population.
    W : scipy.sparse matrix, shape (n_target_cells, n_target_cells)
        Target-target spatial graph.
    statistic : {"edge_weight", "morans_i"}
        edge_weight used as default.
    test_mode : {"auto", "pairwise", "global"}
        "auto" (default): pairwise if reference_expression is given,
        else global.
    reference_expression : np.ndarray, shape (n_target_cells, d), optional
        A purely expression-based embedding Required for test_mode="pairwise"; 
        also used as the merge-target rule when action="merge".
    n_perm : int
        Number of random permutations/re-splits used to build the null distribution.
    alpha : float
        Significance level for determining whether the observed spatial statistic is non-random.
    random_state : int
        Random seed for reproducible permutation results.
    action : {"flag", "merge", "drop"}
        "flag": labels unchanged, stats returned for inspection.
        "drop" (default): non-significant clusters relabeled "unresolved".
        "merge": non-significant cluster merged into its nearest
            sibling.

    Returns
    labels_final : np.ndarray
    stats : pd.DataFrame
        Pairwise mode columns: cluster, nearest_sibling, n_cells,
            observed, p_value, p_adj, significant.
        Global mode columns: cluster, n_cells, observed, p_value,
            p_adj, significant.
    """
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {list(_STATISTICS)}, got {statistic!r}")
    if action not in {"flag", "merge", "drop"}:
        raise ValueError(f"action must be 'flag', 'merge', or 'drop', got {action!r}")
    if test_mode not in {"auto", "pairwise", "global"}:
        raise ValueError(f"test_mode must be 'auto', 'pairwise', or 'global', got {test_mode!r}")

    labels = np.asarray(labels)
    n = len(labels)
    if W.shape != (n, n):
        raise ValueError(f"W shape {W.shape} does not match len(labels)={n}")

    if test_mode == "auto":
        test_mode = "pairwise" if reference_expression is not None else "global"
    if test_mode == "pairwise" and reference_expression is None:
        raise ValueError("test_mode='pairwise' requires reference_expression")
    if action == "merge" and reference_expression is None:
        raise ValueError("action='merge' requires reference_expression")

    clusters = np.unique(labels)
    stat_fn = _STATISTICS[statistic]

    if test_mode == "global":
        k = len(clusters)
        one_hot = np.stack([(labels == c).astype(float) for c in clusters], axis=1)
        observed = stat_fn(W, one_hot)

        rng = np.random.default_rng(random_state)
        null = np.zeros((n_perm, k))
        for p in range(n_perm):
            perm_idx = rng.permutation(n)
            null[p] = stat_fn(W, one_hot[perm_idx])

        p_raw = (np.sum(null >= observed[None, :], axis=0) + 1) / (n_perm + 1)
        p_adj = _bh_correct(p_raw)
        stats = pd.DataFrame(
            {"cluster": clusters, "n_cells": one_hot.sum(axis=0).astype(int),
             "observed": observed, "p_value": p_raw, "p_adj": p_adj,
             "significant": p_adj < alpha}
        )
    else:  # pairwise
        stats = _pairwise_validate(labels, W, reference_expression, statistic, n_perm, random_state)
        stats["significant"] = stats["p_adj"] < alpha

    labels_final = labels.copy().astype(object)
    significant = stats.set_index("cluster")["significant"]
    fail_clusters = clusters[~np.array([significant[c] for c in clusters])]

    if action == "flag":
        pass
    elif action == "drop":
        labels_final[np.isin(labels, fail_clusters)] = "unresolved"
    elif action == "merge":
        if test_mode == "pairwise":
            nearest = stats.set_index("cluster")["nearest_sibling"]
            # Union-find over clusters
            parent = {c: c for c in clusters}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for c in fail_clusters:
                union(c, nearest[c])

            # pick a canonical representative per component: prefer a
            # significant cluster if the component contains one, else
            # the largest cluster in the component.
            components: dict = {}
            for c in clusters:
                components.setdefault(find(c), []).append(c)

            sizes = stats.set_index("cluster")["n_cells"]
            rename = {}
            for root, members in components.items():
                sig_members = [m for m in members if significant[m]]
                rep = sig_members[0] if sig_members else max(members, key=lambda m: sizes[m])
                for m in members:
                    rename[m] = rep

            for c in clusters:
                if rename[c] != c:
                    labels_final[labels == c] = rename[c]
        else:
            centroids = {c: reference_expression[labels == c].mean(axis=0) for c in clusters}
            pass_clusters = clusters[np.array([significant[c] for c in clusters])]
            for c in fail_clusters:
                if len(pass_clusters) == 0:
                    break
                dists = [np.linalg.norm(centroids[c] - centroids[o]) for o in pass_clusters]
                labels_final[labels == c] = pass_clusters[int(np.argmin(dists))]

    return labels_final, stats