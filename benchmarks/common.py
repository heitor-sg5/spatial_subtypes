"""
Shared helpers for the benchmark scripts: data loading (real or simulated),
scoring against the reference labels, a cached experiment runner, and plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.spatialsubtypes import (  # noqa: E402
    adjacency, apply_kernel, build_graph, cluster_to_counts, embed, neighbour_block, prepare_inputs,
)

from benchmarks import config as cfg  # noqa: E402

GROUPS = ("SPP1+", "SELENOP+", "other")


# --------------------------------------------------------------------------- data

def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", type=Path, default=cfg.DATA_PATH, help="path to the annotated .h5ad file")
    ap.add_argument("--demo", action="store_true",
                    help="use simulated data instead of the real file (figures go to figures/demo/)")
    return ap.parse_args()


def load(args) -> tuple[AnnData, np.ndarray, Path]:
    """Return (full-tissue AnnData, target mask, output directory)."""
    import anndata as ad

    if args.demo:
        adata = make_demo_data()
        out = cfg.DEMO_FIGURES_DIR
    else:
        if not args.data.exists():
            raise SystemExit(f"Data file not found: {args.data}\nPlace it there, pass --data, or try --demo.")
        adata = ad.read_h5ad(args.data)
        out = cfg.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    mask = target_mask(adata)
    print(f"{adata.n_obs:,} bins, {int(mask.sum()):,} target bins "
          f"({cfg.TARGET_TYPE}, {cfg.PERIPHERY_VALUE})")
    return adata, mask, out


def target_mask(adata) -> np.ndarray:
    return ((adata.obs[cfg.CELL_TYPE_KEY] == cfg.TARGET_TYPE)
            & (adata.obs[cfg.PERIPHERY_KEY] == cfg.PERIPHERY_VALUE)).to_numpy()


def reference_groups(adata, idx) -> np.ndarray:
    """Three-group reference for the target bins: SPP1+, SELENOP+, other."""
    ref = adata.obs[cfg.REFERENCE_KEY].iloc[idx].astype(object).to_numpy()
    return np.where(ref == cfg.REFERENCE_SPP1, "SPP1+", np.where(ref == cfg.REFERENCE_SELENOP, "SELENOP+", "other"))


def load_best() -> dict:
    path = cfg.FIGURES_DIR / "best_config.json"
    return json.loads(path.read_text()) if path.exists() else dict(cfg.DEFAULT_BEST)


def make_demo_data(seed: int = 0, grid: int = 140, n_genes: int = 200) -> AnnData:
    """Simulated Visium-HD-like tissue (NOT real data), for testing the scripts.

    A grid of 8 um bins is split into two niches by a wavy boundary. Macrophages
    near the boundary fall into five states: SPP1-like (mostly in niche A),
    SELENOP-like (mostly in niche B) and three unnamed states. Only the first two
    receive reference labels, as in Oliveira et al. Cell-type labels differ weakly
    between niches (CAFs vs epithelial cells); 10% of bins are unlabelled.
    """
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(grid), np.arange(grid))
    xy = np.stack([gx.ravel(), gy.ravel()], 1) * 8.0
    n = len(xy)
    boundary = grid * 4.0 + 120 * np.sin(xy[:, 1] / 150)
    niche = (xy[:, 0] > boundary).astype(int)
    periphery = np.abs(xy[:, 0] - boundary) < 160

    # ---- cell types ----
    p_mac = np.where(periphery, 0.14, 0.03)
    u = rng.random(n)
    ctype = np.empty(n, dtype=object)
    is_mac = u < p_mac
    other = np.array(["Tumor", "CAF", "T cell", "Epithelial", "Endothelial"], dtype=object)
    probs = {0: [0.45, 0.25, 0.10, 0.10, 0.10], 1: [0.45, 0.10, 0.10, 0.25, 0.10]}
    for k in (0, 1):
        m = (niche == k) & ~is_mac
        ctype[m] = rng.choice(other, m.sum(), p=probs[k])
    ctype[is_mac] = "Macrophage"
    unlabelled = rng.random(n) < 0.10
    label = ctype.copy(); label[unlabelled] = None

    # ---- macrophage states ----
    state = np.full(n, "", dtype=object)
    mac = np.flatnonzero(is_mac)
    p_state = {0: [0.20, 0.02, 0.26, 0.26, 0.26], 1: [0.02, 0.20, 0.26, 0.26, 0.26]}
    names = np.array(["SPP1", "SELENOP", "o1", "o2", "o3"], dtype=object)
    for k in (0, 1):
        sel = mac[niche[mac] == k]
        state[sel] = rng.choice(names, len(sel), p=p_state[k])

    # ---- expression ----
    base = rng.normal(0.0, 0.8, n_genes)
    eff = np.zeros((n, n_genes))
    type_genes = {t: np.arange(i * 12, i * 12 + 12) for i, t in enumerate(
        ["Tumor", "CAF", "T cell", "Epithelial", "Endothelial", "Macrophage"])}
    for t, gs in type_genes.items():
        eff[np.ix_(ctype == t, gs)] += 1.5
    niche_a, niche_b = np.arange(100, 120), np.arange(120, 140)
    bleed = np.where(is_mac, 0.3, 1.0)[:, None]          # macrophages only partly show niche programmes
    eff[np.ix_(niche == 0, niche_a)] += 0.8 * bleed[niche == 0]
    eff[np.ix_(niche == 1, niche_b)] += 0.8 * bleed[niche == 1]
    markers = {"SPP1": np.arange(140, 150), "SELENOP": np.arange(150, 160),
               "o1": np.arange(160, 170), "o2": np.arange(170, 180), "o3": np.arange(180, 190)}
    for s, gs in markers.items():
        eff[np.ix_(state == s, gs)] += 1.6
    depth = np.exp(rng.normal(0, 0.3, n))[:, None]
    counts = rng.poisson(np.exp(base + eff) * depth).astype(np.float32)
    totals = counts.sum(1, keepdims=True); totals[totals == 0] = 1
    X = np.log1p(counts / totals * 1e4).astype(np.float32)

    var_names = np.array([f"G{i:03d}" for i in range(n_genes)], dtype=object)
    var_names[140], var_names[150] = "SPP1", "SELENOP"
    adata = AnnData(X=sp.csr_matrix(X), obsm={"spatial_um": xy})
    adata.var_names = var_names
    adata.obs_names = [f"bin{i}" for i in range(n)]
    adata.obs[cfg.CELL_TYPE_KEY] = pd.Series(label, index=adata.obs_names).astype("category")
    adata.obs[cfg.PERIPHERY_KEY] = np.where(periphery, "50 micron", "Tissue")
    ref = np.full(n, np.nan, dtype=object)
    ref[(state == "SPP1") & periphery] = cfg.REFERENCE_SPP1
    ref[(state == "SELENOP") & periphery] = cfg.REFERENCE_SELENOP
    adata.obs[cfg.REFERENCE_KEY] = pd.Series(ref, index=adata.obs_names).astype("category")
    adata.obs["demo_niche"] = niche
    return adata


# ------------------------------------------------------------------------ metrics

def recovery_f1(labels, groups, named=("SPP1+", "SELENOP+")) -> float:
    """Mean over the named subtypes of the best F1 that any single cluster reaches.

    F1 is computed over ALL target bins, so a cluster that swallows the subtype
    together with many unnamed bins is penalised.
    """
    labels = np.asarray(labels)
    scores = []
    for g in named:
        pos = groups == g
        if pos.sum() == 0:
            continue
        best = 0.0
        for c in np.unique(labels):
            m = labels == c
            tp = int((m & pos).sum())
            if tp:
                p, r = tp / m.sum(), tp / pos.sum()
                best = max(best, 2 * p * r / (p + r))
        scores.append(best)
    return float(np.mean(scores))


def pair_ari(labels, groups) -> float:
    """ARI restricted to the bins that carry a reference name."""
    named = groups != "other"
    return float(adjusted_rand_score(groups[named], np.asarray(labels)[named]))


def embedding_accuracy(emb, groups, k: int = 15, seed: int = 0) -> float:
    """Cross-validated balanced accuracy of a kNN classifier of the three reference groups."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(KNeighborsClassifier(k), emb, groups, cv=cv)
    return float(balanced_accuracy_score(groups, pred))


def spatial_coherence(labels, A: sp.spmatrix) -> float:
    """Share of target-only graph edges whose two ends carry the same label."""
    labels = np.asarray(labels)
    coo = sp.triu(A, k=1).tocoo()
    return float((labels[coo.row] == labels[coo.col]).mean())


# ------------------------------------------------------------------------ runner

class Bench:
    """Runs one configuration at a time and caches the expensive intermediates."""

    def __init__(self, adata, mask, composition: bool = True):
        self.adata = adata
        self.coords = np.asarray(adata.obsm[cfg.SPATIAL_KEY], dtype=float)
        print("Preparing genes and own block ...")
        self.inputs = prepare_inputs(adata, mask, n_hvg=cfg.N_HVG,
                                     composition_key=cfg.CELL_TYPE_KEY if composition else None)
        self.idx = self.inputs.target_idx
        self.groups = reference_groups(adata, self.idx)
        print("Reference groups:", {g: int((self.groups == g).sum()) for g in GROUPS})
        self.A_val = adjacency(build_graph(self.coords[self.idx], "knn",
                                           n_neighbors=cfg.VALIDATION_NEIGHBORS, symmetric=True))
        self._graph, self._weights, self._block, self._emb = {}, {}, {}, {}

    def graph(self, spec: dict):
        key = tuple(sorted(spec.items()))
        if key not in self._graph:
            self._graph[key] = build_graph(self.coords, query=self.idx, **spec)
        return self._graph[key]

    def weights(self, spec: dict, kernel: str):
        key = (tuple(sorted(spec.items())), kernel)
        if key not in self._weights:
            self._weights[key] = apply_kernel(self.graph(spec), kernel)
        return self._weights[key]

    def block(self, spec: dict, kernel: str, feature: str):
        key = (tuple(sorted(spec.items())), kernel, feature)
        if key not in self._block:
            self._block[key] = neighbour_block(self.inputs, self.weights(spec, kernel), feature)
        return self._block[key]

    def embedding(self, spec, kernel, feature, lam):
        if lam == 0:
            key = ("own", 0.0)
            if key not in self._emb:
                self._emb[key] = embed(self.inputs.own, None, 0.0, cfg.N_PCS)
            return self._emb[key]
        key = (tuple(sorted(spec.items())), kernel, feature, lam)
        if key not in self._emb:
            self._emb[key] = embed(self.inputs.own, self.block(spec, kernel, feature), lam, cfg.N_PCS)
        return self._emb[key]

    def evaluate(self, spec: dict, kernel: str, lam: float, feature: str = "expression", **tags) -> dict:
        """Embed, cluster at every target count and seed, and score."""
        emb = self.embedding(spec, kernel, feature, lam)
        runs = cluster_to_counts(emb, cfg.CLUSTER_COUNTS, cfg.SEEDS, cfg.LEIDEN_NEIGHBORS)
        per_seed = {}
        for (count, seed), lab in runs.items():
            per_seed.setdefault(seed, []).append(
                (recovery_f1(lab, self.groups), pair_ari(lab, self.groups), spatial_coherence(lab, self.A_val)))
        arr = np.array([np.mean(v, axis=0) for v in per_seed.values()])   # (seeds, 3)
        degree = float(self.graph(spec).degree.mean()) if lam > 0 else float("nan")
        return {**tags, "kernel": kernel, "lam": lam, "feature": feature,
                "mean_degree": degree,
                "recovery_f1": arr[:, 0].mean(), "recovery_f1_sd": arr[:, 0].std(ddof=1) if len(arr) > 1 else 0.0,
                "pair_ari": arr[:, 1].mean(), "coherence": arr[:, 2].mean(),
                "coherence_sd": arr[:, 2].std(ddof=1) if len(arr) > 1 else 0.0,
                "embedding_acc": embedding_accuracy(emb, self.groups)}


# -------------------------------------------------------------------------- plots

def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    return _plt


def savefig(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    return path


def scatter_labels(ax, coords, labels, title, grey=("NA", "other", "unlabelled")):
    labels = pd.Series(np.asarray(labels, dtype=object)).fillna("unlabelled")
    muted = labels.isin(grey).to_numpy()
    ax.scatter(coords[muted, 0], coords[muted, 1], s=3, c="#d4d4d4", linewidths=0, label="other / unlabelled")
    cmap = plt().get_cmap("tab10")
    for i, cat in enumerate(sorted(labels[~muted].unique())):
        m = (labels == cat).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=4, color=cmap(i % 10), linewidths=0, label=str(cat))
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.legend(markerscale=3, fontsize=7, loc="best", frameon=False)
