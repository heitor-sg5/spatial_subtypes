"""
Benchmark 1: which graph and which kernel?

Compares 3 graph types x 4 weightings (uniform baseline + Gaussian, exponential,
reciprocal) at a matched neighbourhood size (about 48 bins), a fixed lambda, and
the mean-expression neighbour feature. Reference rows: expression only (lambda = 0)
and a BANKSY-style configuration (kNN 15, Gaussian, lambda = 0.2).

Writes figures/grid_results.csv, figures/best_config.json and
figures/01_graph_kernel_grid.png.

    python benchmarks/01_graph_kernel_grid.py            # real data
    python benchmarks/01_graph_kernel_grid.py --demo     # simulated data
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from benchmarks import common, config as cfg

GRAPHS = {
    "radius": dict(mode="radius", radius=cfg.RADIUS_UM),
    "knn": dict(mode="knn", n_neighbors=cfg.KNN_K),
    "delaunay": dict(mode="delaunay", rings=cfg.DELAUNAY_RINGS),
}


def main():
    args = common.parse_args(__doc__)
    adata, mask, out = common.load(args)
    bench = common.Bench(adata, mask, composition=False)

    rows = []
    for gname, spec in GRAPHS.items():
        for kernel in cfg.GRID_KERNELS:
            r = bench.evaluate(spec, kernel, cfg.GRID_LAM, graph=gname)
            rows.append(r)
            print(f"{gname:9s} {kernel:11s} deg={r['mean_degree']:5.1f}  recovery={r['recovery_f1']:.3f}"
                  f" +/- {r['recovery_f1_sd']:.3f}  emb_acc={r['embedding_acc']:.3f}  coherence={r['coherence']:.3f}")
    grid = pd.DataFrame(rows)

    refs = [bench.evaluate({}, "uniform", 0.0, graph="none", label="expression only (lam=0)")]
    b = cfg.BANKSY_STYLE
    refs.append(bench.evaluate(dict(mode=b["mode"], n_neighbors=b["n_neighbors"]), b["kernel"], b["lam"],
                               graph="knn", label="BANKSY-style (kNN 15, Gaussian, lam 0.2)"))
    ref = pd.DataFrame(refs)
    print("\nReference rows:\n", ref[["label", "recovery_f1", "recovery_f1_sd", "embedding_acc", "coherence"]].round(3).to_string(index=False))

    pd.concat([grid.assign(label=grid["graph"] + " / " + grid["kernel"]), ref]).to_csv(out / "grid_results.csv", index=False)

    # ---- best weighted configuration (uniform is a baseline, not a candidate) ----
    cand = grid[grid["kernel"] != "uniform"].sort_values("recovery_f1", ascending=False)
    best = cand.iloc[0]
    margin = best["recovery_f1"] - cand.iloc[1]["recovery_f1"]
    spec = dict(GRAPHS[best["graph"]])
    (out / "best_config.json").write_text(json.dumps(
        {**spec, "kernel": best["kernel"], "lam": cfg.GRID_LAM, "feature": "expression",
         "note": f"best of the grid at lam={cfg.GRID_LAM}; margin over runner-up {margin:.3f}; "
                 f"seed sd {best['recovery_f1_sd']:.3f}"}, indent=2))
    print(f"\nBest: {best['graph']} + {best['kernel']}  (margin over runner-up {margin:.3f}, seed sd {best['recovery_f1_sd']:.3f})")
    if margin < best["recovery_f1_sd"]:
        print("The margin is smaller than the seed spread: graph and kernel choice are not clearly distinguishable here.")

    # ---- figure ----
    plt = common.plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    gnames, knames = list(GRAPHS), list(cfg.GRID_KERNELS)
    for ax, col, title in ((axes[0], "recovery_f1", "Recovery F1 (primary)"),
                           (axes[1], "embedding_acc", "Embedding accuracy")):
        M = np.array([[grid[(grid["graph"] == g) & (grid["kernel"] == k)][col].iloc[0] for k in knames] for g in gnames])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white" if M[i, j] < M.mean() else "black", fontsize=9)
        ax.set_xticks(range(len(knames)), [k if k != "uniform" else "uniform\n(baseline)" for k in knames])
        ax.set_yticks(range(len(gnames)), gnames)
        ax.axvline(0.5, color="white", lw=3)
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    labels = ["expression only\n(lam = 0)", "BANKSY-style", f"best grid cell\n({best['graph']}, {best['kernel']})"]
    vals = [ref.iloc[0]["recovery_f1"], ref.iloc[1]["recovery_f1"], best["recovery_f1"]]
    errs = [ref.iloc[0]["recovery_f1_sd"], ref.iloc[1]["recovery_f1_sd"], best["recovery_f1_sd"]]
    axes[2].bar(labels, vals, yerr=errs, color=["#9a9a9a", "#e0a030", "#2a7fd4"], capsize=4)
    axes[2].set_ylabel("Recovery F1 (mean +/- seed sd)")
    axes[2].set_title("Reference points")
    fig.suptitle(f"Graph x kernel at lambda = {cfg.GRID_LAM}"
                 + ("   [SIMULATED DEMO DATA]" if args.demo else ""), y=1.03)
    fig.tight_layout()
    common.savefig(fig, out, "01_graph_kernel_grid")


if __name__ == "__main__":
    main()
