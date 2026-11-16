"""
Benchmark 3: the final subtypes.

Runs find_spatial_subtypes with the best configuration (figures/best_config.json,
written by scripts 01 and 02) from a deliberately over-clustered start, then shows:
  - the reference labels and the validated subtypes on the tissue,
  - the cell-type composition around each subtype,
  - marker genes per subtype (SPP1 and SELENOP included when present).
Writes figures/03_final_subtypes.png, figures/final_contingency.csv and
figures/final_report.txt.

    python benchmarks/03_final_subtypes.py [--demo]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmarks import common, config as cfg
from src.spatialsubtypes import find_spatial_subtypes, knn_graph, resolution_for_n_clusters
from src.spatialsubtypes.features import neighbour_matrix


def main():
    args = common.parse_args(__doc__)
    adata, mask, out = common.load(args)
    best = common.load_best() if not args.demo else json.loads((out / "best_config.json").read_text())
    spec = {k: best[k] for k in ("mode", "radius", "n_neighbors", "rings") if k in best}
    feature = best.get("feature", "expression")
    print(f"Configuration: {spec}, kernel={best['kernel']}, lam={best['lam']}, feature={feature}")

    bench = common.Bench(adata, mask, composition=True)
    res = find_spatial_subtypes(
        adata, target_mask=mask, spatial_key=cfg.SPATIAL_KEY, kernel=best["kernel"], lam=best["lam"],
        neighbour_feature=feature, composition_key=cfg.CELL_TYPE_KEY, n_hvg=cfg.N_HVG, n_pcs=cfg.N_PCS,
        n_clusters=cfg.FINAL_START_CLUSTERS, cluster_neighbors=cfg.LEIDEN_NEIGHBORS,
        validation_neighbors=cfg.VALIDATION_NEIGHBORS, validation_kwargs=dict(cfg.FINAL_VALIDATION),
        inputs=bench.inputs, key_added="spatial_subtype", **spec)

    idx, groups = res.target_idx, bench.groups
    print(f"\n{len(set(res.labels_raw))} raw clusters -> {len(set(res.labels))} validated subtypes")
    print("\nValidation table:\n", res.stats.round(4).to_string(index=False))
    print("\nMerges (raw cluster -> sibling, reason):")
    for m in res.merges:
        print("  ", m)

    lines = [f"configuration: {spec} kernel={best['kernel']} lam={best['lam']} feature={feature}",
             f"raw clusters: {len(set(res.labels_raw))}; validated subtypes: {len(set(res.labels))}"]
    for name, lab in (("raw", res.labels_raw), ("validated", res.labels)):
        lines.append(f"{name}: recovery F1={common.recovery_f1(lab, groups):.3f}  pair ARI={common.pair_ari(lab, groups):.3f}"
                     f"  coherence={common.spatial_coherence(lab, bench.A_val):.3f}")
    # expression-only baseline clustered to the same number of subtypes, averaged over seeds
    own_emb = bench.embedding({}, "uniform", "expression", 0.0)
    g_own = knn_graph(own_emb, cfg.LEIDEN_NEIGHBORS)
    n_sub = len(set(res.labels))
    base = [resolution_for_n_clusters(g_own, n_sub, seed=s)[1] for s in cfg.SEEDS]
    lines.append(f"expression only, {n_sub} clusters, mean of {len(base)} seeds: "
                 f"recovery F1={np.mean([common.recovery_f1(b, groups) for b in base]):.3f}  "
                 f"pair ARI={np.mean([common.pair_ari(b, groups) for b in base]):.3f}  "
                 f"coherence={np.mean([common.spatial_coherence(b, bench.A_val) for b in base]):.3f}")
    cont = pd.crosstab(pd.Series(res.labels, name="subtype"), pd.Series(groups, name="reference"))
    cont.to_csv(out / "final_contingency.csv")
    print("\nSubtype x reference:\n", cont.to_string())

    # ---- composition around each subtype (kernel-weighted, whole-tissue neighbours) ----
    W = bench.weights(spec, best["kernel"])
    comp = neighbour_matrix(bench.inputs, W, "composition")
    cats = bench.inputs.categories
    comp_df = pd.DataFrame(comp, columns=cats).groupby(res.labels).mean()
    top = comp_df.mean().sort_values(ascending=False).index[:12]
    comp_df = comp_df[top]

    # ---- marker genes ----
    X = adata.X[idx]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    labels = np.asarray(res.labels)
    uniq = sorted(set(labels), key=int)
    mean_expr = np.vstack([np.asarray(X[labels == u].mean(axis=0)).ravel() for u in uniq])
    rest = np.vstack([np.asarray(X[labels != u].mean(axis=0)).ravel() if (labels != u).any() else np.zeros(X.shape[1]) for u in uniq])
    genes = list(adata.var_names)
    picks = []
    for k in range(len(uniq)):
        for j in np.argsort(-(mean_expr[k] - rest[k]))[:3]:
            picks.append(genes[j])
    for g in ("SPP1", "SELENOP"):
        if g in genes and g not in picks:
            picks.append(g)
    cols = [genes.index(g) for g in dict.fromkeys(picks)]
    sub = mean_expr[:, cols]
    z = (sub - sub.mean(0)) / np.where(sub.std(0) > 0, sub.std(0), 1)

    # ---- figure ----
    plt = common.plt()
    coords = np.asarray(adata.obsm[cfg.SPATIAL_KEY])[idx]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    common.scatter_labels(axes[0, 0], coords, np.where(groups == "other", "unlabelled", groups), "Reference (Oliveira et al.)")
    common.scatter_labels(axes[0, 1], coords, res.labels, f"Spatial subtypes (n = {len(uniq)})", grey=("NA",))
    im = axes[1, 0].imshow(comp_df.to_numpy(), aspect="auto", cmap="magma")
    axes[1, 0].set_xticks(range(comp_df.shape[1]), comp_df.columns, rotation=60, ha="right", fontsize=8)
    axes[1, 0].set_yticks(range(comp_df.shape[0]), [f"subtype {u}" for u in comp_df.index])
    axes[1, 0].set_title("Neighbourhood composition (kernel-weighted fraction)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.04)
    im2 = axes[1, 1].imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[1, 1].set_xticks(range(len(cols)), [genes[c] for c in cols], rotation=60, ha="right", fontsize=8)
    axes[1, 1].set_yticks(range(len(uniq)), [f"subtype {u}" for u in uniq])
    axes[1, 1].set_title("Marker genes (mean log expression, z-scored across subtypes)")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.04)
    fig.suptitle("Final spatial subtypes" + ("   [SIMULATED DEMO DATA]" if args.demo else ""), y=0.99)
    fig.tight_layout()
    common.savefig(fig, out, "03_final_subtypes")
    (out / "final_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
