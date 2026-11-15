"""
Benchmark 2: how much neighbourhood, how wide, and which feature?

Using the graph and kernel chosen by script 01 (figures/best_config.json):
  A. lambda sweep for the two neighbour features (mean expression vs cell-type composition)
  B. neighbourhood size sweep (radius graph, best lambda)
  C. spatial coherence across the lambda sweep

Writes figures/lambda_sweep.csv, figures/radius_sweep.csv, updates
figures/best_config.json and writes figures/02_lambda_scale_sweep.png.

    python benchmarks/02_lambda_scale_sweep.py [--demo]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from benchmarks import common, config as cfg


def main():
    args = common.parse_args(__doc__)
    adata, mask, out = common.load(args)
    best = common.load_best() if not args.demo else json.loads((out / "best_config.json").read_text())
    spec = {k: best[k] for k in ("mode", "radius", "n_neighbors", "rings") if k in best}
    kernel = best["kernel"]
    print(f"Using graph {spec} with kernel {kernel!r}")
    bench = common.Bench(adata, mask, composition=True)

    # ---- A/C: lambda x feature ----
    rows = []
    for feature in ("expression", "composition"):
        for lam in cfg.LAMBDAS:
            r = bench.evaluate(spec, kernel, lam, feature)
            rows.append(r)
            print(f"{feature:11s} lam={lam:.1f}  recovery={r['recovery_f1']:.3f} +/- {r['recovery_f1_sd']:.3f}"
                  f"  emb_acc={r['embedding_acc']:.3f}  coherence={r['coherence']:.3f}")
    lam_df = pd.DataFrame(rows)
    lam_df.to_csv(out / "lambda_sweep.csv", index=False)

    expr = lam_df[(lam_df["feature"] == "expression") & (lam_df["lam"] > 0)]
    best_lam = float(expr.sort_values("recovery_f1", ascending=False).iloc[0]["lam"])
    comp_best = lam_df[(lam_df["feature"] == "composition") & (lam_df["lam"] > 0)]["recovery_f1"].max()
    feature = "composition" if comp_best > expr["recovery_f1"].max() else "expression"
    print(f"\nBest lambda (expression): {best_lam};  best feature overall: {feature}")

    # ---- B: neighbourhood size ----
    rows = []
    for radius in cfg.RADII_UM:
        r = bench.evaluate(dict(mode="radius", radius=radius), kernel, best_lam, "expression")
        rows.append(r)
        print(f"radius {radius:5.1f} um  deg={r['mean_degree']:6.1f}  recovery={r['recovery_f1']:.3f} +/- {r['recovery_f1_sd']:.3f}")
    rad_df = pd.DataFrame(rows).assign(radius=list(cfg.RADII_UM))
    rad_df.to_csv(out / "radius_sweep.csv", index=False)
    best_radius = float(rad_df.sort_values("recovery_f1", ascending=False).iloc[0]["radius"])

    new = dict(best, lam=best_lam, feature=feature, best_radius_um=best_radius)
    (out / "best_config.json").write_text(json.dumps(new, indent=2))

    # ---- figure ----
    plt = common.plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    colors = {"expression": "#2a7fd4", "composition": "#d9622b"}
    for feat, g in lam_df.groupby("feature"):
        axes[0].errorbar(g["lam"], g["recovery_f1"], g["recovery_f1_sd"], marker="o", capsize=3, color=colors[feat], label=feat)
        axes[2].errorbar(g["lam"], g["coherence"], g["coherence_sd"], marker="o", capsize=3, color=colors[feat], label=feat)
    lam0 = lam_df[lam_df["lam"] == 0].iloc[0]
    for ax, col in ((axes[0], "recovery_f1"), (axes[2], "coherence")):
        ax.axhline(lam0[col], color="#777", ls="--", lw=1, label="lambda = 0")
        ax.set_xlabel("lambda (neighbourhood share of variance)")
    axes[0].set_ylabel("Recovery F1"); axes[0].set_title("A. Lambda and neighbour feature"); axes[0].legend(frameon=False, fontsize=8)
    axes[2].set_ylabel("Spatial coherence"); axes[2].set_title("C. Spatial coherence of the clusters")
    axes[1].errorbar(rad_df["radius"], rad_df["recovery_f1"], rad_df["recovery_f1_sd"], marker="o", capsize=3, color=colors["expression"])
    axes[1].axhline(lam0["recovery_f1"], color="#777", ls="--", lw=1)
    axes[1].set_xlabel("neighbourhood radius (um)"); axes[1].set_ylabel("Recovery F1")
    axes[1].set_title(f"B. Neighbourhood size (lambda = {best_lam})")
    fig.suptitle("Lambda, neighbourhood size and neighbour feature" + ("   [SIMULATED DEMO DATA]" if args.demo else ""), y=1.03)
    fig.tight_layout()
    common.savefig(fig, out, "02_lambda_scale_sweep")


if __name__ == "__main__":
    main()
