"""Compare neighbour-block scalings on synthetic tissue where target cells carry NO niche signal of their own."""
import numpy as np, scipy.sparse as sp
from anndata import AnnData
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from src.spatialsubtypes import build_graph, apply_kernel, prepare_inputs, neighbour_block, embed

def make(seed, G=90, n_genes=2000, n_marker=8, heavy=100, marker_shift=1.0):
    rng = np.random.default_rng(seed)
    xy = np.stack(np.meshgrid(np.arange(G), np.arange(G)), -1).reshape(-1, 2) * 8.0
    n = len(xy)
    niche = (xy[:, 0] + 120 * np.sin(xy[:, 1] / 150) > G * 4).astype(int)
    target = rng.random(n) < 0.10
    mu = rng.uniform(0.05, 1.5, n_genes); sd = rng.uniform(0.2, 0.8, n_genes)
    sd[:heavy] = rng.uniform(2.0, 3.5, heavy)              # heavy-variance, niche-unrelated genes
    sd[heavy:heavy + 2 * n_marker] = 0.5
    X = mu + sd * rng.standard_normal((n, n_genes))
    m = heavy
    X[:, m:m + n_marker][niche == 0] += marker_shift          # niche-0 markers (non-target cells only)
    X[:, m + n_marker:m + 2 * n_marker][niche == 1] += marker_shift
    X[target][:, m:m + 2 * n_marker]
    # target cells: remove niche signal from own expression
    X[target, m:m + 2 * n_marker] = (mu + sd * rng.standard_normal((target.sum(), n_genes)))[:, m:m + 2 * n_marker]
    X = np.clip(X, 0, None).astype(np.float32)
    ad = AnnData(X=sp.csr_matrix(X), obsm={"spatial": xy}); ad.obs["niche"] = niche
    return ad, target, niche, xy

rows = []
for seed in range(4):
    ad, target, niche, xy = make(seed)
    inp = prepare_inputs(ad, target, n_hvg=2000)
    q = inp.target_idx; y = niche[q]
    for mode, kw in [("knn", dict(n_neighbors=15)), ("radius", dict(radius=32.0))]:
        W = apply_kernel(build_graph(xy, mode, query=q, **kw), "gaussian")
        for scaling in ("input_std", "unit_variance", "per_gene"):
            nb = neighbour_block(inp, W, "expression", scaling)
            for lam in (0.2, 0.5):
                e = embed(inp.own, nb, lam, 30, 0)
                sil = silhouette_score(e, y)
                pred = cross_val_predict(KNeighborsClassifier(15), e, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
                rows.append((mode, scaling, lam, sil, balanced_accuracy_score(y, pred)))
import pandas as pd
df = pd.DataFrame(rows, columns=["graph", "scaling", "lam", "silhouette", "knn_bal_acc"])
print(df.groupby(["graph", "lam", "scaling"])[["silhouette", "knn_bal_acc"]].agg(["mean", "std"]).round(3).to_string())
