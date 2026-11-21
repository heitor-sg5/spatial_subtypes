"""Parameters and column names used in benchmarks"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "VisiumHD_P2CRC_8um_Annotated.h5ad"
FIGURES_DIR = REPO_ROOT / "figures"
DEMO_FIGURES_DIR = FIGURES_DIR / "demo"

# ---- Data conventions (Oliveira et al. 2025, patient P2CRC, 8 um bins) ----
SPATIAL_KEY = "spatial_um"
CELL_TYPE_KEY = "DeconvolutionLabel1"
TARGET_TYPE = "Macrophage"
PERIPHERY_KEY = "Periphery"
PERIPHERY_VALUE = "50 micron"
REFERENCE_KEY = "MacrophageSubtype"
REFERENCE_SPP1 = "Macrophage-SPP1+"
REFERENCE_SELENOP = "Macrophage-SELENOP+"

# ---- Method settings shared by all scripts ----
N_HVG = 2000
N_PCS = 50
LEIDEN_NEIGHBORS = 15
VALIDATION_NEIGHBORS = 10  # target-only kNN graph 
CLUSTER_COUNTS = (4, 6, 8, 10)  # cluster counts 
SEEDS = (0, 1, 2, 3, 4)  # Leiden seeds

# ---- Script 01: graph x kernel grid ----
GRID_LAM = 0.2
RADIUS_UM = 32.0  # 48 bins on the 8 um grid
KNN_K = 48
DELAUNAY_RINGS = 3  # about 41 bins on the 8 um grid
GRID_KERNELS = ("uniform", "gaussian", "exponential", "reciprocal")
BANKSY_STYLE = dict(mode="knn", n_neighbors=15, kernel="gaussian", lam=0.2)

# ---- Script 02: sweeps ----
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9)
RADII_UM = (12.0, 20.0, 32.0, 48.0, 64.0)

# ---- Script 03: final subtypes ----
FINAL_START_CLUSTERS = 8
FINAL_VALIDATION = dict(n_perm=1000, alpha=0.05, min_cluster_size=20, auc_threshold=0.8)

# ---- Fallback if script 01/02 have not written figures/best_config.json ----
DEFAULT_BEST = dict(mode="knn", n_neighbors=KNN_K, kernel="gaussian", lam=0.2, feature="expression")
