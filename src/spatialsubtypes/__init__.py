"""Spatially informed subtyping of one cell type in spatial transcriptomics data."""

from .cluster import cluster_embedding, cluster_to_counts, knn_graph, resolution_for_n_clusters
from .features import embed, neighbour_block, prepare_inputs, select_hvg
from .graph import adjacency, apply_kernel, build_graph
from .pipeline import SubtypeResult, find_spatial_subtypes
from .validate import validate_subtypes

__version__ = "0.1.0"

__all__ = [
    "find_spatial_subtypes", "SubtypeResult",
    "build_graph", "apply_kernel", "adjacency",
    "prepare_inputs", "neighbour_block", "embed", "select_hvg",
    "cluster_embedding", "cluster_to_counts", "knn_graph", "resolution_for_n_clusters",
    "validate_subtypes",
]
