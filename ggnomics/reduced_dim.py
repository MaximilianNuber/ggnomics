"""Backward-compatible re-exports for the embedding plot hierarchy."""

from .scatter import (
    dim_plot,
    plot_embedding,
    plot_pca,
    plot_reduced_dim,
    plot_tsne,
    plot_umap,
)

__all__ = [
    "plot_embedding",
    "plot_reduced_dim",
    "dim_plot",
    "plot_umap",
    "plot_pca",
    "plot_tsne",
]
