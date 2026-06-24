"""Backwards-compatibility re-export: plot_reduced_dim now lives in scatter.py."""

from .scatter import plot_reduced_dim, plot_umap, plot_pca, plot_tsne

__all__ = ["plot_reduced_dim", "plot_umap", "plot_pca", "plot_tsne"]
