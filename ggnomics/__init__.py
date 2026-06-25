"""ggnomics — plotnine-based genomics / single-cell plotting library."""

__all__ = [
    # Significance brackets
    "geom_signif",
    "run_comparisons",
    "map_pvalue_to_stars",
    # Scatter hierarchy
    "plot_scatter",
    "plot_reduced_dim",
    "plot_umap",
    "plot_pca",
    "plot_tsne",
    # Expression / annotation
    "plot_expression",
    "plot_dot",
    "plot_heatmap",
    "plot_coldata",
    "plot_rowdata",
    "plot_highest_exprs",
    "plot_pairs",
    "plot_abundance",
    # Stats
    "plot_violin_stats",
    "plot_box_stats",
    "plot_scatter_marginal",
    "plot_embedding_panel",
    # DE
    "plot_volcano",
    "plot_ma",
    "plot_coef_lollipop",
    "plot_coef_expression",
    # Pseudobulk
    "plot_pseudobulk_qc",
    "plot_pseudobulk_de",
    # Multimodal
    "plot_bimodal_scatter",
    "plot_adt_qc",
    # Repertoire
    "plot_clonotype_abundance",
    "plot_clonotype_overlap",
    "plot_clonotype_embedding",
    # Palettes / utilities
    "get_palette",
    "TABLEAU_10",
    "TABLEAU_20",
    "HeatmapResult",
    # Composition helpers
    "hstack",
    "vstack",
    "grid",
    "annotate_composition",
    "save_composition",
    # Legacy API (kept for backwards compatibility)
    "expression_violin",
    "expression_violin_sce",
    "expression_violin_se",
    "volcano_plot",
    "heatmap_long",
    "heatmap_from_matrix",
    "marker_dotplot",
    "marker_dotplot_from_matrix",
    "qc_scatter",
    "qc_histogram",
    "cluster_composition_barplot",
    "ridge_density",
]

# Significance brackets
from .signif import geom_signif, run_comparisons, map_pvalue_to_stars

# New scatter hierarchy
from .scatter import plot_scatter, plot_reduced_dim, plot_umap, plot_pca, plot_tsne

# Expression / annotation
from .expression import plot_expression, plot_dot, plot_heatmap
from .coldata import plot_coldata, plot_rowdata
from .highest_exprs import plot_highest_exprs
from .pairs import plot_pairs
from .abundance import plot_abundance
from .palettes import get_palette, TABLEAU_10, TABLEAU_20
from ._utils import HeatmapResult
from ._compose import hstack, vstack, grid, annotate_composition, save_composition

# Stats
from .stats_plots import (
    plot_violin_stats,
    plot_box_stats,
    plot_scatter_marginal,
    plot_embedding_panel,
)

# DE
from .de_plots import plot_volcano, plot_ma, plot_coef_lollipop, plot_coef_expression

# Pseudobulk
from .pseudobulk import plot_pseudobulk_qc, plot_pseudobulk_de

# Multimodal
from .multimodal import plot_bimodal_scatter, plot_adt_qc

# Repertoire
from .repertoire import (
    plot_clonotype_abundance,
    plot_clonotype_overlap,
    plot_clonotype_embedding,
)

# Legacy API
from .violin import (
    expression_violin,
    expression_violin_sce,
    expression_violin_se,
)
from .volcano import volcano_plot
from .heatmap import heatmap_long, heatmap_from_matrix
from .dotplot import marker_dotplot, marker_dotplot_from_matrix
from .qc import qc_scatter, qc_histogram
from .composition import cluster_composition_barplot
from .ridge import ridge_density

# Sub-module imports (unchanged)
from .bulk import pca as bulk_pca
from .singlecell import pca as sc_pca
from .singlecell import umap as sc_umap
