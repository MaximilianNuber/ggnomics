"""ggnomics — plotnine-based genomics / single-cell plotting library."""

__all__ = [
    # Significance brackets
    "geom_signif",
    "run_comparisons",
    "map_pvalue_to_stars",
    # Scatter hierarchy
    "plot_scatter",
    "plot_embedding",
    "plot_reduced_dim",
    "dim_plot",
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
    "resolve_palette",
    "TABLEAU_10",
    "TABLEAU_20",
    "BIOC_COLORS",
    "IGV_DEFAULT",
    "IGV_ALTERNATING",
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
    "upset",
]

# Significance brackets
# independent submodules
from . import upset

# Register only the optional container packages that are actually installed.
# This must happen after the public generic functions have been imported.
from ._backends import register_installed_backends as _register_installed_backends
from ._compose import annotate_composition, grid, hstack, save_composition, vstack
from ._utils import HeatmapResult
from .abundance import plot_abundance

# Sub-module imports (unchanged)
from .bulk import pca as bulk_pca
from .coldata import plot_coldata, plot_rowdata
from .composition import cluster_composition_barplot

# DE
from .de_plots import plot_coef_expression, plot_coef_lollipop, plot_ma, plot_volcano
from .dotplot import marker_dotplot, marker_dotplot_from_matrix

# Expression / annotation
from .expression import plot_dot, plot_expression, plot_heatmap
from .heatmap import heatmap_from_matrix, heatmap_long
from .highest_exprs import plot_highest_exprs

# Multimodal
from .multimodal import plot_adt_qc, plot_bimodal_scatter
from .pairs import plot_pairs
from .palettes import (
    BIOC_COLORS,
    IGV_ALTERNATING,
    IGV_DEFAULT,
    TABLEAU_10,
    TABLEAU_20,
    get_palette,
    resolve_palette,
)

# Pseudobulk
from .pseudobulk import plot_pseudobulk_de, plot_pseudobulk_qc
from .qc import qc_histogram, qc_scatter

# Repertoire
from .repertoire import (
    plot_clonotype_abundance,
    plot_clonotype_embedding,
    plot_clonotype_overlap,
)
from .ridge import ridge_density

# New scatter hierarchy
from .scatter import (
    dim_plot,
    plot_embedding,
    plot_pca,
    plot_reduced_dim,
    plot_scatter,
    plot_tsne,
    plot_umap,
)
from .signif import geom_signif, map_pvalue_to_stars, run_comparisons
from .singlecell import pca as sc_pca
from .singlecell import umap as sc_umap

# Stats
from .stats_plots import (
    plot_box_stats,
    plot_embedding_panel,
    plot_scatter_marginal,
    plot_violin_stats,
)

# Legacy API
from .violin import (
    expression_violin,
    expression_violin_sce,
    expression_violin_se,
)
from .volcano import volcano_plot

# Register only the optional container packages that are actually installed.
# This call must come after every public generic function above has been
# imported: the backends register their implementations onto those generics.
# (The import itself sits at the top with the others - importing
# ggnomics._backends has no side effects; only this call does.)
_register_installed_backends()
del _register_installed_backends
