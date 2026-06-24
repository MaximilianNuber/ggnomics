"""Generate all README images and save them to docs/img/.

Run from the repo root:
    python docs/generate_readme_plots.py
"""

import matplotlib
matplotlib.use("Agg")

import importlib.util
import pathlib
import sys

# Ensure the repo root is on sys.path so `ggnomics` can be imported.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load 00_mock_data.py via importlib (filename starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "mock_data", _REPO_ROOT / "examples" / "00_mock_data.py"
)
_mock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mock)  # type: ignore[union-attr]

make_mock_anndata    = _mock.make_mock_anndata
make_mock_de_results = _mock.make_mock_de_results
make_mock_coefs      = _mock.make_mock_coefs
make_mock_mudata     = _mock.make_mock_mudata
make_mock_repertoire = _mock.make_mock_repertoire

import numpy as np
import pandas as pd

from ggnomics import (
    plot_scatter,
    plot_umap,
    plot_pca,
    plot_expression,
    plot_dot,
    plot_heatmap,
    plot_coldata,
    plot_highest_exprs,
    plot_abundance,
    plot_violin_stats,
    plot_box_stats,
    plot_scatter_marginal,
    plot_embedding_panel,
    plot_volcano,
    plot_ma,
    plot_coef_lollipop,
    plot_coef_expression,
    plot_pseudobulk_qc,
    plot_pseudobulk_de,
    plot_bimodal_scatter,
    plot_adt_qc,
    plot_clonotype_abundance,
    plot_clonotype_overlap,
    plot_clonotype_embedding,
)
from ggnomics.heatmap import heatmap_from_matrix

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT = _REPO_ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build mock data (n_cells = 800)
# ---------------------------------------------------------------------------
print("Building mock data (n_cells=800) …")
adata       = make_mock_anndata(n_cells=800)
de_df       = make_mock_de_results()
coefs       = make_mock_coefs()
mudata      = make_mock_mudata(n_cells=800)

top20 = list(adata.var_names[:20])
top10 = list(adata.var_names[:10])
results_dict = {c: make_mock_de_results() for c in list(adata.obs["cluster"].unique())[:4]}

# First two gene names (stand-ins for "CD3E" / "CD8A" in README examples)
_g1 = adata.var_names[0]   # Gene0001
_g2 = adata.var_names[1]   # Gene0002

# Build adata_prot: combine protein features + isotype controls into one AnnData
adata_prot = None
try:
    import anndata as _ad

    _prot = mudata["prot"]
    _iso  = mudata["iso"]
    # Rename IgG2a → IgG2 so the call matches isotype_controls=["IgG1","IgG2"]
    _X = np.concatenate([np.asarray(_prot.X), np.asarray(_iso.X[:, :2])], axis=1)
    adata_prot = _ad.AnnData(X=_X.astype(np.float32))
    adata_prot.var_names  = list(_prot.var_names) + ["IgG1", "IgG2"]
    adata_prot.obs_names  = _prot.obs_names.copy()
    for _col in _prot.obs.columns:
        adata_prot.obs[_col] = _prot.obs[_col].values
except Exception as _e:
    print(f"  ⚠ Could not build adata_prot: {_e}")

# Simple numpy matrix for the heatmap_matrix demo (20 rows × 10 cols)
_rng   = np.random.default_rng(42)
matrix = _rng.normal(0, 1, (20, 10))

print("Mock data ready.\n")

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_plot(p, name, width=6, height=4):
    p.save(str(OUT / name), dpi=150, width=width, height=height)
    print(f"✓ saved docs/img/{name}")


def _save_compose(c, name, width=10, height=5):
    c.save(str(OUT / name), dpi=150, width=width, height=height)
    print(f"✓ saved docs/img/{name}")


def _save_result(r, name):
    r.savefig(str(OUT / name), dpi=150)
    print(f"✓ saved docs/img/{name}")


# ===========================================================================
# Dimensionality reduction
# ===========================================================================

try:
    _save_plot(plot_umap(adata, color="cluster"), "reduced_dim_cluster.png")
except Exception as e:
    print(f"⚠ reduced_dim_cluster.png failed: {e}")

try:
    _save_plot(plot_umap(adata, color=_g1), "reduced_dim_gene.png")
except Exception as e:
    print(f"⚠ reduced_dim_gene.png failed: {e}")

try:
    _save_plot(
        plot_umap(adata, color="cluster", facet_by="batch"),
        "reduced_dim_facet.png", width=9, height=4,
    )
except Exception as e:
    print(f"⚠ reduced_dim_facet.png failed: {e}")

try:
    _save_plot(plot_pca(adata, color="cluster", components=(1, 3)), "pca_components.png")
except Exception as e:
    print(f"⚠ pca_components.png failed: {e}")

# ===========================================================================
# General scatter
# ===========================================================================

try:
    _save_plot(
        plot_scatter(adata, x="n_counts", y="n_genes_detected", color="cluster"),
        "scatter_obs.png",
    )
except Exception as e:
    print(f"⚠ scatter_obs.png failed: {e}")

try:
    _save_plot(
        plot_scatter(adata, x=_g1, y=_g2, color="cluster"),
        "scatter_gene.png",
    )
except Exception as e:
    print(f"⚠ scatter_gene.png failed: {e}")

# ===========================================================================
# Multi-panel embedding
# ===========================================================================

try:
    _save_compose(
        plot_embedding_panel(adata, features=top10[:4], ncol=2),
        "embedding_panel.png",
    )
except Exception as e:
    print(f"⚠ embedding_panel.png failed: {e}")

# ===========================================================================
# Statistical plots
# ===========================================================================

try:
    _save_plot(
        plot_violin_stats(adata, _g1, group_by="cluster", annotation="stars"),
        "violin_stats.png",
    )
except Exception as e:
    print(f"⚠ violin_stats.png failed: {e}")

try:
    _save_plot(
        plot_box_stats(adata, "n_counts", group_by="batch"),
        "box_stats.png",
    )
except Exception as e:
    print(f"⚠ box_stats.png failed: {e}")

# ===========================================================================
# Scatter with marginals
# ===========================================================================

try:
    _save_compose(
        plot_scatter_marginal(adata, x="n_counts", y="n_genes_detected", color="cluster"),
        "scatter_marginal.png", width=8, height=6,
    )
except Exception as e:
    print(f"⚠ scatter_marginal.png failed: {e}")

# ===========================================================================
# Differential expression
# ===========================================================================

try:
    _save_plot(plot_volcano(de_df, gene_col="gene", label_top_n=8), "volcano.png")
except Exception as e:
    print(f"⚠ volcano.png failed: {e}")

try:
    _save_plot(plot_ma(de_df, gene_col="gene"), "ma_plot.png")
except Exception as e:
    print(f"⚠ ma_plot.png failed: {e}")

# ===========================================================================
# Elastic net / LASSO coefficient plots
# ===========================================================================

try:
    _save_plot(plot_coef_lollipop(coefs, top_n=15), "coef_lollipop.png")
except Exception as e:
    print(f"⚠ coef_lollipop.png failed: {e}")

try:
    _save_plot(
        plot_coef_expression(adata, coefs, group_by="cluster", plot_type="dot"),
        "coef_expression.png",
    )
except Exception as e:
    print(f"⚠ coef_expression.png failed: {e}")

# ===========================================================================
# Heatmaps
# ===========================================================================

try:
    # heatmap_from_matrix = plot_heatmap_matrix equivalent
    _save_plot(heatmap_from_matrix(matrix, zscore_rows=True), "heatmap_matrix.png")
except Exception as e:
    print(f"⚠ heatmap_matrix.png failed: {e}")

try:
    # plot_heatmap with group_by = plot_heatmap_grouped equivalent
    _save_plot(
        plot_heatmap(adata, features=top20, group_by="cluster", cluster_rows=True),
        "heatmap_grouped.png",
    )
except Exception as e:
    print(f"⚠ heatmap_grouped.png failed: {e}")

try:
    # Full clustering on rows + cols = clustermap equivalent
    _save_plot(
        plot_heatmap(adata, features=top20, group_by="cluster",
                     cluster_rows=True, cluster_cols=True),
        "clustermap.png",
    )
except Exception as e:
    print(f"⚠ clustermap.png failed: {e}")

# ===========================================================================
# Expression plots
# ===========================================================================

try:
    _save_plot(plot_dot(adata, features=top10, group_by="cluster"), "dot_plot.png")
except Exception as e:
    print(f"⚠ dot_plot.png failed: {e}")

try:
    _save_plot(
        plot_expression(adata, features=top10[:3], group_by="cluster"),
        "expression_violin.png", width=8, height=4,
    )
except Exception as e:
    print(f"⚠ expression_violin.png failed: {e}")

try:
    _save_plot(plot_highest_exprs(adata, n=20), "highest_exprs.png", height=6)
except Exception as e:
    print(f"⚠ highest_exprs.png failed: {e}")

# ===========================================================================
# Metadata plots
# ===========================================================================

try:
    _save_plot(
        plot_coldata(adata, x="n_counts", y="n_genes_detected", color_by="cluster"),
        "coldata_scatter.png",
    )
except Exception as e:
    print(f"⚠ coldata_scatter.png failed: {e}")

try:
    _save_plot(
        plot_coldata(adata, x="cluster", y="n_counts", shape="violin"),
        "coldata_violin.png",
    )
except Exception as e:
    print(f"⚠ coldata_violin.png failed: {e}")

try:
    _save_plot(
        plot_abundance(adata, group_by="cluster", color_by="batch"),
        "abundance.png",
    )
except Exception as e:
    print(f"⚠ abundance.png failed: {e}")

# ===========================================================================
# Pseudobulk
# ===========================================================================

try:
    _save_compose(
        plot_pseudobulk_qc(adata, sample_by="sample", group_by="cluster", condition_by="batch"),
        "pseudobulk_qc.png", width=12, height=8,
    )
except Exception as e:
    print(f"⚠ pseudobulk_qc.png failed: {e}")

try:
    _save_compose(
        plot_pseudobulk_de(results_dict, mode="volcano", ncol=2),
        "pseudobulk_de_volcano.png", width=10, height=8,
    )
except Exception as e:
    print(f"⚠ pseudobulk_de_volcano.png failed: {e}")

try:
    _save_plot(
        plot_pseudobulk_de(results_dict, mode="summary_bar"),
        "pseudobulk_de_bar.png",
    )
except Exception as e:
    print(f"⚠ pseudobulk_de_bar.png failed: {e}")

# ===========================================================================
# CITEseq / Multimodal
# ===========================================================================

try:
    _x_feat = mudata["rna"].var_names[0]   # Gene0001
    _y_feat = mudata["prot"].var_names[0]  # Prot01
    _save_plot(
        plot_bimodal_scatter(mudata, x_feature=_x_feat, y_feature=_y_feat,
                              x_mod="rna", y_mod="prot"),
        "bimodal_scatter.png",
    )
except Exception as e:
    print(f"⚠ bimodal_scatter.png failed: {e}")

try:
    if adata_prot is None:
        print("⚠ adt_qc.png skipped (adata_prot unavailable — mudata import failed)")
    else:
        _save_plot(
            plot_adt_qc(adata_prot, isotype_controls=["IgG1", "IgG2"]),
            "adt_qc.png", width=9, height=4,
        )
except Exception as e:
    print(f"⚠ adt_qc.png failed: {e}")

# ===========================================================================
# Repertoire (TCR/BCR)
# ===========================================================================

try:
    _save_plot(
        plot_clonotype_abundance(adata, clonotype_col="clonotype_id"),
        "clonotype_abundance.png",
    )
except Exception as e:
    print(f"⚠ clonotype_abundance.png failed: {e}")

try:
    _save_result(
        plot_clonotype_overlap(adata, clonotype_col="clonotype_id", sample_col="sample"),
        "clonotype_overlap.png",
    )
except Exception as e:
    print(f"⚠ clonotype_overlap.png failed: {e}")

try:
    _save_plot(
        plot_clonotype_embedding(adata, clonotype_col="clonotype_id"),
        "clonotype_embedding.png",
    )
except Exception as e:
    print(f"⚠ clonotype_embedding.png failed: {e}")

# ===========================================================================
# Composition example
# ===========================================================================

try:
    _comp = plot_umap(adata, color="cluster") | plot_violin_stats(adata, _g1, group_by="cluster")
    _save_compose(_comp, "composition_example.png")
except Exception as e:
    print(f"⚠ composition_example.png failed: {e}")

# ===========================================================================
# Final count
# ===========================================================================
pngs = sorted(OUT.glob("*.png"))
print(f"\n✓ Done — {len(pngs)} PNG(s) in docs/img/")
