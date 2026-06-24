# Plot gallery

## Embedding
```python
p = plot_reduced_dim(df, x="umap_1", y="umap_2", color="cluster")
```

## Expression violin (Seurat-like)
```python
p = expression_violin(df, value="expr", group="cluster", log1p=True)
```

## Ridge density
```python
p = ridge_density(df, value="expr", group="cluster")
```

## Volcano
```python
p = volcano_plot(de_df, log2fc="log2FC", padj="FDR")
```

## Heatmap
```python
p = heatmap_from_matrix(mat, row_names=genes, col_names=clusters, zscore_rows=True)
```

## Marker dot-plot
```python
p = marker_dotplot_from_matrix(expr, groups=clusters, genes=["CD3D", "MS4A1"])
```

## QC
```python
p = qc_scatter(df, x="n_counts", y="n_genes", color="mito_frac")
p = qc_histogram(df, value="mito_frac")
```

## Composition
```python
p = cluster_composition_barplot(df, cluster_col="cluster", group_col="condition", normalize=True)
```
