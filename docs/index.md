# gg-singlecell

Plotnine-based plotting utilities for single-cell and genomics.

## Installation
```bash
pip install -e .
# extras
pip install -e .[test]
pip install -e .[docs]
```

## Quick start
```python
from gg_singlecell import plot_reduced_dim
p = plot_reduced_dim(df, x="umap_1", y="umap_2", color="cluster")
```

See more plots in [Plots](plots.md).
