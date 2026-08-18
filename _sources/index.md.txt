# ggnomics

**Composable genomics visualization with plotnine.**

`ggnomics` provides plotting functions for bulk, single-cell, and multimodal
genomics in Python. It covers familiar visualizations from tools such as
scater, Seurat, and scanpy while returning ordinary
[plotnine](https://plotnine.org) objects that can be extended with the grammar
of graphics.

The package works with tidy `pandas` data and common genomics containers,
including AnnData, BiocPy `SingleCellExperiment` and `SummarizedExperiment`,
and MuData. The public `plot_*` functions use Python's `singledispatch`, so the
function name and principal arguments stay the same across containers, and
container support is registered only when the corresponding optional package is
installed.

## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Tutorials <tutorials/index>
Module Reference <api/modules>
Contributing <contributing>
Code of Conduct <code_of_conduct>
License <license>
Authors <authors>
Changelog <changelog>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[MyST]: https://myst-parser.readthedocs.io/en/latest/
