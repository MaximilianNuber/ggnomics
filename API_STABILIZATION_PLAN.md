# ggnomics API stabilization plan

*Status: discussion draft. Written from a full read-through of `ggnomics/` (~11.6k lines)
on 2026-08-21, ahead of committing to a stable public API.*

## 1. Why this document

`ggnomics` already has a working, well-tested architecture: a `singledispatch` generic
per plot family, a canonical `pandas.DataFrame` implementation, and lazily-registered
adapters for `AnnData`, `SingleCellExperiment`, `SummarizedExperiment`, and `MuData`
(`tests/test_public_api_contract.py` pins this contract — return types, non-mutation,
error messages). That pattern is the right foundation and does not need to change.

What's *not* settled is the **naming and grouping of the ~50 public functions built on
top of it** — which verb/noun a user reaches for, which names are aliases of which,
and where the boundary sits between "plot", "compute", and "legacy". This document is
an inventory of the current state, the specific inconsistencies that will bite users
once the API is declared stable, and concrete proposals for resolving them.

Nothing here is a decision — several sections end with an explicit choice for you to
make.

## 2. The architecture that already works (keep this)

Every modern plot function follows the same three-to-four layer shape:

1. **Private plotnine builder** — pure geometry from an already-tidy `DataFrame`, no
   data extraction (e.g. `_scatter_ggplot`, `_build_adt_qc_plot`,
   `_build_highest_exprs_plot`). Independently testable, reused across containers.
2. **Public `@singledispatch` generic**, registered for `pd.DataFrame` — this is the
   canonical implementation and "the interface". It validates columns, builds the tidy
   frame, and calls layer 1. Its docstring is the one source of truth for the
   function's contract.
3. **Container adapters** in `ggnomics/_backends/{anndata,singlecellexperiment,summarizedexperiment,mudata}.py`
   — extract a `DataFrame` slice from the container (sparse-aware, only densifying the
   requested columns) and re-dispatch to layer 2. Registered only if the optional
   package is importable (`_backends/__init__.py`).
4. **Convenience wrappers** on top of a layer-2/3 generic, for a fixed common case
   (e.g. `plot_umap(data, ...)` = `plot_embedding(data, dimred="X_umap", ...)`).

This gives every function in the "modern" set: one docstring, one error-message
convention (`TypeError` naming the type + remedy, `KeyError` listing available
columns), one non-mutation guarantee, and free multi-container support. **This is the
part of the API that should be declared stable as-is** — the problem is entirely in
the layer above it (which names exist, what they're called, what they alias).

## 3. Two generations of API currently coexist

`ggnomics/__init__.py` explicitly separates them, but both ship in the same
top-level namespace today:

| | Modern (singledispatch) | Legacy (flat, DataFrame-only) |
|---|---|---|
| Embedding | `plot_embedding` / `plot_reduced_dim` / `dim_plot` / `plot_umap` / `plot_pca` / `plot_tsne` | — |
| Expression violin | `plot_expression` | `expression_violin`, `expression_violin_sce`, `expression_violin_se` |
| Dot plot | `plot_dot` | `marker_dotplot`, `marker_dotplot_from_matrix`, `marker_dotplot_sce` |
| Heatmap | `plot_heatmap` | `heatmap_long`, `heatmap_from_matrix` |
| Volcano | `plot_volcano` | `volcano_plot` |
| Composition | `plot_abundance` | `cluster_composition_barplot` |
| QC scatter/hist | `plot_coldata` (covers scatter) | `qc_scatter`, `qc_histogram` |
| Ridge density | *(no modern replacement)* | `ridge_density` |

Every legacy function has a modern 1:1 replacement **except `ridge_density`** (gap —
no modern replacement exists, and it should be exempted from any deprecation until one
does) and the matrix-input convenience wrappers (`*_from_matrix`, which the modern
generics don't offer — see §4.7).

The legacy functions aren't dispatch-based, don't share the modern error-message or
non-mutation conventions, and use a different naming grammar entirely (`volcano_plot`,
`expression_violin`, `ridge_density` put the verb *after* the noun; `heatmap_long`,
`marker_dotplot` put a qualifier after the noun). Keeping ~15 of them in the same flat
`__all__` as the modern set is the single biggest source of "which function do I
use?" confusion right now — worse than any individual naming choice below.

## 4. Specific naming problems to resolve before declaring the API stable

### 4.1 `plot_embedding` has three names for one function — no distinct audiences

```python
plot_reduced_dim = plot_embedding
dim_plot = plot_embedding
```

These are literal object aliases (`ggnomics/scatter.py:392-394`), not wrappers — same
function, same dispatch registry, three import names. `dim_plot` targets Seurat users
(`DimPlot`), `plot_reduced_dim` targets... the same audience as `plot_embedding`,
just with a different word for "embedding". It adds a name to remember without adding
a distinct mental model. Recommendation in §7.1.

### 4.2 `plot_pca` and `plot_umap` mean two different things depending on namespace

This is the sharpest landmine in the current API:

| Call | Behavior |
|---|---|
| `gg.plot_pca(data, color=...)` | Displays an **already-computed** `X_pca` embedding. Container-dispatched (DataFrame/AnnData/SCE). Never runs PCA. |
| `gg.sc_pca.plot_pca(sce, run_if_missing=True, ...)` | **Computes** PCA via scikit-learn if missing, stores it in `reduced_dims`, then plots. SCE-only, not dispatched. |
| `gg.bulk_pca.plot_pca(se, pca_res=None, n_top_genes=None, ...)` | **Computes** PCA (optionally on top-variance genes) if `pca_res` not supplied, then plots. SE-only. |

Same name, three incompatible signatures and three different "does this run an
analysis?" answers, distinguished only by which submodule you happened to import
(`gg.plot_pca` vs `gg.sc_pca.plot_pca` vs `gg.bulk_pca.plot_pca`). The equivalent
collision exists for `plot_umap` vs `gg.sc_umap.plot_umap`. `bulk.pca` and
`singlecell.pca` are also ~80% duplicate code (`build_plot_df_from_sce` /
`build_plot_df_from_result` do the same "embedding + coldata → tidy DataFrame" job
that `_embedding_frame_from_dataframe` / `_embedding_frame` already do elsewhere).

This is also the one place where the "ggnomics is a visualization layer; BiocPy /
scranpy / PyDESeq2 / scverse do analysis" boundary stated in the README is violated —
`sc_pca.run_pca` / `bulk_pca.run_pca` are genuine analysis functions, not adapters.
Recommendation in §7.2.

### 4.3 `plot_coldata` re-implements what `plot_scatter` already does

For two numeric columns, `plot_coldata` builds its own `geom_point` + continuous
color-scale plot (`ggnomics/coldata.py:110-115`) rather than delegating to
`plot_scatter`, which does the identical thing with a more complete feature set
(facets, adaptive size/stroke, palettes, aspect ratio). `plot_bimodal_scatter`, by
contrast, does delegate to `plot_scatter` for its non-marginal case — showing the
delegation pattern is known and used elsewhere, just not applied consistently. Two
consequences: (a) any future improvement to `plot_scatter`'s numeric-numeric path
silently doesn't reach `plot_coldata`, and (b) it's not obvious to a new user *when*
to reach for `plot_scatter` vs `plot_coldata` — the honest answer today is "`plot_scatter`
takes any two columns; `plot_coldata` takes any two columns from obs/colData and adds
violin/box/bar for categorical x" i.e. `plot_coldata` is a strict superset scoped to
metadata. That scoping rule is real and worth keeping, but should be enforced by
delegation, not duplication.

### 4.4 Container-mediated modality selection has two competing conventions

`plot_bimodal_scatter`'s `AnnData` adapter accepts *both* `x_mod`/`y_mod` (older,
implicit: names an obsm/layer key) and `layer_x`/`layer_y` (newer, explicit layer
name), with the docstring itself noting `layer_x`/`layer_y` were added for backward
compatibility and take precedence. That's a sign the parameter design is still
settling — worth locking down before "stable" rather than after.

### 4.5 Inconsistent return type for the same visual family

`plot_heatmap` returns a bare `ggplot`. `plot_clonotype_overlap` and
`plot_coef_expression(plot_type="heatmap")` return `HeatmapResult` (a `ggplot` +
the underlying matrix) for what is visually the same geom (`geom_tile`). A user who
row-clusters with `plot_heatmap(cluster_rows=True)` gets back the *reordered* matrix
nowhere — they'd have to recompute clustering themselves to get row order. That's the
exact case `HeatmapResult` exists to solve for the other two callers.

### 4.6 Backend coverage is asymmetric in ways that look unintentional

Diffing the `.register()` calls across all four backends:

- **`SummarizedExperiment` never registers `plot_scatter`** (only
  `plot_scatter_marginal`, which requires it internally). A bulk user cannot make a
  plain x/y scatter of two `colData` columns through the dispatch API — looks like an
  oversight, not a deliberate scope limit.
- **`SummarizedExperiment` *does* register `plot_adt_qc` and
  `plot_clonotype_abundance`/`overlap`** — single-cell-protein and immune-repertoire
  concepts, on a container whose defining trait is "no per-cell reduced dimensions."
  Harmless if the columns exist, but it's not obvious this was a deliberate design
  choice vs. mechanical copy-paste across backend files.
- **`MuData` registers only `plot_adt_qc` and `plot_bimodal_scatter`** — none of
  `plot_embedding`, `plot_scatter`, `plot_expression`, `plot_coldata`, etc. have a
  `MuData` adapter, so a `MuData` object is unusable for most of the library. This may
  be intentional (MuData's whole point is cross-modality; maybe every other generic
  is meant to be called on `mdata.mod["rna"]` directly), but it isn't documented as a
  deliberate scope decision anywhere, so it currently reads as "unfinished."
- **Legacy `expression_violin` is registered for `SCE`/`SE` but not `AnnData`** — the
  legacy API's own coverage is inconsistent, on top of being superseded.

### 4.7 No modern replacement for matrix-input convenience wrappers

`marker_dotplot_from_matrix` and `heatmap_from_matrix` (legacy) take a bare
`np.ndarray`/`DataFrame` matrix plus a group vector and do the aggregation themselves,
which is genuinely convenient when a user has a raw expression matrix and no tidy
`DataFrame` yet. The modern `plot_dot`/`plot_heatmap` generics only dispatch on
`DataFrame` and registered containers — there's no `ndarray` path. This isn't a naming
inconsistency like the others in this section, it's a real feature gap: deprecating
the legacy matrix wrappers per §7.3 without replacing them would remove capability,
not just rename it. Resolve by either registering `plot_dot`/`plot_heatmap` for
`np.ndarray` directly (consistent with the dispatch pattern used everywhere else) or
explicitly deciding matrix input is out of scope for the modern API and keeping the
two `*_from_matrix` legacy functions permanently exempt from deprecation.

### 4.8 `ggnomics/pca/` sits in an ambiguous public/private zone

`run_pca_sklearn`, `run_pca_svd`, `PcaResult` live in a package-visible path
(`ggnomics/pca/`, no leading underscore) but are deliberately **not** re-exported from
`ggnomics.__init__` — the plot-gallery article says so explicitly ("It is not
re-exported from the top-level namespace"). A no-underscore top-level subpackage is
normally read as "this is public, just deep"; right now it's public-by-path but
private-by-convention, which is exactly the kind of ambiguity that becomes a support
burden once users start `from ggnomics.pca import PcaResult`. Pick one (see §7.6).

## 5. The "levels" of plot you described, made explicit

Your framing (container extensions vs. results-table plots vs. everything else) is
accurate and is a good organizing axis for docs even if the file layout doesn't
change. Five kinds of public surface exist today:

1. **Container-dispatched plots** — the majority. Take a `DataFrame` *or* a registered
   container; the container adapter reduces to the `DataFrame` case. Examples:
   `plot_embedding` family, `plot_scatter`, `plot_coldata`/`plot_rowdata`,
   `plot_expression`/`plot_dot`/`plot_heatmap`, `plot_abundance`,
   `plot_highest_exprs`, `plot_pairs`, `plot_violin_stats`/`plot_box_stats`,
   `plot_scatter_marginal`, `plot_embedding_panel`, `plot_bimodal_scatter`,
   `plot_adt_qc`, `plot_clonotype_*`, `plot_pseudobulk_qc`.
2. **Results-table plots** — operate on a plain DE/coefficient/count table that has no
   natural container form, so dispatch buys nothing. Examples: `plot_volcano`,
   `plot_ma`, `plot_coef_lollipop`, `plot_pseudobulk_de`. `plot_coef_expression` is a
   hybrid: it takes *both* a container/DataFrame (`data`, for expression) and a
   results table (`coefs`, for feature selection) — worth calling out as its own
   pattern rather than forcing it into one bucket.
3. **Compute + plot combos** — run an analysis, then plot the result. Currently only
   `singlecell.pca`, `singlecell.umap`, `bulk.pca` (not exported at top level). This
   is the one place analysis and visualization are fused, and it's the source of the
   `plot_pca` collision in §4.2.
4. **Grammar extensions** — not "a plot", but new geoms/stats/scales for plotnine
   itself, usable standalone: `ggnomics.signif` (`geom_signif`, `run_comparisons`,
   `map_pvalue_to_stars`) and `ggnomics.upset` (a full ComplexUpset-style
   compute→select→apply pipeline with its own ~35 exports). These are already
   correctly scoped as their own subpackages/namespaces and don't need to change.
5. **Composition & palette utilities** — `hstack`/`vstack`/`grid`/
   `annotate_composition`/`save_composition`; `get_palette`/`resolve_palette` +
   palette constants. Cross-cutting, used by everything above.

Formalizing these five categories (in docs navigation, and in a short "how to name a
new function" note in `CONTRIBUTING.md`) will do most of the organizational work you're
after, independent of any renaming below.

## 6. Recommended naming convention going forward

A new public plot function should satisfy:

- **Prefix**: `plot_` for anything returning `ggplot` or `Compose`. No exceptions —
  this is already true for 100% of the modern set and should stay a hard rule.
- **Subject, not container**: the name describes *what* is plotted
  (`embedding`, `coldata`, `expression`, `volcano`), never the input type. Already
  followed everywhere in the modern set — keep enforcing it.
- **Qualifiers as suffixes, not alternate verbs**: `_stats` for automatic
  significance annotation (`plot_violin_stats`, `plot_box_stats`), `_panel`/`_marginal`
  for multi-panel composition (`plot_embedding_panel`, `plot_scatter_marginal`). This
  convention already exists and reads well — codify it explicitly so new functions
  follow it instead of inventing a new pattern (e.g. a hypothetical future
  "expression + stats" function should be `plot_expression_stats`, not
  `plot_stat_expression` or `plot_expression_with_stats`).
- **At most one familiarity alias per function**, and it must be declared as an alias,
  not shipped as an equal-weight second name. See §7.1.
- **Compute-then-plot functions get their own verb**, never `plot_*`. See §7.2.

## 7. Decisions to make

### 7.1 Collapse the embedding aliases

Recommendation: **`plot_embedding` is canonical.** It's the most general (any
`obsm`/`reducedDims` key, not just PCA/UMAP/t-SNE) and already mirrors
`scanpy.pl.embedding`, which is the convention most of your target audience already
knows.

- Drop `plot_reduced_dim` — it's a pure synonym of `plot_embedding` with no distinct
  audience; keeping it only doubles the surface a user has to learn is "the same
  thing."
- Keep `dim_plot` **only** as a documented Seurat-familiarity alias (one line in the
  docstring: "alias of `plot_embedding`, for users coming from Seurat's `DimPlot`"),
  not as an equal `__all__` entry with its own doc page.
- Keep `plot_umap`/`plot_pca`/`plot_tsne` — these are legitimate convenience wrappers
  (fixed `dimred=`), not synonyms, and match `scanpy.pl.umap`/`.pca`/`.tsne` closely
  enough to lower the adoption cost. (`plot_pca` needs the fix in §7.2 first.)

### 7.2 Split "compute" from "plot" for PCA/UMAP

The `plot_pca`/`plot_umap` name collision (§4.2) should be resolved by removing the
second definition, not by renaming around it. Two ways to get there — pick one:

**Option A — compute-only helpers, single plot entry point (recommended).**
Rename `sc_pca.plot_pca` → drop it entirely; `sc_pca.run_pca` keeps computing and
storing the embedding, and the user calls the one canonical `gg.plot_pca` /
`gg.plot_embedding` on the result afterward, exactly as they already do for any
precomputed embedding. Same for `sc_umap`/`bulk_pca`. This is a small deletion (the
`plot_pca`/`plot_umap` wrappers in `singlecell/pca.py`, `singlecell/umap.py`,
`bulk/pca.py` are ~15 lines each of pure pass-through to `plot_embedding` already) and
it makes the analysis/visualization boundary from the README literally true in code,
not just in prose. It also removes the `build_plot_df_from_sce`/
`build_plot_df_from_result` duplication for free, since there's no longer a
compute-module-local plot path to feed.

**Option B — keep the combo, rename it out of the `plot_` namespace.** e.g.
`sc_pca.compute_embedding(sce, ..., plot=True)` or a distinctly-named
`sc.pca(...)` / `bulk.pca(...)` callable that isn't spelled `plot_pca`. Lower-value
than A (keeps two ways to do the same thing) but preserves the one-call convenience
some users may like for exploratory work.

Either way, this is the highest-priority fix in this document: it's the one place two
functions with the *identical name* do observably different things depending on
import path, which is the kind of surprise that erodes trust in a "stable" API fast.

### 7.3 Retire the legacy namespace as a namespace, not just individually

Rather than leaving ~15 legacy functions permanently mixed into the root `__all__`
next to their modern replacements, move them behind `ggnomics.legacy`:

1. Now: add a `DeprecationWarning` inside each legacy function pointing at its modern
   replacement (table in §3), keep the root-level import working.
2. Next minor version: move the definitions under `ggnomics/legacy/` (or keep the
   files where they are and just re-export from `ggnomics.legacy`), stop importing
   them into `ggnomics.__all__` directly — `import ggnomics.legacy as gglegacy` still
   works, `ggnomics.volcano_plot` starts raising `AttributeError` with a message
   pointing at `ggnomics.legacy.volcano_plot`.
3. Next major version: delete.

`ridge_density` (no modern replacement, §3) and the `*_from_matrix` convenience
wrappers (§4.7) are the two exceptions — decide whether those get a modern
replacement before deprecating, or get promoted (rather than deprecated) since they
fill a real gap.

### 7.4 Fill or document the backend-coverage gaps (§4.6)

- Register `plot_scatter` for `SummarizedExperiment` (looks like a straightforward
  bug fix, not a design question).
- For `MuData`'s narrow coverage: either (a) write adapters for the rest of the
  container-dispatched set (treating a `MuData` mostly like "AnnData with a modality
  selector"), or (b) keep it scoped to cross-modality functions and add one sentence
  to the docs/README stating that scope explicitly, so it reads as a decision rather
  than an omission. Given multimodal is already flagged as a smaller, newer surface
  area, (b) is the lower-effort and probably-correct near-term choice — just make it
  explicit.
- Decide whether `plot_adt_qc`/`plot_clonotype_*` on `SummarizedExperiment` is
  intentional generality (works if the columns exist, no harm) or should be narrowed;
  either is defensible, but it should be a recorded decision.

### 7.5 Decide the `HeatmapResult` policy

Recommendation: return `HeatmapResult` from `plot_heatmap` whenever
`cluster_rows`/`cluster_cols` reorders the matrix (the reordered matrix is exactly the
derived artifact the caller can't otherwise recover), and a bare `ggplot` when no
clustering happened — matching the existing rule that `HeatmapResult` marks "this
call produced a matrix you couldn't already reconstruct."

### 7.6 Decide `ggnomics/pca/`'s visibility (§4.8)

Either rename to `ggnomics/_pca/` to make "internal plumbing" true by path (cheapest,
recommended if the sklearn/SVD PCA core isn't meant to be a public API in its own
right), or keep the path and add it to `ggnomics.__all__` with the same documentation
weight as everything else (only worth it if there's a real audience for calling
`run_pca_sklearn` directly rather than through `sc_pca`/`bulk_pca`).

## 8. Suggested sequencing

1. Fix §7.2 (the `plot_pca`/`plot_umap` collision) — highest confusion-per-line-of-fix
   ratio, and a pure subtraction (Option A deletes code, doesn't add any).
2. Fix §7.1 (drop `plot_reduced_dim`, demote `dim_plot` to a documented alias) —
   another pure subtraction from `__all__`.
3. Land §4.3 (`plot_coldata` delegates to `plot_scatter` for the numeric/numeric case)
   and §4.5 (`HeatmapResult` policy) — internal correctness fixes, no public API
   change, but they remove behavior that would otherwise need to be preserved forever
   once "stable" is declared.
4. Start the §7.3 legacy-deprecation clock (add warnings now, move the namespace in
   the next minor release) — this is the change most visible to existing users, so it
   should go out with the most advance notice.
5. §7.4/§7.5/§7.6 can land opportunistically — they're gaps and small inconsistencies,
   not landmines, so they don't block declaring the rest of the API stable.

Once 1–4 are done, the "modern" `plot_*` set (§5, category 1) plus `plot_volcano`/
`plot_ma`/`plot_coef_*`/`plot_pseudobulk_de` (category 2), `ggnomics.signif`,
`ggnomics.upset`, and the composition/palette utilities are in good shape to freeze as
`ggnomics` 1.0's public surface.
