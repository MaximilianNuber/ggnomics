"""Deterministic, cached loaders for every real public dataset used across the
ggnomics vignettes. Each function fetches once (network on first call, then the
BiocPy loaders' own on-disk cache under `setup.DATA_CACHE` on every call after),
validates the object's class/shape/columns before returning, and is safe to call
repeatedly from multiple vignettes without re-fetching.

Nothing here computes plots — that stays in the vignettes themselves, next to the
ggnomics calls being demonstrated.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from setup import EXPERIMENTHUB_CACHE, EXPRESSIONATLAS_CACHE, PROCESSED_CACHE, SCRNASEQ_CACHE

ZEISEL_LEVEL1_CLASSES = [
    "pyramidal CA1",
    "oligodendrocytes",
    "pyramidal SS",
    "interneurons",
    "endothelial-mural",
    "astrocytes_ependymal",
    "microglia",
]


def _cache_pickle(name: str, builder):
    """Cache the (potentially expensive) result of `builder()` under
    PROCESSED_CACHE/<name>.pkl. Reused across vignette re-renders and across
    documentation-building sessions on the same machine.
    """
    path = PROCESSED_CACHE / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    result = builder()
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    return result


# ---------------------------------------------------------------------------
# Dataset A: Zeisel mouse brain (2015)
# ---------------------------------------------------------------------------


def load_zeisel_full():
    """Fetch the full Zeisel mouse brain SCE (20,006 features x 3,005 cells).

    Validated on first fetch (see single-cell-scranpy.qmd): class
    SingleCellExperiment, shape (20006, 3005), `level1class`/`level2class`
    present in column data, `repeat`/`ERCC` alternative experiments present.
    """
    import scrnaseq

    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
    assert sce.shape == (20006, 3005), f"Unexpected Zeisel shape: {sce.shape}"
    assert {"level1class", "level2class"}.issubset(sce.column_data.get_column_names())
    return sce


def run_scranpy_pipeline_on_zeisel(n_hvgs_top: int = 4000, n_pcs: int = 25):
    """Run the full scranpy QC -> normalize -> HVG -> PCA -> graph-cluster ->
    UMAP/t-SNE pipeline on the full Zeisel dataset, and cache the result.

    Returns a `SingleCellExperiment` with reduced dims `PCA`/`TSNE`/`UMAP` and
    column-data columns `sum`, `detected`, `subset_proportion.mito`,
    `size_factor`, `clusters` (graph-based cluster label) merged in alongside
    the original `level1class`/`level2class` annotations. See
    single-cell-scranpy.qmd for the annotated, step-by-step version of this
    same pipeline.
    """

    def _build():
        import scranpy

        sce = load_zeisel_full()
        row_names = sce.get_row_names()
        mito_mask = [str(g).lower().startswith("mt-") for g in row_names]
        qc = scranpy.compute_rna_qc_metrics(sce.get_assay("counts"), subsets={"mito": mito_mask})
        qc_df = qc.to_pandas()
        thresholds = scranpy.suggest_rna_qc_thresholds(qc)
        keep = scranpy.filter_rna_qc_metrics(thresholds, qc)

        filtered = sce[:, keep]
        # Merge the pre-filter QC metrics for the *retained* cells into column
        # data, so downstream plots can show real QC values, not just the
        # pass/fail decision.
        cd = filtered.column_data
        for col in ("sum", "detected", "subset_proportion.mito"):
            cd = cd.set_column(col, qc_df.loc[keep, col].to_numpy())
        filtered = filtered.set_column_data(cd)

        size_factors = qc_df.loc[keep, "sum"].to_numpy()
        normed = scranpy.normalize_rna_counts_se(filtered, assay_type="counts", size_factors=size_factors)
        hvg_out = scranpy.choose_rna_hvgs_se(normed, assay_type="logcounts", top=n_hvgs_top)
        hvg_flags = hvg_out.row_data.get_column("hvg")
        pca_out = scranpy.run_pca_se(hvg_out, features=hvg_flags, number=n_pcs, assay_type="logcounts")
        final = scranpy.run_all_neighbor_steps_se(pca_out, reddim_type="PCA")
        return final

    return _cache_pickle("zeisel_scranpy_pipeline", _build)


def load_zeisel_subset(n_per_class: int = 40, seed: int = 0):
    """Deterministic stratified subset of Zeisel, preserving `level1class`
    proportions rather than taking the first N cells (which would introduce
    plate/batch ordering bias — Zeisel's cells are ordered by 384-well plate).

    For each `level1class`, samples up to `n_per_class` cells using a fixed
    numpy Generator seeded with `seed`; classes with fewer cells than
    `n_per_class` contribute all of their cells. Returns a subsetted SCE.
    """
    sce = load_zeisel_full()
    classes = np.asarray(sce.column_data.get_column("level1class"))
    rng = np.random.default_rng(seed)
    chosen_idx = []
    for cls in ZEISEL_LEVEL1_CLASSES:
        cls_idx = np.flatnonzero(classes == cls)
        n = min(n_per_class, len(cls_idx))
        chosen_idx.append(rng.choice(cls_idx, size=n, replace=False))
    idx = np.sort(np.concatenate(chosen_idx))
    return sce[:, idx]


# ---------------------------------------------------------------------------
# Dataset B: Baron human pancreas (2016)
# ---------------------------------------------------------------------------


def load_baron():
    """Fetch the Baron human pancreas SCE. Confirmed columns: `donor` (4 GEO
    sample IDs) and `label` (14 pancreatic cell types).
    """
    import scrnaseq

    sce = scrnaseq.fetch_dataset("baron-pancreas-2016", "2023-12-14", path="human", realize_assays=True)
    assert {"donor", "label"}.issubset(sce.column_data.get_column_names())
    return sce


# ---------------------------------------------------------------------------
# Dataset C: Expression Atlas E-MTAB-1625 (rice salt stress, bulk RNA-seq)
# ---------------------------------------------------------------------------


def load_emtab1625():
    """Fetch the E-MTAB-1625 rnaseq SummarizedExperiment.

    Confirmed live: 38,866 genes x 18 samples (rice, shoot tissue only), a
    balanced 2 (growth condition) x 3 (time) x 3 (replicate) design — NOT the
    58,735 x 24 initially assumed from the accession description; the live
    object is authoritative.
    """
    from pyexpressionatlas import ExpressionAtlasClient

    atlas = ExpressionAtlasClient(cache_dir=str(EXPRESSIONATLAS_CACHE))
    exp = atlas.get_experiment("E-MTAB-1625")
    se = exp["rnaseq"]
    assert se.shape[1] == 18, f"Unexpected E-MTAB-1625 sample count: {se.shape[1]}"
    cd = se.column_data.get_column_names()
    assert {"growth condition", "time"}.issubset(cd)
    return se


def run_deseq2_on_emtab1625(timepoint: str = "24 hour"):
    """Fit PyDESeq2 on the E-MTAB-1625 salt-vs-control contrast at one matched
    timepoint (default: 24 hour, the most differentiated/chronic response;
    confirmed live to have exactly 3 replicates per group at every timepoint).

    Returns a dict with `counts` (samples x genes, raw integer counts used for
    the fit), `metadata` (sample-level design table), `results` (PyDESeq2's
    `results_df`, indexed by gene ID, with `log2FoldChange`/`padj`/etc.),
    `vst` (samples x genes, variance-stabilized - for visualization/PCA only,
    never for testing), and `gene_names` (gene ID -> `Gene Name` mapping from
    the SE's row data, for labeling plots).

    Matplotlib/pydeseq2 are imported before pyexpressionatlas deliberately:
    importing pyexpressionatlas first (transitively pulling in rds2py) before
    matplotlib has ever been imported has been observed, live, to leave
    `matplotlib.ft2font` in a broken partially-imported state that surfaces as
    an unrelated `RdsParserError` on the *next* unrelated matplotlib import in
    the same process - a real, reproducible import-order fragility, not a
    ggnomics issue. Importing matplotlib (already a plotnine dependency) first
    avoids it entirely.
    """
    import matplotlib

    matplotlib.use("Agg")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    def _build():
        se = load_emtab1625()
        cd = se.get_column_data().to_pandas().rename(columns={"growth condition": "growth_condition"})
        cd["growth_condition"] = cd["growth_condition"].map(
            {"300 millimolar sodium chloride": "salt", "normal watering": "control"}
        )
        cd["time"] = cd["time"].str.replace(" ", "_")

        sel = cd["time"] == timepoint.replace(" ", "_")
        sub_cd = cd.loc[sel]
        assert sub_cd["growth_condition"].value_counts().min() >= 2, "fewer than 2 replicates in a group"

        gene_names = se.get_row_names()
        counts_mat = se.get_assay("counts")
        counts_df = pd.DataFrame(np.asarray(counts_mat), index=gene_names, columns=se.get_column_names()).T
        counts_df = counts_df.loc[sub_cd.index].round().astype(int)

        metadata = sub_cd[["growth_condition"]].copy()
        metadata["growth_condition"] = pd.Categorical(metadata["growth_condition"], categories=["control", "salt"])

        dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~growth_condition", refit_cooks=True, quiet=True)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=["growth_condition", "salt", "control"], alpha=0.05, quiet=True)
        ds.summary()

        dds.vst_fit()
        vst = pd.DataFrame(dds.vst_transform(), index=counts_df.index, columns=counts_df.columns)

        gene_name_map = se.get_row_data().to_pandas()["Gene Name"]

        return {
            "counts": counts_df,
            "metadata": metadata,
            "results": ds.results_df,
            "vst": vst,
            "gene_names": gene_name_map,
        }

    return _cache_pickle(f"emtab1625_deseq2_{timepoint.replace(' ', '_')}", _build)


# ---------------------------------------------------------------------------
# Dataset D: PBMC CITE-seq (EH7740, or the documented EH3534-EH3538 fallback)
# ---------------------------------------------------------------------------


def load_pbmc_cite_seq() -> tuple[dict, str]:
    """Return (`{"rna": AnnData, "adt": AnnData}`, resource_note).

    Tries EH7740 first; falls back to EH3534-EH3538 if EH7740 cannot be
    parsed by the installed `experimenthub`/`rds2py` version (confirmed live:
    EH7740's RDS fails with `RdsParserError: failed to parse an S4 object's
    body` — a real parser limitation for this resource's nested int_colData,
    not a data problem). The fallback bundle is real ExperimentHub data (Mair
    et al. BD Rhapsody PBMC CITE-seq: 499 RNA features, 42 ADT features,
    29,033 cells across 3 main donors), not synthetic.
    """
    from experimenthub import ExperimentHubRegistry
    import anndata as ad

    hub = ExperimentHubRegistry(cache_dir=str(EXPERIMENTHUB_CACHE))

    try:
        obj = hub.load("EH7740")
        # If this ever succeeds (parser fixed upstream), validate before use.
        return _sce_to_rna_adt_anndata(obj), "EH7740"
    except Exception as exc:  # noqa: BLE001 - documented, deliberate fallback
        note = f"EH7740 failed to load ({type(exc).__name__}: {exc}); using EH3534-EH3538 fallback"

    rna_counts = hub.load("EH3534")  # MatrixWrapper: 499 x 29033 (features x cells)
    rna_row = hub.load("EH3535").to_pandas()  # 499 x {Symbol, RefSeq, Type}
    adt_counts = hub.load("EH3536")  # MatrixWrapper: 42 x 29033
    adt_row = hub.load("EH3537").to_pandas()  # 42 x {Symbol, Alternative, ID}
    cell_meta = hub.load("EH3538").to_pandas()  # 29033 x {Sample_Tag, Sample_Name, Cartridge}

    # Cell identifiers: matrix dimnames carry no "-1" suffix; colData rownames do.
    # Verify the correspondence explicitly rather than assuming it.
    rna_barcodes = np.asarray(rna_counts.dimnames[1])
    adt_barcodes = np.asarray(adt_counts.dimnames[1])
    # The "-1"/"-2" suffix encodes the cartridge (confirmed live: it matches
    # the `Cartridge` column exactly), so it must be stripped generically -
    # stripping only "-1" would silently misalign every cartridge-2 cell.
    meta_barcodes = cell_meta.index.str.replace(r"-\d+$", "", regex=True).to_numpy()
    assert np.array_equal(rna_barcodes, adt_barcodes), "RNA/ADT matrix cell order mismatch"
    assert np.array_equal(rna_barcodes, meta_barcodes), "RNA matrix / cell metadata order mismatch"

    # Keep the original cell_meta.index (barcode + cartridge suffix) as the cell
    # identifier - the bare numeric barcode alone is reused across cartridges
    # (confirmed live: 271 collisions out of 29,033) so it is not unique on its own.
    obs = cell_meta.rename(
        columns={"Sample_Tag": "sample_tag", "Sample_Name": "donor", "Cartridge": "cartridge"}
    ).drop(columns=["rownames"], errors="ignore")
    # Drop multiplets/undetermined droplets - not real single cells.
    keep = ~obs["donor"].isin(["Multiplet", "Undetermined"])
    keep_np = keep.to_numpy()
    obs = obs.loc[keep]

    rna_x = np.asarray(rna_counts.matrix).T[keep_np]
    adt_x = np.asarray(adt_counts.matrix).T[keep_np]

    rna_var = rna_row.set_index("Symbol")
    adt_var = adt_row.set_index("Symbol")

    rna_adata = ad.AnnData(X=rna_x, obs=obs.copy(), var=rna_var)
    adt_adata = ad.AnnData(X=adt_x, obs=obs.copy(), var=adt_var)
    rna_adata.var_names_make_unique()
    adt_adata.var_names_make_unique()
    return {"rna": rna_adata, "adt": adt_adata}, note


def run_scranpy_pipeline_on_cite_seq_rna():
    """Minimal QC -> normalize -> PCA -> UMAP pass on the CITE-seq RNA modality
    (all 499 targeted-panel genes used directly - no HVG selection needed at
    this panel size), so multimodal.qmd has a real embedding to color by RNA
    and protein abundance. Returns the final `SingleCellExperiment`.
    """

    def _build():
        import scranpy
        from singlecellexperiment import SingleCellExperiment

        mods, _note = load_pbmc_cite_seq()
        rna = mods["rna"]
        sce = SingleCellExperiment(assays={"counts": rna.X.T}, row_data=rna.var.copy(), column_data=rna.obs.copy())

        qc = scranpy.compute_rna_qc_metrics(sce.get_assay("counts"), subsets={})
        qc_df = qc.to_pandas()
        thresholds = scranpy.suggest_rna_qc_thresholds(qc)
        keep = scranpy.filter_rna_qc_metrics(thresholds, qc)

        filtered = sce[:, keep]
        size_factors = qc_df.loc[keep, "sum"].to_numpy()
        normed = scranpy.normalize_rna_counts_se(filtered, assay_type="counts", size_factors=size_factors)
        pca_out = scranpy.run_pca_se(normed, features=[True] * normed.shape[0], number=20, assay_type="logcounts")
        final = scranpy.run_all_neighbor_steps_se(pca_out, reddim_type="PCA")
        return final

    return _cache_pickle("cite_seq_rna_scranpy_pipeline", _build)


def _sce_to_rna_adt_anndata(sce):  # pragma: no cover - only exercised if EH7740 ever parses
    raise NotImplementedError(
        "EH7740 parsed successfully in this environment; RNA/ADT extraction from its "
        "altExp structure was not implemented because live inspection during this "
        "documentation pass always hit the rds2py parser error documented above."
    )


# ---------------------------------------------------------------------------
# Dataset E: ECCITE-seq TCR repertoire (EH4616 CTCL, EH4621 control)
# ---------------------------------------------------------------------------


def load_tcr_repertoire() -> pd.DataFrame:
    """Load and combine the CTCL (EH4616) and control (EH4621) TCR contig
    tables into one long DataFrame with a `condition` column and a globally
    unique `clonotype_id`.

    Both tables already carry a native `raw_clonotype_id` (10x Cell Ranger
    VDJ convention) but IDs are only unique *within* each 10x run - both
    files reuse labels like "clonotype1". `clonotype_id` is therefore
    deterministically derived as `f"{condition}_{raw_clonotype_id}"`.
    """
    from experimenthub import ExperimentHubRegistry

    hub = ExperimentHubRegistry(cache_dir=str(EXPERIMENTHUB_CACHE))

    frames = []
    for accession, condition in [("EH4616", "CTCL"), ("EH4621", "control")]:
        obj = hub.load(accession)
        key = list(obj.keys())[0]
        df = obj[key].to_pandas().drop(columns=["rownames"])
        df["condition"] = condition
        df["clonotype_id"] = condition + "_" + df["raw_clonotype_id"].astype(str)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    assert {"barcode", "chain", "cdr3", "v_gene", "j_gene", "productive"}.issubset(combined.columns)
    return combined
