"""Cross-cutting edge-case tests: indexes, duplicates, missing layers, tiny inputs."""

import numpy as np
import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import (
    plot_coldata,
    plot_expression,
    plot_heatmap,
    plot_highest_exprs,
    plot_scatter,
)


# ---------------------------------------------------------------------------
# Non-default / shuffled indexes
# ---------------------------------------------------------------------------


def test_dataframe_non_default_index_preserved():
    df = pd.DataFrame(
        {"g1": [1.0, 2.0, 3.0], "cluster": ["A", "B", "A"]},
        index=["z9", "a1", "m5"],
    )
    p = plot_expression(df, features=["g1"], group_by="cluster")
    assert isinstance(p, ggplot_class)
    np.testing.assert_allclose(sorted(p.data["expression"]), [1.0, 2.0, 3.0])


def test_anndata_obs_order_matches_x_even_when_obs_names_unsorted():
    anndata = pytest.importorskip("anndata")

    rng = np.random.default_rng(0)
    n = 10
    X = rng.normal(size=(n, 2)).astype(np.float32)
    ad = anndata.AnnData(X=X.copy())
    ad.var_names = ["g1", "g2"]
    # Deliberately non-alphabetical, non-sequential obs_names.
    ad.obs_names = [f"cell{i}" for i in [9, 3, 7, 0, 5, 1, 8, 2, 6, 4]]
    ad.obs["cluster"] = rng.choice(["A", "B"], n)

    p = plot_expression(ad, features=["g1"], group_by="cluster")
    long_expr = p.data.sort_values("expression")["expression"].to_numpy()
    np.testing.assert_allclose(sorted(long_expr), sorted(X[:, 0]))


# ---------------------------------------------------------------------------
# Duplicate / collision names
# ---------------------------------------------------------------------------


def test_expression_rejects_feature_metadata_collision():
    df = pd.DataFrame({"g1": [1.0, 2.0], "cluster": ["A", "B"]})
    with pytest.raises(ValueError, match="ambiguous"):
        plot_expression(df, features=["g1"], group_by="g1")


def test_heatmap_rejects_duplicate_features():
    df = pd.DataFrame({"g1": [1.0, 2.0, 3.0], "g2": [3.0, 2.0, 1.0]})
    with pytest.raises(ValueError, match="duplicate"):
        plot_heatmap(df, features=["g1", "g1", "g2"])


# ---------------------------------------------------------------------------
# Missing layers / assays
# ---------------------------------------------------------------------------


def test_expression_missing_layer_raises_informative_error(mock_adata):
    with pytest.raises(KeyError, match="Layer"):
        plot_expression(
            mock_adata, features=["Gene0001"], group_by="cluster", layer="nonexistent_layer"
        )


def test_expression_missing_assay_raises_informative_error(mock_sce):
    with pytest.raises(KeyError):
        plot_expression(
            mock_sce, features=["Gene0001"], group_by="cluster", layer="nonexistent_assay"
        )


# ---------------------------------------------------------------------------
# Tiny inputs
# ---------------------------------------------------------------------------


def test_coldata_single_observation():
    df = pd.DataFrame({"g1": [5.0], "cluster": ["A"]})
    p = plot_coldata(df, x="cluster", y="g1", shape="point")
    assert isinstance(p, ggplot_class)
    p.draw()


def test_scatter_single_observation():
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    p = plot_scatter(df, x="x", y="y")
    assert isinstance(p, ggplot_class)
    p.draw()


def test_highest_exprs_single_observation():
    df = pd.DataFrame({"Gene0001": [1.0], "Gene0002": [2.0]})
    p = plot_highest_exprs(df, n=2, features=["Gene0001", "Gene0002"])
    assert isinstance(p, ggplot_class)
    p.draw()


def test_coldata_single_group():
    df = pd.DataFrame({"g1": [1.0, 2.0, 3.0], "cluster": ["A", "A", "A"]})
    p = plot_coldata(df, x="cluster", y="g1", shape="box")
    assert isinstance(p, ggplot_class)
    p.draw()
