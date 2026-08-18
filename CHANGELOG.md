# Changelog

All notable changes to ggnomics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Development tooling: `tox.ini` (`py`, `core`, `lint`, `format`, `build`,
  `docs`, `linkcheck`), `.pre-commit-config.yaml`, and Ruff lint/format
  configuration in `pyproject.toml`.
- `tools/check_internal_links.py`, an offline internal-link checker for the
  rendered Quarto site.
- `tests/test_public_api_contract.py`: return-type contracts, non-mutation
  guarantees for DataFrame / AnnData / `SingleCellExperiment` /
  `SummarizedExperiment` inputs, and error-message checks for unsupported
  containers.
- Project documents: `CONTRIBUTING.md`, `CHANGELOG.md`, `AUTHORS.md`,
  `CODE_OF_CONDUCT.md`, plus pull-request and issue templates.
- GitHub Actions workflows for documentation validation and for PyPI trusted
  publishing (the trusted publisher itself is not yet configured).
- Automatic versioning with `setuptools_scm`: the version is derived from the
  git tag, written to the generated `ggnomics/_version.py` at build time, and
  exposed as `ggnomics.__version__`. `MANIFEST.in` keeps the sdist to the
  package plus its test suite, since setuptools_scm's file finder would
  otherwise ship the whole documentation tree.

### Changed

- The "UpSet plots" vignette now appears in the site navigation bar and is
  linked from the documentation home page.
- `pyproject.toml` metadata completed: real author and maintainer information,
  classifiers, keywords, and the full set of project URLs. The `docs` extra now
  installs quartodoc instead of MkDocs.
- Coverage configuration added; the full suite covers about 83% of `ggnomics/`
  and CI enforces a floor of 80%.
- The test workflow matrix now covers Python 3.10, 3.12, and 3.13.
- `pyproject.toml` no longer carries a static `version`; it is `dynamic`.

### Removed

- The legacy MkDocs site (`mkdocs.yml`, `docs/index.md`, `docs/vignettes.md`,
  `docs/api/*.md`). Quarto with quartodoc is the single documentation renderer.
  `docs/img/` and `docs/generate_readme_plots.py` are unaffected.

### Fixed

- Two bare `except:` clauses in `ggnomics/singlecell/` narrowed to
  `except Exception:` so `KeyboardInterrupt` and `SystemExit` propagate.
- `run_umap` now chains the original `ImportError` when `umap-learn` is missing.
- The "many groups" warning in `ggnomics.signif` now reports the caller's
  location via `stacklevel=2`.
- Removed dead assignments in `ggnomics/dotplot.py` and in several tests; one
  test asserted on a variable that was never bound.

## [0.1.0]

Initial release.

- `plot_*` plotting functions built on `functools.singledispatch`, returning
  ordinary plotnine objects.
- Backends for AnnData, BiocPy `SingleCellExperiment` and
  `SummarizedExperiment`, and MuData, registered lazily and only when the
  corresponding package is installed.
- Native plotnine UpSet plots and Venn diagrams (`ggnomics.upset`).
- Significance brackets and statistical annotations (`ggnomics.signif`).
- Marsilea-backed composable heatmaps.
- Quarto workflow guides and a quartodoc-generated API reference.

[Unreleased]: https://github.com/MaximilianNuber/ggnomics/compare/main...HEAD
