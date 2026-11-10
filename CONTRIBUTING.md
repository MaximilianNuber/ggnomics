# Contributing to ggnomics

Thanks for your interest in ggnomics. This document covers the development
setup and the checks that run in CI.

## Development environment

ggnomics targets Python 3.10–3.13. The workflow articles additionally need
analysis and data-access packages when their code is re-executed.

### Testing and linting only (pip)

```bash
git clone https://github.com/MaximilianNuber/ggnomics
cd ggnomics

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

The `dev` extra pulls in `all` and `test`, plus Ruff, pre-commit, tox, build,
and twine. Documentation dependencies are installed separately with `docs`.

### Documentation tools

Use Python 3.11 or newer for the documentation tools (Python 3.12 is used in
`environment-docs.yml`). Install [Quarto](https://quarto.org/docs/get-started/)
and make sure `quarto` is on `PATH`. From the repository root, install the
Python dependencies:

```bash
python -m pip install -e ".[all,docs]"
quarto --version
```

The `docs` extra installs `great-docs`, the site generator used by
`.github/workflows/docs.yml`. See [Documentation](#documentation) below for
building the site and re-executing the public-data articles.

## Running the checks

### Tests

```bash
pytest                                   # full suite
pytest tests/test_upset_plot.py -v       # one module
pytest --cov --cov-report=term-missing   # with coverage
```

`MPLBACKEND=Agg` is set by `tests/conftest.py`, so no display is needed.

### tox

tox is the common entry point and mirrors what CI runs:

```bash
tox              # py, core, lint, format, build
tox -e py        # full pytest suite with coverage
tox -e core      # core-only install; proves the backends stay lazy
tox -e lint      # ruff check
tox -e format    # ruff format --check (read-only)
tox -e build     # python -m build + twine check
tox -e docs      # great-docs build, as in the documentation CI workflow
tox -e linkcheck # internal-link validation of great-docs/_site
```

The documentation environments are deliberately excluded from the default
envlist. Build the site before running `linkcheck`. Re-executing the workflow
articles requires their analysis dependencies and network access on first use.

### Ruff

```bash
ruff check .          # lint
ruff check . --fix    # lint and apply safe fixes
ruff format .         # format
ruff format --check . # verify formatting without writing
```

Configuration lives in `[tool.ruff]` in `pyproject.toml`. The line length is
120, matching the BiocSetup scaffold default. `Optional[...]` and `List[...]`
are kept deliberately in public signatures (the `UP006`/`UP007`/`UP035`/`UP045`
rules are disabled) so that the annotations in the generated reference match what
Python 3.10 users see.

### pre-commit

```bash
pip install pre-commit
pre-commit install       # install the git hook
pre-commit run --all-files
```

Generated site output (`great-docs/`) and the committed execution cache
(`_freeze/`) are excluded from the hooks. Edit the article sources under
`articles/`, not generated reference pages or cached outputs.

### Coverage

The full suite currently covers about **83%** of `ggnomics/`. `tox -e py`
enforces a floor of **80%**, chosen to leave a little headroom without letting
coverage silently erode. Raise the floor when coverage improves; please do not
lower it to make a change pass.

## Documentation

The published site uses [great-docs](https://posit-dev.github.io/great-docs/),
configured by `great-docs.yml`. Workflow sources live in `articles/*.qmd`;
shared data/setup helpers live in `articles/_includes/`. API pages are generated
from package docstrings. The generated project and HTML are under `great-docs/`
and `great-docs/_site/`, respectively, and are gitignored.

### Building the site

After installing the documentation tools above, run from the repository root:

```bash
great-docs build
# or use an isolated tox environment:
tox -e docs
```

These use the same site generator and output location as
`.github/workflows/docs.yml`. The retired Sphinx/quartodoc site, `vignettes/`,
and `tools/build_tutorial_notebooks.py` are no longer build entry points.
Do not edit files in `great-docs/`; the next build regenerates them.

### Re-executing workflow articles

`great-docs.yml` sets `freeze.mode: auto`, and the committed Quarto cache
lives under `_freeze/articles/`. Unchanged pages can reuse cached outputs;
changed code may trigger execution during a build. A cached page is not a
fresh execution of its analysis. After changing article code, explicitly
re-execute the affected article and review its results before updating its
cache; do not treat a cached HTML build as proof that changed code ran. For
example, after installing the analysis dependencies below:

```bash
great-docs freeze articles/bulk-rnaseq.qmd
great-docs build
```

Review and commit the refreshed `_freeze/articles/bulk-rnaseq/` outputs along
with the source change. `great-docs freeze --info` shows the cache status.

The analysis/data-access dependencies are additional to `ggnomics[all,docs]`:

```bash
python -m pip install scranpy scrnaseq pyexpressionatlas experimenthub pydeseq2 marsilea
```

Alternatively, `environment-docs.yml` provides a Python 3.12 conda environment
with the documentation and analysis dependencies. Install ggnomics itself
editable after creating it:

```bash
mamba env create -f environment-docs.yml
mamba run -n ggnomics-docs python -m pip install -e ".[all,docs]"
```

The helper imports at the start of each article resolve `_includes` relative
to the article directory. If copying cells into a notebook or Python session,
start it from `articles/`, or explicitly add `articles/_includes` to `sys.path`.
Initialize matplotlib before importing the dataset loaders, as shown in the
articles. Their first run fetches public data; `GGNOMICS_DATA_CACHE` overrides
the cache root (default: `~/.cache/ggnomics-docs`). Download/proxy failures must
be distinguished from plotting errors, and cached output must not be reported
as a successful fresh run.

### Checking links

After building the site:

```bash
python tools/check_internal_links.py great-docs/_site
# or:
tox -e linkcheck
```

The checker verifies local links and HTML fragments. External HTTP links are
skipped so this check is deterministic and works offline. A missing site or a
site containing no HTML files is an error, not a successful check.

## Distributions

```bash
python -m build
python -m twine check dist/*
```

or `tox -e build`, which cleans `dist/` first.

## Versioning and releases

The version is **not** written anywhere in the source tree. `setuptools_scm`
derives it from the git tag, so the tag is the single source of truth:

| State | Version |
|---|---|
| clean tree, exactly at `v0.1.0` | `0.1.0` |
| one commit later | `0.1.0.post1.dev1+g84d03a2d` |
| dirty working tree | ...`+g84d03a2d.d20260818` |
| no git metadata (unpacked sdist) | `fallback_version`, currently `0.1.0` |

At build time setuptools_scm writes `ggnomics/_version.py`. That file is
generated and gitignored — never edit or commit it. At runtime,
`ggnomics.__version__` reads it, falling back to the installed distribution
metadata when running from a source checkout.

Check what the current tree would produce:

```bash
python -m setuptools_scm
python -c "import ggnomics; print(ggnomics.__version__)"
```

### Cutting a release

```bash
# 1. Land everything, update the CHANGELOG's Unreleased section, commit.
# 2. Tag. It must be annotated and the tree must be clean.
git tag -a v0.2.0 -m "ggnomics 0.2.0"

# 3. Confirm the version resolves cleanly - no "dev", no "+" local part.
python -m setuptools_scm

# 4. Push the tag. This triggers .github/workflows/publish-pypi.yml.
git push origin v0.2.0
```

The publish workflow refuses to upload anything containing `dev` or a local
version segment, so a shallow checkout, an unannotated tag, or a dirty tree
fails the build rather than shipping a broken version to PyPI.

Anything that resolves the version needs the full history. The GitHub Actions
workflows use `fetch-depth: 0`; use a full checkout for local release builds too.

## The optional-backend architecture

This is the part of the codebase most worth understanding before changing it.

**Public plotting functions use `functools.singledispatch`, not an accessor
class.** `plot_umap`, `plot_expression`, and friends are generic functions
whose `pandas.DataFrame` implementation carries the public docstring. Adapters
for AnnData, `SingleCellExperiment`, `SummarizedExperiment`, and MuData live in
`ggnomics/_backends/` and register themselves onto those generics.

**Optional containers must never be imported eagerly.** The core install is
`numpy`, `pandas`, and `plotnine` — nothing else. `import ggnomics` must
succeed, and all DataFrame plotting plus `ggnomics.upset` must work, with none
of anndata, mudata, singlecellexperiment, or summarizedexperiment installed.

`ggnomics/_backends/__init__.py` enforces this: it uses
`importlib.util.find_spec` to check whether each optional package is
importable, and imports the corresponding backend module only if it is. A
backend that is installed but broken is allowed to raise, so a broken
installation is not silently mistaken for an absent one.

When adding a backend or a new dispatch target:

- put the container import **inside** the backend module, never at the top of a
  core module;
- add the package to `_BACKENDS` in `ggnomics/_backends/__init__.py`;
- add the extra to `[project.optional-dependencies]` in `pyproject.toml` —
  never to `dependencies`;
- extend `tests/test_backends.py`, which asserts that each backend registers
  only when its package is present.

`tests/test_backends.py` and `tests/test_public_api_contract.py` are the
guard rails for all of this. Both run on every pull request.

## Design conventions

ggnomics follows the [BiocPy developer guide](https://github.com/BiocPy/developer_guide):

- public functions are `snake_case` and carry type hints;
- plotting functions **do not mutate** the container they are given —
  `tests/test_public_api_contract.py` asserts this for DataFrame, AnnData,
  `SingleCellExperiment`, and `SummarizedExperiment`;
- BiocPy containers are features × samples (or features × cells); extraction
  transposes into tidy one-row-per-observation frames;
- `SingleCellExperiment` reduced dimensions are supported; a plain
  `SummarizedExperiment` has none, and embedding plots on one must raise an
  error that says so;
- the legacy API (`expression_violin`, `volcano_plot`, `heatmap_from_matrix`, …)
  is retained for backwards compatibility and should not be removed without a
  deprecation cycle.

## Pull requests

1. Branch from `main`.
2. Make the change and add tests.
3. Run `pre-commit run --all-files` and `tox`.
4. Add an entry to the `Unreleased` section of `CHANGELOG.md`.
5. Open a pull request describing the change and why it is needed.
