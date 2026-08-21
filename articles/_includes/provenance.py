"""Render the human-readable data-provenance section from datasets.yml.

Used by index.qmd. Kept dependency-light (pyyaml only) so it can run even in a
minimal environment.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DATASETS_YML = Path(__file__).resolve().parent.parent / "datasets.yml"


def load_datasets_manifest() -> list[dict]:
    with open(_DATASETS_YML) as fh:
        manifest = yaml.safe_load(fh)
    return manifest["datasets"]


def provenance_markdown() -> str:
    """Build a compact Markdown table: dataset, provider, accession/version, organism,
    used in which vignettes. Intended to be displayed via IPython.display.Markdown.
    """
    rows = load_datasets_manifest()
    lines = [
        "| Dataset | Provider | Accession | Version | Organism | Used in |",
        "|---|---|---|---|---|---|",
    ]
    for d in rows:
        used_in = ", ".join(d.get("used_in", []))
        version = d.get("version") or "—"
        lines.append(
            f"| {d['display_name']} | {d['provider']} | `{d['accession']}` | "
            f"{version} | {d.get('organism', '—')} | {used_in} |"
        )
    return "\n".join(lines)
