#!/usr/bin/env python3
"""Validate internal links in a rendered Quarto site.

Walks ``<site>/**/*.html`` and checks every ``href``/``src`` that points at
something inside the site. External links (``http://``, ``https://``,
``mailto:``, ``//cdn...``), ``javascript:`` handlers, and ``data:`` URIs are
ignored so the check is deterministic and works offline.

Fragment-only links (``#section``) and fragments on internal targets are
verified against the ``id``/``name`` attributes of the target page when that
page is HTML.

Usage::

    python tools/check_internal_links.py vignettes/_site

Exits non-zero if any internal link cannot be resolved.
"""

from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Attributes that can carry a URL we care about.
_URL_ATTRS = {"href", "src"}

# Schemes that are never checked locally.
_SKIP_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "javascript:",
    "data:",
    "tel:",
    "ftp:",
    "//",
)

# Bootstrap/Quarto emit these as toggles rather than navigation targets.
_SKIP_EXACT = {"#", ""}


class _LinkParser(HTMLParser):
    """Collect link targets and anchor ids from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, int]] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        mapping = {key: value for key, value in attrs if value is not None}

        for key in ("id", "name"):
            if key in mapping:
                self.anchors.add(mapping[key])

        # data-bs-target etc. are UI state, not navigation.
        if "data-bs-toggle" in mapping or "data-toggle" in mapping:
            return

        for attr in _URL_ATTRS:
            if attr in mapping:
                self.links.append((mapping[attr], self.getpos()[0]))


def _parse(path: Path) -> _LinkParser:
    parser = _LinkParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser


def _resolve(site: Path, page: Path, target: str) -> Path:
    """Resolve an internal link relative to the page that contains it."""
    if target.startswith("/"):
        return site / target.lstrip("/")
    return (page.parent / target).resolve()


def check_site(site: Path) -> list[str]:
    """Return a list of human-readable problems found in ``site``."""
    site = site.resolve()
    pages = sorted(site.rglob("*.html"))
    if not pages:
        return [f"{site}: no HTML files found - was the site rendered?"]

    anchors_by_page: dict[Path, set[str]] = {}
    problems: list[str] = []

    parsed = {page: _parse(page) for page in pages}
    for page, result in parsed.items():
        anchors_by_page[page] = result.anchors

    for page, result in parsed.items():
        rel_page = page.relative_to(site)
        for raw, line in result.links:
            target = raw.strip()
            if target in _SKIP_EXACT or target.lower().startswith(_SKIP_PREFIXES):
                continue

            split = urlsplit(target)
            path_part = unquote(split.path)
            fragment = split.fragment

            if not path_part:
                # Fragment-only link: check against this page's own anchors.
                if fragment and fragment not in anchors_by_page[page]:
                    problems.append(f"{rel_page}:{line}: fragment '#{fragment}' has no matching id")
                continue

            resolved = _resolve(site, page, path_part)

            if resolved.is_dir():
                resolved = resolved / "index.html"

            if not resolved.exists():
                problems.append(f"{rel_page}:{line}: broken link -> {raw}")
                continue

            try:
                resolved.relative_to(site)
            except ValueError:
                problems.append(f"{rel_page}:{line}: link escapes the site root -> {raw}")
                continue

            if fragment and resolved.suffix == ".html":
                known = anchors_by_page.get(resolved)
                if known is None:
                    known = _parse(resolved).anchors
                    anchors_by_page[resolved] = known
                if fragment not in known:
                    problems.append(
                        f"{rel_page}:{line}: '{raw}' resolves, but "
                        f"'#{fragment}' is missing from {resolved.relative_to(site)}"
                    )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site",
        nargs="?",
        default="vignettes/_site",
        type=Path,
        help="Path to the rendered site directory (default: vignettes/_site)",
    )
    args = parser.parse_args(argv)

    if not args.site.exists():
        print(f"error: {args.site} does not exist; render the site first", file=sys.stderr)
        return 2

    problems = check_site(args.site)
    if problems:
        print(f"Found {len(problems)} broken internal link(s):\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"OK: all internal links in {args.site} resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
