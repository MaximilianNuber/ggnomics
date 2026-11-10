"""Regression checks for the GreatDocs documentation workflow."""

import configparser
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK_LINKS = ROOT / "tools" / "check_internal_links.py"
# MANIFEST.in intentionally omits documentation tooling/sources from the sdist.
pytestmark = pytest.mark.skipif(
    not CHECK_LINKS.is_file() or not (ROOT / "tox.ini").is_file(),
    reason="documentation workflow checks require a source checkout",
)


def test_docs_tox_uses_current_site_generator():
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    assert config["testenv:docs"]["commands"].strip() == "great-docs build {posargs}"
    assert set(config["testenv:docs"]["extras"].split()) == {"all", "docs"}
    assert "great-docs/_site" in config["testenv:linkcheck"]["commands"]
    assert not {"testenv:notebooks", "testenv:quartosite", "testenv:doclinks"}.intersection(config.sections())


def test_linkcheck_defaults_to_great_docs_site(tmp_path):
    site = tmp_path / "great-docs" / "_site"
    site.mkdir(parents=True)
    (site / "index.html").write_text('<a href="article.html#example">Example</a>', encoding="utf-8")
    (site / "article.html").write_text('<h1 id="example">Example</h1>', encoding="utf-8")
    result = subprocess.run([sys.executable, str(CHECK_LINKS)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "great-docs/_site" in result.stdout


def test_linkcheck_rejects_missing_site(tmp_path):
    result = subprocess.run([sys.executable, str(CHECK_LINKS)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "great-docs/_site does not exist" in result.stderr


def test_linkcheck_rejects_missing_fragment(tmp_path):
    site = tmp_path / "great-docs" / "_site"
    site.mkdir(parents=True)
    (site / "index.html").write_text('<a href="#missing">Broken</a>', encoding="utf-8")
    result = subprocess.run([sys.executable, str(CHECK_LINKS)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing" in result.stderr
