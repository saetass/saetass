import os
import shutil
import sys
from importlib.metadata import version as package_version
from pathlib import Path

sys.path.insert(0, os.path.abspath(r"../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SAETASS"
copyright = "2026, José María García Morillo"
author = "José María García Morillo"
# The version of the installed package, which hatch-vcs takes from the latest git tag
release = package_version("saetass")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "nbsphinx",
]

# The tutorials are committed with their outputs (see tutorials/execute.py). Executing
# them here could publish warnings and errors with paths of the machine that builds
nbsphinx_execute = "never"

# Copied instead of linked: Sphinx resolves symbolic links before the relative paths of
# images, so it would not find the figures of linked notebooks
_docs = Path(__file__).parent
_copies = _docs / "tutorials" / "tutorials_folder"
shutil.rmtree(_copies, ignore_errors=True)
_copies.mkdir()
for _notebook in (_docs.parents[1] / "tutorials").glob("*.ipynb"):
    shutil.copy2(_notebook, _copies)

bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_favicon = r"_static/saetass_favicon.ico"
html_logo = r"_static/saetass_logotagline_horizontal.svg"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/saetass/saetass",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/saetass/",
            "icon": "fab fa-python",
        },
    ],
    "footer_end": ["changelog-link", "theme-version"],
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

add_module_names = False


def setup(app):
    app.add_css_file("custom.css")
