import os
import sys
from importlib.metadata import version as package_version

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

nbsphinx_allow_errors = True

bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]
exclude_patterns = []

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
# The surrogates need PyTorch, an optional dependency
autodoc_mock_imports = ["torch"]

add_module_names = False


def setup(app):
    app.add_css_file("custom.css")
