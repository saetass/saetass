import os
import sys

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
release = "0.1.0"

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
    "footer_end": ["changelog-link", "theme-version"],
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

add_module_names = False


def setup(app):
    app.add_css_file("custom.css")
