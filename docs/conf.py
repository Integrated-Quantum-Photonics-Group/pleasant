"""Sphinx configuration."""
project = "PLEasant"
author = "Kilian Unterguggenberger"
copyright = f"2023, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
]
html_theme = "alabaster"
html_theme_options = {
    "description": "Routines for post-processing and analyzing PLE experiment data.",
    "sidebar_width": "30%",
}
