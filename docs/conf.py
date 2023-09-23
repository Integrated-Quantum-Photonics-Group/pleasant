"""Sphinx configuration."""
project = "pleasant"
author = "Kilian Unterguggenberger"
copyright = f"2023, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
]
html_theme = "alabaster"
html_theme_options = {"sidebar_width": "30%"}
