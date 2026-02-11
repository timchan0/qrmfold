from __future__ import annotations

import os
import sys

# Add repository root to sys.path so autodoc can import qrmfold
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

project = "qrmfold"
author = "Tim Chan"

# Keep in sync with pyproject.toml where practical; Sphinx uses this for display only.
release = "0.6.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
