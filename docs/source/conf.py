from __future__ import annotations

import os
import sys
from pathlib import Path
import tomllib

# Add repository root to sys.path so autodoc can import qrmfold
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

project = "qrmfold"
author = "Tim Chan"

def _project_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text("utf-8"))
    return data["project"]["version"]


# Keep in sync with pyproject.toml; Sphinx uses this for display only.
release = _project_version()
version = release

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
