"""Configuration for the FOMO-Z documentation."""

project = "FOMO-Z"
author = "Giorgio Francesco Lesci"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

html_theme = "furo"

autosummary_generate = True

# Parse NumPy-style docstrings.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Include constructor documentation in class descriptions.
autoclass_content = "both"

autodoc_member_order = "bysource"
autodoc_typehints = "description"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
