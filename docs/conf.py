# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from respredai import __version__

project = "ResPredAI"
copyright = "2025, Ettore Rocchi"
author = "Ettore Rocchi"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "assets/logo_ResPredAI.png"
html_favicon = "assets/logo_ResPredAI.png"

html_theme_options = {
    "logo": {
        "text": "ResPredAI",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/EttoreRocchi/ResPredAI",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
}

# -- Options for source suffix -----------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
