from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import warnings

# Keep docs output focused on content by silencing third-party deprecation noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"torch\.jit\._script",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"matplotlib\._fontconfig_pattern",
)
try:
    from pyparsing import PyparsingDeprecationWarning

    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "OPTIMEO"
author = "Colin Bousige"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_class_signature = "mixed"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

nbsphinx_execute = "always"
nbsphinx_timeout = 1800
nbsphinx_allow_errors = False
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', 'svg'}",
    "--InlineBackend.rc={'figure.dpi': 110}",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js",
    "https://cdn.plot.ly/plotly-2.35.2.min.js",
]
html_logo = "_static/logo.png"
html_title = "OPTIMEO documentation"
html_theme_options = {
    "logo": {
        "text": "OPTIMEO",
    },
    "github_url": "https://github.com/colinbousige/OPTIMEO",
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "secondary_sidebar_items": ["page-toc"],
}
