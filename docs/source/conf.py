# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.abspath("../../src"))


import sid  # noqa: E402


# -- Project information -----------------------------------------------------

project = "sid"
year = datetime.now().year
copyright = f"2020-{year}, Janos Gabler, Tobias Raabe, Klara Röhrl"  # noqa: A001
author = "Janos Gabler, Tobias Raabe, Klara Röhrl"
version = sid.__version__
release = version

# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_panels",
    "autoapi.extension",
]

# List of patterns, relative to source directory, that match files and directories to
# ignore when looking for source files. This pattern also affects html_static_path and
# html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = "default"
pygments_dark_style = "monokai"


# -- Package configuration ---------------------------------------------------

# Configuration for autodoc.
autodoc_member_order = "bysource"
autosummary_generate = True
add_module_names = False

autodoc_mock_imports = [
    "bokeh",
    "dask",
    "numba",
    "numpy",
    "pandas",
    "tqdm",
]

extlinks = {
    "ghuser": ("https://github.com/%s", "@"),
    "gh": ("https://github.com/covid-19-impact-lab/sid/pulls/%s", "#"),
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.8", None),
    "dask": ("https://docs.dask.org/en/latest", None),
}

# Configuration for autoapi.
autoapi_type = "python"
autoapi_dirs = ["../../src"]
autoapi_keep_files = False
autoapi_add_toctree_entry = False

# Remove prefixed $ for bash, >>> for Python prompts, and In [1]: for IPython prompts.
copybutton_prompt_text = r"\$ |>>> |In \[\d\]: "
copybutton_prompt_is_regexp = True


# Configuration for nbsphinx.
nbsphinx_execute = "never"


# -- Options for HTML output -----------------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for a list of
# built-in themes.
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here, relative
# to this directory. They are copied after the built-in static files, so a file named
# "default.css" will overwrite the built-in "default.css".
html_css_files = ["css/custom.css"]

# The name of an image file (within the static path) to use as favicon of the docs.
# This file should be a Windows icon file (.ico) being 16x16 or 32x32 pixels large.
# html_logo = "_static/images/pytask_w_text.svg"  # noqa: E800

# The name of an image file (within the static path) to use as favicon of the docs.
# This file should be a Windows icon file (.ico) being 16x16 or 32x32 pixels large.
# html_favicon = "_static/images/pytask.ico"  # noqa: E800

# Add any paths that contain custom static files (such as style sheets) here, relative
# to this directory. They are copied after the builtin static files, so a file named
# "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]
