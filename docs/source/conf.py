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

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "sid"
copyright = "2020, Janos Gabler, Tobias Raabe, Klara Röhrl"  # noqa: A001
author = "Janos Gabler, Tobias Raabe, Klara Röhrl"
release = "0.0.1"


# -- General configuration ---------------------------------------------------

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
    "nbsphinx",
    "sphinx_copybutton",
    "autoapi.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]


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
]

extlinks = {
    "ghuser": ("https://github.com/%s", "@"),
    "gh": ("https://github.com/covid-19-impact-lab/sid/pulls/%s", "#"),
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.8", None),
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
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}
{% set github = 'https://github.com/covid-19-impact-lab/sid' %}

.. only:: html

    .. nbinfo::

        View and download the notebook `here
        <{{ github }}/tree/v{{ env.config.release }}/docs/source/{{ docname }}>`_!

"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "alabaster"

html_theme_options = {
    "extra_nav_links": {"On Github": "https://github.com/covid-19-impact-lab/sid"},
    "logo_name": True,
    "description": "A simulator for infectious diseases",
    "github_button": False,
    "github_user": "covid-19-impact-lab",
    "github_repo": "sid",
    "font_family": '"Avenir Next", Calibri, "PT Sans", sans-serif',
    "head_font_family": '"Avenir Next", Calibri, "PT Sans", sans-serif',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]
