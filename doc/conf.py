# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import beam_networks

project = 'beam_networks'
copyright = '2025, Hannes Holey'
author = 'Hannes Holey'
version = beam_networks.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'myst_nb']

myst_enable_extensions = ['dollarmath', 'amsmath']

# Treat .py files as Jupytext percent-format notebooks
nb_custom_formats = {
    '.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
    '.py': 'myst-nb',
}
nb_execution_timeout = 120

# lattice_zoo.ipynb uses PyVista for 3-D rendering; PyVista/VTK requires
# OpenGL/EGL/OSMesa which is not available on CI.  Exclude the notebook from
# execution so Sphinx shows its static source.  Run it interactively to see
# the live 3-D plots.
nb_execution_excludepatterns = ["**/lattice_zoo.ipynb"]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'conf.py']

suppress_warnings = ['autosummary']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
