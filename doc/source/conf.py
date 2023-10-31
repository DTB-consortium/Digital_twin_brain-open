# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os.path
import sys
from os.path import dirname, abspath
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'utils'))

project = 'DTB'
copyright = '2022, leopold'
author = 'leopold'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['cuda', 'default_params.py']

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']

autodoc_mock_imports = ['torch', 'cuda', 'numba', 'prettytable', 'h5py', 'seaborn', 'matplotlib', 'pandas', 'scikit-learn', 'sparse', 'scipy', 'mpi4py']
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

master_doc = 'index'
