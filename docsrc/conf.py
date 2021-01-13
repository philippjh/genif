extensions = [
    'breathe',
    'exhale',
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting"
]
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'doxyoutput']
autodoc_typehints = 'none'
autodoc_docstring_signature = True

breathe_projects = {
    "genif": "./doxyoutput/xml"
}
breathe_default_project = "genif"
exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ../genif"
}

project = "Generalized Isolation Forest"
html_title = "Generalized Isolation Forest Documentation"
html_theme = "sphinx_rtd_theme"
