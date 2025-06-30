import os

project = "tlm_adjoint"

extensions = ["autoapi.extension",
              "nbsphinx",
              "numpydoc",
              "sphinx.ext.intersphinx",
              "sphinx_rtd_theme"]

autoapi_type = "python"
autoapi_dirs = ["../../tlm_adjoint"]
autoapi_ignore = ["*/checkpoint_schedules/__init__.py",
                  "*/fenics/__init__.py",
                  "*/fenics/backend.py",
                  "*/fenics/backend_patches.py",
                  "*/fenics/parameters.py",
                  "*/firedrake/__init__.py",
                  "*/firedrake/backend.py",
                  "*/firedrake/backend_patches.py",
                  "*/firedrake/parameters.py",
                  "*/patch.py",
                  "*/petsc.py"]
autoapi_add_toctree_entry = False
autoapi_options = []

nbsphinx_execute = "auto"

numpydoc_validation_checks = {"all", "GL08"}

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": False}

exclude_patterns = []
html_static_path = ["static"]
templates_path = []

html_css_files = ["custom.css"]

intersphinx_mapping = {"checkpoint_schedules": ("https://www.firedrakeproject.org/checkpoint_schedules", None),  # noqa: E501
                       "firedrake": ("https://www.firedrakeproject.org", None),
                       "h5py": ("https://docs.h5py.org/en/stable", None),
                       "numpy": ("https://numpy.org/doc/stable", None),
                       "petsc4py": ("https://petsc.org/main/petsc4py", None),
                       "pyop2": ("https://op2.github.io/PyOP2", None),
                       "python": ("https://docs.python.org/3", None),
                       "scipy": ("https://docs.scipy.org/doc/scipy", None),
                       "sympy": ("https://docs.sympy.org/latest", None),
                       "torch": ("https://pytorch.org/docs/stable", None),
                       "ufl": ("https://docs.fenicsproject.org/ufl/main", None)}  # noqa: E501

if "PYTHONWARNINGS" not in os.environ:
    os.environ["PYTHONWARNINGS"] = "ignore"
