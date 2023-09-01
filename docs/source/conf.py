project = "tlm_adjoint"

extensions = ["autoapi.extension",
              "nbsphinx",
              "sphinx_rtd_theme"]

autoapi_type = "python"
autoapi_dirs = ["../../tlm_adjoint"]
autoapi_ignore = ["*/_code_generator/__init__.py",
                  "*/checkpoint_schedules/__init__.py",
                  "*/fenics/__init__.py",
                  "*/fenics/backend.py",
                  "*/fenics/backend_code_generator_interface.py",
                  "*/fenics/backend_interface.py",
                  "*/fenics/backend_overrides.py",
                  "*/firedrake/__init__.py",
                  "*/firedrake/backend.py",
                  "*/firedrake/backend_code_generator_interface.py",
                  "*/firedrake/backend_interface.py",
                  "*/firedrake/backend_overrides.py",
                  "*/numpy/__init__.py",
                  "*/numpy/backend.py",
                  "*/numpy/backend_interface.py"]
autoapi_add_toctree_entry = False
autoapi_options = {"private-members": False}

nbsphinx_execute = "auto"

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": False}

exclude_patterns = []
html_static_path = ["static"]
templates_path = []

html_css_files = ["custom.css"]
