project = "tlm_adjoint"

extensions = ["autoapi.extension",
              "nbsphinx",
              "sphinx_rtd_theme"]

autoapi_type = "python"
autoapi_dirs = ["../../tlm_adjoint"]
autoapi_ignore = ["*/_code_generator/*",
                  "*/checkpoint_schedules/*",
                  "*/fenics/*",
                  "*/firedrake/*",
                  "*/numpy/*"]
autoapi_add_toctree_entry = False
autoapi_options = {"private-members": False}

nbsphinx_execute = "never"

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": False,
                      "titles_only": True}

exclude_patterns = []
html_static_path = ["static"]
templates_path = []

html_css_files = ["custom.css"]
