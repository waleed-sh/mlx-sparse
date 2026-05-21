# Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from importlib.metadata import version as _pkg_version
from importlib.util import find_spec

# Make the package importable from the source tree if not installed.
# conf.py lives in docs/, the package root is one level up.
sys.path.insert(0, os.path.abspath(".."))

project = "mlx-sparse"
copyright = "2026, mlx-sparse contributors"
author = "mlx-sparse contributors"

try:
    release = _pkg_version("mlx-sparse")
except Exception:
    release = "0.0.0.dev0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

# Never re-execute notebooks at build time, use pre-computed outputs.
nb_execution_mode = "off"

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__matmul__",
}

# Read the Docs builds on Linux, while MLX/Metal support is macOS-centric.
# Notebooks are not executed during documentation builds, so mocking MLX when it
# is absent keeps API docs importable without pretending RTD can run kernels.
autodoc_mock_imports = []
if find_spec("mlx") is None:
    autodoc_mock_imports.extend(["mlx", "mlx.core"])

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/waleed-sh/mlx-sparse",
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "primary_sidebar_end": [],
    "secondary_sidebar_items": [],
    "footer_start": ["copyright"],
    "footer_end": [],
    "logo": {
        "text": "mlx-sparse",
    },
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

html_title = "mlx-sparse"
html_short_title = "mlx-sparse"
html_show_sourcelink = True
html_copy_source = False
