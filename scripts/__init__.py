"""Alias package that forwards 'scripts' imports to 'new_scripts'.

This shim allows legacy code that imports modules via the old
    import scripts.utils.data_utils
syntax to continue working after the codebase was reorganised into
    new_scripts/utils/.

It dynamically maps every sub-module from the real package
`new_scripts` onto the `scripts` namespace in `sys.modules`.
"""

import importlib
import pkgutil
import sys

# Import the real top-level package
_real_pkg = importlib.import_module("new_scripts")

# Expose the top-level package itself under the alias name
sys.modules[__name__] = _real_pkg  # type: ignore

# Walk through all sub-modules inside ``new_scripts`` and create
# corresponding entries under the ``scripts`` namespace so that, e.g.
# ``import scripts.utils.config`` resolves correctly.
for module_info in pkgutil.walk_packages(_real_pkg.__path__, prefix="new_scripts."):
    try:
        mod = importlib.import_module(module_info.name)
        alias_name = module_info.name.replace("new_scripts", __name__, 1)
        sys.modules[alias_name] = mod
    except Exception:  # pragma: no cover – ignore failures during lazy import
        # If a sub-module raises at import-time we still register an empty
        # placeholder so that ``import scripts.xxx`` doesn’t fail with
        # ``ModuleNotFoundError`` but raises the original error later when
        # the code actually accesses the module.
        sys.modules.setdefault(alias_name, None) 