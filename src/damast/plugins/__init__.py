"""
Resolve :class:`damast.core.transformations.PipelineElement` 'plugin' transformers per plugin
package, so that e.g. ``from damast.plugins.acme import MyTransformer`` works for any transformer
discoverable by :class:`damast.core.transformations.PluginManager` - whether it comes from an
installed package's ``damast.transformers`` entry point (for a single class, or for a whole
module), or a local plugin directory on ``DAMAST_PLUGIN_PATH``.

The plugin package is the top-level package of the module defining the transformer (see
:func:`damast.core.transformations.PluginManager.plugin_package`): ``acme`` for an installed
``acme.transformers:MyTransformer``, or ``name`` for a local directory registered as
``name=path``. Scoping names by package means two plugins can provide a transformer of the same
name without clashing.

Names are resolved lazily on first access: ``damast.plugins.<package>`` is created on import
without importing anything, and a transformer is only looked up once it is actually requested - a
module entry-point is only imported if nothing else in that package provides the name.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType

from damast.core.transformations import PipelineElement, plugin_manager

__all__: list[str] = []


class _PluginPackageFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Creates the ``damast.plugins.<package>`` namespace modules on import."""

    def find_spec(self, fullname: str, path=None, target=None):
        package = fullname.removeprefix(f"{__name__}.")
        if package == fullname or "." in package:
            return None
        if package not in plugin_manager.plugin_packages():
            raise ModuleNotFoundError(f"No plugin package '{package}' - available:"
                                      f" {sorted(plugin_manager.plugin_packages())}", name=fullname)
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType):
        package = module.__name__.rpartition(".")[2]

        def __getattr__(name: str) -> type[PipelineElement]:
            if name.startswith("__"):
                raise AttributeError(name)
            return plugin_manager.resolve_plugin(package, name)

        def __dir__() -> list[str]:
            prefix = f"{package}."
            return sorted(name.removeprefix(prefix) for name in plugin_manager.list_plugins()
                          if name.startswith(prefix))

        module.__getattr__ = __getattr__
        module.__dir__ = __dir__


# appended, so a real submodule of damast.plugins would still take precedence
if not any(isinstance(finder, _PluginPackageFinder) for finder in sys.meta_path):
    sys.meta_path.append(_PluginPackageFinder())


def __getattr__(name: str) -> ModuleType:
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        raise AttributeError(f"module '{__name__}' has no plugin package '{name}'") from e


def __dir__() -> list[str]:
    return sorted(plugin_manager.plugin_packages())
