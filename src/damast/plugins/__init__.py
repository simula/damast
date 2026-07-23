"""
Resolve :class:`damast.core.transformations.PipelineElement` 'plugin' transformers by name, so
that e.g. ``from damast.plugins import MyTransformer`` works for any transformer discoverable
by :class:`damast.core.transformations.PluginManager` - whether it comes from an installed
package's ``damast.transformers`` entry point, or a loose file on ``DAMAST_PLUGIN_PATH``.

Names are resolved lazily on first access (via module ``__getattr__``, see :pep:`562`): nothing
is imported/loaded until a specific name is actually requested, so installed plugin packages
that are never referenced here are never imported, and a ``DAMAST_PLUGIN_PATH`` set after this
module was first imported is still picked up.

If a name is registered by more than one source (two local plugin files, two entry-points, or a
local plugin and an entry-point sharing a name), a warning is logged and the first source found
wins - local plugin files are checked before entry-points, matching the precedence used by
:func:`damast.core.transformations.PluginManager.list_plugins`.
"""
from __future__ import annotations

import importlib.metadata
import inspect
from logging import getLogger

from damast.core.transformations import PipelineElement, PluginManager, plugin_manager

__all__: list[str] = []

logger = getLogger(__name__)


def __getattr__(name: str) -> type[PipelineElement]:
    local_matches = [
        (module_name, obj)
        for module_name, module in plugin_manager.load_local_plugins().items()
        for obj in [vars(module).get(name)]
        if (inspect.isclass(obj) and issubclass(obj, PipelineElement)
            and obj is not PipelineElement and obj.__module__ == module_name)
    ]
    entry_point_matches = [
        ep for ep in importlib.metadata.entry_points(group=PluginManager.ENTRY_POINT_GROUP)
        if ep.name == name
    ]

    sources = [module_name for module_name, _ in local_matches] + [ep.value for ep in entry_point_matches]
    if len(sources) > 1:
        logger.warning(
            f"damast.plugins: plugin name '{name}' is ambiguous - registered by more than one"
            f" source ({', '.join(sources)}) - using '{sources[0]}'"
        )

    if local_matches:
        return local_matches[0][1]

    if entry_point_matches:
        return entry_point_matches[0].load()

    raise AttributeError(f"module 'damast.plugins' has no plugin named '{name}'")


def __dir__() -> list[str]:
    return sorted(plugin_manager.list_plugins())
