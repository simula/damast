import os
from argparse import ArgumentParser

from damast.cli.base import BaseParser
from damast.core.transformations import PipelineElement, PluginManager, plugin_manager


class PluginsParser(BaseParser):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = ("damast plugins - list transformer plugins registered by installed packages"
                              f" (entry-point group '{PluginManager.ENTRY_POINT_GROUP}') or via the"
                              f" '{PluginManager.PLUGIN_PATH_ENV}' environment variable")

    def execute(self, args):
        plugins = PipelineElement.list_plugins()
        if not plugins:
            plugin_path = os.environ.get(PluginManager.PLUGIN_PATH_ENV, "<unset>")
            print(f"No transformer plugins registered (entry-point group '{PluginManager.ENTRY_POINT_GROUP}', "
                 f"{PluginManager.PLUGIN_PATH_ENV}={plugin_path})")
            return

        # '<package>.<class>' -> 'module:class', grouped per plugin package
        packages: dict[str, list[tuple[str, str]]] = {}
        for qualified_name, target in plugins.items():
            package, class_name = qualified_name.split(".", 1)
            packages.setdefault(package, []).append((class_name, target.split(":")[0]))

        for package, transformers in sorted(packages.items()):
            source = self._describe_source(plugin_manager.resolve_requirement(transformers[0][1]))
            print(f"{package}{f' ({source})' if source else ''}")

            width = max(len(class_name) for class_name, _ in transformers)
            for class_name, module_name in sorted(transformers):
                # module relative to the package - omitted if defined in the package module itself
                relative_module = module_name.removeprefix(package)
                print(f"    {class_name:<{width}}  {relative_module}".rstrip())

    @staticmethod
    def _describe_source(requirement: dict[str, str] | None) -> str:
        if not requirement:
            return ""
        if requirement.get("distribution"):
            return f"{requirement['distribution']}=={requirement['version']}"
        return f"local: {requirement['path']}"
