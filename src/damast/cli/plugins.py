import os
from argparse import ArgumentParser

from damast.cli.base import BaseParser
from damast.core.transformations import PipelineElement, PluginManager


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

        for name, target in sorted(plugins.items()):
            print(f"{name}: {target}")
