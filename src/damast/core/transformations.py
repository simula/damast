from __future__ import annotations

import copy
import importlib
import importlib.metadata
import importlib.util
import inspect
import os
import re
import sys
from abc import abstractmethod
from logging import getLogger
from pathlib import Path
from types import ModuleType

import numpy as np
import polars

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataSpecification

from .constants import (
    DAMAST_DEFAULT_DATASOURCE,
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
)
from .formatting import DEFAULT_INDENT

logger = getLogger(__name__)


class PluginManager:
    """
    Discovers and resolves :class:`PipelineElement` 'plugin' transformers, i.e.
    transformers that are not necessarily part of the damast package itself.

    Two plugin sources are supported:

    - packages that register :class:`PipelineElement` subclasses via the
      ``damast.transformers`` entry-point group, e.g. in their own pyproject.toml::

        [project.entry-points."damast.transformers"]
        MyTransformer = "acme_pkg.transformers:MyTransformer"

    - loose ``*.py`` files in directories listed in the ``DAMAST_PLUGIN_PATH``
      environment variable (os.pathsep-separated), for local/ad-hoc transformers that
      are not part of an installed package. Every top-level file found there is
      imported once (using its filename stem as 'module_name'), so that any
      :class:`PipelineElement` subclasses it defines become resolvable exactly like
      classes from an installed package.
    """

    #: Entry-point group that plugin packages use to advertise PipelineElement subclasses
    ENTRY_POINT_GROUP = "damast.transformers"

    #: Environment variable with an os.pathsep-separated list of local plugin directories
    PLUGIN_PATH_ENV = "DAMAST_PLUGIN_PATH"

    def __init__(self):
        #: module_name -> loaded module, for modules imported from PLUGIN_PATH_ENV
        self._local_modules: dict[str, ModuleType] = {}
        #: module_name -> source file, used to detect/warn about name collisions
        self._local_files: dict[str, Path] = {}
        self._loaded = False
        self._requirement_cache: dict[str, dict[str, str] | None] = {}

    @property
    def local_modules(self) -> dict[str, ModuleType]:
        return dict(self._local_modules)

    @property
    def local_files(self) -> dict[str, Path]:
        return dict(self._local_files)

    def plugin_path_dirs(self) -> list[Path]:
        raw = os.environ.get(self.PLUGIN_PATH_ENV, "")
        return [Path(p) for p in raw.split(os.pathsep) if p.strip()]

    def load_local_plugins(self, force: bool = False) -> dict[str, ModuleType]:
        """
        Import loose '*.py' files found in :attr:`PLUGIN_PATH_ENV` directories, so that
        any PipelineElement subclasses they define become resolvable by
        'module_name'/'class_name' - the same way as classes from an installed package.

        :param force: Re-scan the configured directories and re-import their files, even
            if they were already loaded in this process
        """
        if self._loaded and not force:
            return self._local_modules

        if force:
            self._local_modules.clear()
            self._local_files.clear()
            self._requirement_cache.clear()

        for plugin_dir in self.plugin_path_dirs():
            if not plugin_dir.is_dir():
                logger.warning(f"PluginManager: {self.PLUGIN_PATH_ENV} entry '{plugin_dir}'"
                               " is not a directory - skipping")
                continue

            for py_file in sorted(plugin_dir.glob("*.py")):
                module_name = py_file.stem
                if module_name.startswith("_"):
                    continue

                existing_file = self._local_files.get(module_name)
                if existing_file is not None:
                    if existing_file != py_file:
                        logger.warning(
                            f"PluginManager: plugin module '{module_name}' from '{py_file}' collides with"
                            f" already loaded '{existing_file}' - keeping the first one"
                        )
                    continue

                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.warning(f"PluginManager: failed to load plugin '{py_file}': {e}")
                    continue

                sys.modules[module_name] = module
                self._local_modules[module_name] = module
                self._local_files[module_name] = py_file

        self._loaded = True
        return self._local_modules

    def reload(self):
        """
        Force re-scanning of :attr:`PLUGIN_PATH_ENV` directories and re-importing their files.

        Useful after the environment variable was changed, or files were added/edited, since
        directories are otherwise only scanned once per process.
        """
        self.load_local_plugins(force=True)

    def resolve_requirement(self, module_name: str) -> dict[str, str] | None:
        """
        Identify what installable distribution package - or local plugin file - provides
        ``module_name``.

        This is used to record which package (or local file) a transformer originates from
        when a pipeline is saved, so that loading the pipeline elsewhere can point users at
        the missing package/file instead of failing with a bare :class:`ImportError`.

        :param module_name: Dotted module path of a :class:`PipelineElement` subclass
        :return: Dict with 'distribution' and 'version' for an installed package; a dict with
            'hint': 'local' and 'path' for a transformer loaded from :attr:`PLUGIN_PATH_ENV`;
            or None if it could not be resolved at all (e.g. the class is defined in a script or
            notebook that is neither installed nor on the plugin path)
        """
        if module_name in self._requirement_cache:
            return self._requirement_cache[module_name]

        self.load_local_plugins()

        result = None
        local_file = self._local_files.get(module_name)
        if local_file is not None:
            result = {"hint": "local", "path": str(local_file)}
        else:
            top_level = module_name.split(".")[0]
            try:
                distributions = importlib.metadata.packages_distributions().get(top_level)
            except Exception:
                distributions = None

            if distributions:
                distribution = distributions[0]
                try:
                    version = importlib.metadata.version(distribution)
                    result = {"distribution": distribution, "version": version}
                except importlib.metadata.PackageNotFoundError:
                    result = None

        self._requirement_cache[module_name] = result
        return result

    def list_plugins(self) -> dict[str, str]:
        """
        Discover transformer plugins from both the entry-point group and local plugin path.

        This is purely a discovery/documentation aid - :func:`PipelineElement.create_new`
        resolves classes by ``module_name``/``class_name`` regardless of whether they are
        registered here.

        :return: Mapping of class name to its 'module_name:class_name' target
        """
        plugins = {
            ep.name: ep.value
            for ep in importlib.metadata.entry_points(group=self.ENTRY_POINT_GROUP)
        }

        for module_name, module in self.load_local_plugins().items():
            for attr_name, obj in vars(module).items():
                if (inspect.isclass(obj)
                        and issubclass(obj, PipelineElement)
                        and obj is not PipelineElement
                        and obj.__module__ == module_name):
                    plugins[attr_name] = f"{module_name}:{attr_name}"

        return plugins


#: Default, process-wide plugin manager used by PipelineElement
plugin_manager = PluginManager()


class Transformer:
    uuid: str

    def set_uuid(self, uuid: str):
        self.uuid = uuid

    def fit(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame | None = None):
        pass

    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame | None = None):
        return df

    def fit_transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame | None = None):
        if not other:
            self.fit(df=df)
            return self.transform(df=df)
        else:
            self.fit(df, other)
            return self.transform(df,other)

class PipelineElement(Transformer):
    #: Pipeline in which context this processor will be run
    parent_pipeline: 'DataProcessingPipeline' #noqa

    #: Map names of input and outputs for a particular pipeline
    _name_mappings: dict[str, dict[str, str]]

    #: Map names of datasource (arguments) to a specific (extra) transformer arguments
    def set_parent(self, pipeline: 'DataProcessingPipeline'): #noqa
        """
        Sets the parent pipeline for this pipeline element

        :param pipeline: Parent pipeline
        """
        self.parent_pipeline = pipeline

    @property
    def name_mappings(self) -> dict[str, dict[str, str]]:
        """
        Get current name mappings for this instance
        """
        if not hasattr(self, "_name_mappings"):
            self._name_mappings = { DAMAST_DEFAULT_DATASOURCE: {}}
        return self._name_mappings

    @property
    def parameters(self) -> dict[str, str]:
        """
        Get the current list of parameter for initialization
        """
        if not hasattr(self, "_parameters"):
            self._parameters = None

        self.prepare_parameters()

        return self._parameters

    def prepare_parameters(self):
        parameters = {}
        for p in inspect.signature(self.__init__).parameters:
            parameter_name = p
            if parameter_name not in ['args', 'kwargs']:
                # only process named parameters
                try:
                    parameters[parameter_name] = getattr(self, parameter_name)
                except AttributeError:
                    raise ValueError(f"PipelineElement: please ensure that {self.__class__.__name__}.__init__ keyword arguments"
                            f" are saved in an attribute of the same name: missing '{parameter_name}'")

        self._parameters = parameters

    def get_name(self, name: str, datasource: str | None = None) -> any:
        if datasource is None:
            datasource = DAMAST_DEFAULT_DATASOURCE

        return self._get_name(name=name, datasource=datasource)

    def _get_name(self, name: str, datasource: str | None) -> any:
        """
        Add the fully resolved input/output name for this key.

        :param name: Name as used in the input spec, or pattern "{{x}}_suffix" in order to create a dynamic
                     output based an existing and renameable input
        :param datasource: In cases of multiple input for a node, define the datasource that shall be used
        :return: Name for this input after resolving name mappings and references
        """
        if not isinstance(name, str):
            raise TypeError(f"{self.__class__.__name__}.get_name: provided transformer label is not a string: {name}")

        if datasource is not None:
            try:
                name_mappings = self.name_mappings[datasource]
            except KeyError:
                raise RuntimeError(f"PipelineElement._get_name: not {datasource} in mappings: {self.name_mappings}")
        else:
            name_mappings = self.name_mappings

        if name in name_mappings:
            # allow multiple levels of name resolution, e.g.,
            # x -> y, y -> z --> x -> z
            mapped_name = name_mappings[name]
            if mapped_name == name:
                return name

            return self._get_name(mapped_name, datasource=datasource)

        # Allow to use patterns, so that an existing input
        # reference can be reused for dynamic labelling
        while re.search("{{\\w+}}", name):
            for match in re.finditer("{{\\w+}}", name):
                resolved_name = match.group()[2:-2]
                if resolved_name in name_mappings:
                    resolved_name = name_mappings[resolved_name]

                name = name.replace(match.group(), resolved_name)

        # If multiple sources are involved, allow to use the pattern {{<datasource_label>:<field_name>}}
        m = re.match("{{(\\w+):(\\w+)}}", name)
        if m:
            source_label = m.groups()[0]
            field_name = m.groups()[1]
            if source_label not in self.name_mappings:
                raise RuntimeError(f"{self.__class__.__name__}.get_name: unknown source '{source_label}' defined in {name}")

            resolved_name = field_name
            if field_name in self.name_mappings[source_label]:
                resolved_name = self.name_mappings[source_label][field_name]
            name = name.replace(m.group(), resolved_name)

        return name

    @abstractmethod
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Default transform implementation
        """

    @property
    def input_specs(self) -> dict[str, list[DataSpecification]]:
        if not hasattr(self.transform, DECORATED_INPUT_SPECS):
            raise AttributeError(
                f"{self.__class__.__name__}.validate: missing input specification"
                f" for processing step '{self.__class__.__name__}'"
            )

        generic_spec = getattr(self.transform, DECORATED_INPUT_SPECS)
        specs = copy.deepcopy(generic_spec)
        for label, speclist in specs.items():
            for spec in speclist:
                spec.name = self.get_name(spec.name, label)

        return specs

    @property
    def output_specs(self) -> list[DataSpecification]:
        if not hasattr(self.transform, DECORATED_OUTPUT_SPECS):
            raise AttributeError(
                f"{self.__class__.__name__}.validate: missing output specification"
                f" for processing step '{self.__class__.__name__}'"
            )

        generic_spec = getattr(self.transform, DECORATED_OUTPUT_SPECS)
        specs = copy.deepcopy(generic_spec)
        for spec in specs:
            # there will be only 1 dataframe as output
            spec.name = self.get_name(spec.name)

        return specs

    @classmethod
    def _missing_plugin_message(cls,
                                module_name: str,
                                class_name: str,
                                requires: dict[str, str] | None) -> str:
        if requires and requires.get("distribution"):
            pip_spec = requires["distribution"]
            if requires.get("version"):
                pip_spec += f"=={requires['version']}"
            return (f"{cls.__name__}.create_new: could not load transformer '{class_name}' from '{module_name}'. "
                    f"This pipeline requires the plugin package '{pip_spec}', which is not installed. "
                    f"Install it with: pip install {pip_spec}")

        plugin_path = os.environ.get(PluginManager.PLUGIN_PATH_ENV, "<unset>")
        origin_hint = ""
        if requires and requires.get("hint") == "local":
            origin_hint = f" It was saved as a local transformer, originally loaded from '{requires.get('path')}'."

        return (f"{cls.__name__}.create_new: could not load '{class_name}' from '{module_name}'.{origin_hint} "
                "Ensure that the package providing this transformer is installed and importable, "
                f"or - if it is a local/ad-hoc transformer - that the directory containing "
                f"'{module_name}.py' is listed in the '{PluginManager.PLUGIN_PATH_ENV}' environment variable"
                f" (currently: {plugin_path}).")

    @classmethod
    def _check_requirement(cls,
                           module_name: str,
                           class_name: str,
                           requires: dict[str, str]):
        distribution = requires.get("distribution")
        if not distribution:
            return

        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(cls._missing_plugin_message(module_name, class_name, requires))

        expected_version = requires.get("version")
        if expected_version and installed_version != expected_version:
            logger.warning(
                f"{cls.__name__}.create_new: '{class_name}' was saved with plugin package"
                f" '{distribution}=={expected_version}', but '{installed_version}' is installed."
                " Results may differ from when the pipeline was created."
            )

    @classmethod
    def create_new(cls,
                   module_name: str,
                   class_name: str,
                   name_mappings: dict[str, dict[str, str]] | None = None,
                   parameters: dict[str, any] | None = {},
                   requires: dict[str, str] | None = None) -> PipelineElement:
        """
        Create a new PipelineElement Subclass instance dynamically

        :param module_name: Name of the module for the PipelineElement class
        :param class_name: Name of the PipelineElement subclass
        :param name_mappings: Dictionary of name mappings that should apply
        :param requires: Optional info on where this transformer was resolved from when the
            pipeline was saved - see :func:`__iter__`. Either an installed plugin package
            ('distribution' + 'version'), or a local plugin file ('hint': 'local' + 'path').
            Used to give an actionable error when the providing package/file is missing.
        :return: Instance for the PipelineElement instance

        .. note::
            If the class cannot be found on the regular import path, directories listed in
            the ``DAMAST_PLUGIN_PATH`` environment variable are scanned for a matching
            '<module_name>.py' file before giving up - see :func:`list_plugins`.

        :raise ValueError: If module or class with given name is not specified
        :raise ImportError: If class could not be loaded
        """
        if module_name is None:
            raise ValueError(f"{cls.__name__}.create_new: missing 'module_name'")

        if class_name is None:
            raise ValueError(f"{cls.__name__}.create_new: missing 'class_name'")

        if requires:
            cls._check_requirement(module_name=module_name, class_name=class_name, requires=requires)

        plugin_manager.load_local_plugins()

        try:
            p_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(cls._missing_plugin_message(module_name, class_name, requires))

        if hasattr(p_module, class_name):
            klass = getattr(p_module, class_name)
        else:
            raise ImportError(cls._missing_plugin_message(module_name, class_name, requires))
        if parameters:
            instance = klass(**parameters)
        else:
            instance = klass()

        if name_mappings is None:
            name_mappings = { DAMAST_DEFAULT_DATASOURCE: {}}

        instance._name_mappings = name_mappings
        return instance

    @classmethod
    def list_plugins(cls) -> dict[str, str]:
        """
        Discover transformer plugins - see :class:`PluginManager` for details on the two
        supported sources (installed packages via entry-points, and local files via
        ``DAMAST_PLUGIN_PATH``).

        :return: Mapping of class name to its 'module_name:class_name' target
        """
        return plugin_manager.list_plugins()

    @classmethod
    def reload_plugins(cls):
        """
        Force re-scanning of ``DAMAST_PLUGIN_PATH`` directories and re-importing their files -
        see :func:`PluginManager.reload`.
        """
        plugin_manager.reload()

    def __iter__(self):
        yield "module_name", f"{self.__class__.__module__}"
        yield "class_name", f"{self.__class__.__qualname__}"
        yield "parameters", self.parameters
        yield "name_mappings", self.name_mappings
        yield "requires", plugin_manager.resolve_requirement(self.__class__.__module__)

    def __eq__(self, other):
        return dict(self) == dict(other)

    @classmethod
    def get_types(cls) -> list[PipelineElement]:
        """
        Get all available PipelineElement implementations

        :return: List of PipelineElement classes
        """
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(subclass._subclasses())
        return klasses

    @classmethod
    def _subclasses(cls) -> list[PipelineElement]:
        """
        Generate the list of subclasses for the calling class
        """
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(subclass._subclasses())
        return klasses

    @classmethod
    def generate_subclass_documentation(cls) -> str:
        """
        Generate the documentation for all subclasses of ::class::`PipelineElement`
        """
        implementations = sorted(cls.get_types(), key=str)
        txt = ""
        for k in implementations:
            txt += "=" * 80
            txt += f"\n{k.__name__} -- from {k.__module__}\n"
            txt += k.__doc__
            txt += '\n'
        return txt

    def to_str(self, indent_level: int = 0) -> str:
        hspace = DEFAULT_INDENT * indent_level
        data = hspace + self.__class__.__name__ + "\n"
        if hasattr(self.transform, DECORATED_DESCRIPTION):
             description = getattr(self.transform, DECORATED_DESCRIPTION)
             data += (
                     hspace + DEFAULT_INDENT * 2 + "description: " + description + "\n"
                )

        data += hspace + DEFAULT_INDENT * 2 + "input:\n"
        if hasattr(self.transform, DECORATED_INPUT_SPECS):
            data += DataSpecification.to_str(
                self.input_specs, indent_level=indent_level + 4
            )

        data += hspace + DEFAULT_INDENT * 2 + "output:\n"
        if hasattr(self.transform, DECORATED_OUTPUT_SPECS):
            data += DataSpecification.to_str(
                self.output_specs, indent_level=indent_level + 4
            )
        return data

    def __deepcopy__(self, memo) -> PipelineElement:
        new_element = copy.copy(self)
        new_element._name_mappings = copy.deepcopy(self.name_mappings)
        return new_element

# Only for internal use
class MultiCycleTransformer(Transformer):
    def __init__(self, features: list[str], n: int):
        self.features = features
        self.n = n

    def transform(self, df: AnnotatedDataFrame):
        if type(df) is not AnnotatedDataFrame:
            raise ValueError(f"Transformer requires 'AnnotatedDataFrame',"
                    f" but got '{type(df)}")
        clone = df.copy()

        for feature in self.features:
            clone.lazyframe = clone.lazyframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_x")
                )
            clone.lazyframe = clone.lazyframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_y")
                )
        return clone

