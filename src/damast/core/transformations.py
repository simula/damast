from __future__ import annotations

import copy
import importlib
import importlib.metadata
import importlib.machinery
import importlib.util
import inspect
import keyword
import os
import pkgutil
import re
import sys
from abc import abstractmethod
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from types import ModuleType

import numpy as np
import polars
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion

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


def _check_version_requirement(installed_version: str, expected_version: str) -> tuple[bool, str]:
    """
    Check whether ``installed_version`` satisfies ``expected_version``.

    ``expected_version`` may be a full PEP 440 specifier (``'>=1.2.0'``, ``'==1.2.3'``,
    ``'~=1.4'``) which is honored as-is, or a bare version (``'1.2.0'``, as recorded by
    :func:`PluginManager.resolve_requirement`) which has no operator and is therefore treated as
    a minimum requirement, like a normal Python ``'>='`` dependency specifier.

    :param installed_version: The currently installed version
    :param expected_version: A PEP 440 specifier, or a bare version treated as a minimum
    :return: Tuple of (whether the requirement is satisfied, the specifier text used - for
        display in a warning message)
    """
    try:
        specifier = SpecifierSet(expected_version)
    except InvalidSpecifier:
        try:
            specifier = SpecifierSet(f">={expected_version}")
        except InvalidSpecifier:
            # Not a PEP 440 version/specifier at all (e.g. a VCS/date-based build) - fall back
            # to exact match, since ordering can't be meaningfully compared.
            return installed_version == expected_version, f"=={expected_version}"

    try:
        return installed_version in specifier, str(specifier)
    except InvalidVersion:
        return installed_version == expected_version, f"=={expected_version}"


class PluginManager:
    """
    Discovers and resolves :class:`PipelineElement` 'plugin' transformers, i.e.
    transformers that are not necessarily part of the damast package itself.

    Supported plugin sources:

    - installed packages that register :class:`PipelineElement` subclasses via the
      ``damast.transformers`` entry-point group in their own pyproject.toml - either one
      entry per class, or one entry per module (every PipelineElement defined in that module,
      or in the top-level submodules of that package, is registered)::

        [project.entry-points."damast.transformers"]
        MyTransformer = "acme_pkg.transformers:MyTransformer"
        acme_pkg = "acme_pkg.transformers"

      Either way, the transformers are provided as ``damast.plugins.<top-level package>``, here
      ``damast.plugins.acme_pkg``. The name of a class entry is the transformer's name; the name
      of a module entry is only a label - use the package name, a different one is warned about.

    - local directories listed in the ``DAMAST_PLUGIN_PATH`` environment variable
      (os.pathsep-separated), for transformers that are not part of an installed package:

      - ``name=path`` (or :func:`register_plugin_package`) imports the directory as a package
        called ``name``, so its top-level files become ``name.<stem>`` and may use relative
        imports (``from .helpers import x``), including into subpackages
      - a bare ``path`` (deprecated) imports each top-level file as a flat module named after
        its filename stem - relative imports are not possible there

    In either case every top-level, non-underscore file is imported once, so that the
    :class:`PipelineElement` subclasses it defines become resolvable exactly like classes from
    an installed package.
    """

    #: Entry-point group that plugin packages use to advertise PipelineElement subclasses
    ENTRY_POINT_GROUP = "damast.transformers"

    #: Environment variable with an os.pathsep-separated list of local plugin directories
    PLUGIN_PATH_ENV = "DAMAST_PLUGIN_PATH"

    def __init__(self):
        #: module_name -> loaded module, for modules imported from local plugin directories
        self._local_modules: dict[str, ModuleType] = {}
        #: module_name -> source file, used to detect/warn about name collisions
        self._local_files: dict[str, Path] = {}
        #: package name -> directory, for named local plugin directories that were loaded
        self._local_packages: dict[str, Path] = {}
        #: package name -> directory, registered via register_plugin_package()
        self._registered_packages: dict[str, Path] = {}
        #: module entry-point value -> modules found in it (see scan_module_entry_point)
        self._entry_point_modules: dict[str, dict[str, ModuleType]] = {}
        #: unnamed plugin directories a deprecation warning was already logged for
        self._warned_unnamed: set[Path] = set()
        #: (name, value) of module entry-points whose ignored name was already warned about
        self._warned_entry_points: set[tuple[str, str]] = set()
        self._loaded = False
        self._requirement_cache: dict[str, dict[str, str] | None] = {}

    @property
    def local_modules(self) -> dict[str, ModuleType]:
        return dict(self._local_modules)

    @property
    def local_files(self) -> dict[str, Path]:
        return dict(self._local_files)

    @property
    def local_packages(self) -> dict[str, Path]:
        return dict(self._local_packages)

    def plugin_path_dirs(self) -> list[Path]:
        return [path for _, path in self.plugin_path_entries()]

    def plugin_path_entries(self) -> list[tuple[str | None, Path]]:
        """
        Parse :attr:`PLUGIN_PATH_ENV` into ``(package_name, directory)`` pairs.

        An entry ``name=path`` names the package, a bare ``path`` yields ``None`` as name. An
        entry is only treated as named if the part before the first ``=`` contains no path
        separator, so plain paths that happen to contain ``=`` keep working.
        """
        raw = os.environ.get(self.PLUGIN_PATH_ENV, "")
        entries = []
        for entry in raw.split(os.pathsep):
            if not entry.strip():
                continue
            name, sep, path = entry.partition("=")
            if sep and not any(s in name for s in {"/", os.sep}):
                entries.append((name.strip(), Path(path)))
            else:
                entries.append((None, Path(entry)))
        return entries

    def register_plugin_package(self, name: str, path: str | Path):
        """
        Register a local plugin directory as package ``name`` - same as adding ``name=path``
        to :attr:`PLUGIN_PATH_ENV`. It is loaded on the next plugin lookup.

        :param name: Package name, a valid Python identifier
        :param path: Directory containing the plugin files
        :raise ValueError: If ``name`` is not a valid package name
        """
        if not self._is_valid_package_name(name):
            raise ValueError(f"PluginManager: '{name}' is not a valid plugin package name")
        self._registered_packages[name] = Path(path)
        self._loaded = False

    @staticmethod
    def _is_valid_package_name(name: str) -> bool:
        return name.isidentifier() and not keyword.iskeyword(name)

    def load_local_plugins(self, force: bool = False) -> dict[str, ModuleType]:
        """
        Import the plugin files found in :attr:`PLUGIN_PATH_ENV` directories and in packages
        registered via :func:`register_plugin_package`, so that any PipelineElement subclasses
        they define become resolvable by 'module_name'/'class_name' - the same way as classes
        from an installed package.

        :param force: Re-scan the configured directories and re-import their files, even
            if they were already loaded in this process
        """
        if self._loaded and not force:
            return self._local_modules

        if force:
            self._unload()
            # the import system caches directory listings - make added files visible
            importlib.invalidate_caches()

        # Do not write bytecode for local plugin sources: .pyc files are validated by the
        # source's mtime (in whole seconds) and size only, so an edit within the same second
        # would otherwise be missed by a reload.
        dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            entries = self.plugin_path_entries() + list(self._registered_packages.items())
            for name, plugin_dir in entries:
                if not plugin_dir.is_dir():
                    logger.warning(f"PluginManager: {self.PLUGIN_PATH_ENV} entry '{plugin_dir}'"
                                   " is not a directory - skipping")
                    continue

                if name is None:
                    self._load_flat_directory(plugin_dir)
                else:
                    self._load_package_directory(name, plugin_dir)
        finally:
            sys.dont_write_bytecode = dont_write_bytecode

        self._loaded = True
        return self._local_modules

    def _load_flat_directory(self, plugin_dir: Path):
        if plugin_dir not in self._warned_unnamed:
            self._warned_unnamed.add(plugin_dir)
            logger.warning(
                f"PluginManager: unnamed {self.PLUGIN_PATH_ENV} entry '{plugin_dir}' is deprecated"
                f" - use '<package_name>={plugin_dir}' to load it as a package"
            )

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

    def _load_package_directory(self, name: str, plugin_dir: Path):
        existing_dir = self._local_packages.get(name)
        if existing_dir is not None:
            if existing_dir != plugin_dir:
                logger.warning(f"PluginManager: plugin package '{name}' from '{plugin_dir}' collides"
                               f" with already loaded '{existing_dir}' - keeping the first one")
            return

        if not self._is_valid_package_name(name):
            logger.warning(f"PluginManager: '{name}' in {self.PLUGIN_PATH_ENV} entry"
                           f" '{name}={plugin_dir}' is not a valid package name - skipping")
            return

        # never shadow an importable module, e.g. from the standard library or an installed
        # plugin package - which then takes precedence
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            logger.warning(f"PluginManager: plugin package name '{name}' (for '{plugin_dir}') is"
                           " already importable - skipping")
            return

        # The directory itself becomes the package: its __init__.py if present, an empty
        # package otherwise. Submodules then resolve via the regular import system.
        init_file = plugin_dir / "__init__.py"
        if init_file.is_file():
            spec = importlib.util.spec_from_file_location(
                name, init_file, submodule_search_locations=[str(plugin_dir)])
        else:
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = [str(plugin_dir)]

        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        try:
            if spec.loader is not None:
                spec.loader.exec_module(package)
        except Exception as e:
            sys.modules.pop(name, None)
            logger.warning(f"PluginManager: failed to load plugin package '{name}' from '{init_file}': {e}")
            return

        self._local_packages[name] = plugin_dir
        for module_name, module in self._scan_package(package).items():
            self._local_modules[module_name] = module
            self._local_files[module_name] = Path(module.__file__) if module.__file__ else plugin_dir

    @staticmethod
    def _scan_package(module: ModuleType) -> dict[str, ModuleType]:
        """
        The given module, plus - if it is a package - its top-level, non-underscore,
        non-package submodules (imported here). A failing submodule is skipped with a warning.
        """
        modules = {module.__name__: module}
        for info in pkgutil.iter_modules(getattr(module, "__path__", [])):
            if info.ispkg or info.name.startswith("_"):
                continue
            module_name = f"{module.__name__}.{info.name}"
            try:
                modules[module_name] = importlib.import_module(module_name)
            except Exception as e:
                logger.warning(f"PluginManager: failed to load plugin module '{module_name}': {e}")
        return modules

    @staticmethod
    def is_module_entry_point(entry_point) -> bool:
        """Whether an entry point names a whole module ('pkg.mod') rather than a class ('pkg.mod:Class')."""
        return ":" not in entry_point.value

    def scan_module_entry_point(self, entry_point) -> dict[str, ModuleType]:
        """
        Import the module named by a module entry point - see :func:`is_module_entry_point` -
        and, if it is a package, its top-level submodules.

        :return: module_name -> module, empty if the module could not be imported
        """
        value = entry_point.value.strip()
        if value not in self._entry_point_modules:
            try:
                module = importlib.import_module(value)
            except Exception as e:
                logger.warning(f"PluginManager: failed to load plugin module '{value}' of"
                               f" entry-point '{entry_point.name}': {e}")
                self._entry_point_modules[value] = {}
            else:
                self._entry_point_modules[value] = self._scan_package(module)
        return self._entry_point_modules[value]

    @staticmethod
    def pipeline_elements(modules: dict[str, ModuleType]) -> list[tuple[str, str, type]]:
        """
        :return: (module_name, class_name, class) for each PipelineElement subclass *defined*
            in one of the given modules - re-exported classes are skipped
        """
        return [
            (module_name, attr_name, obj)
            for module_name, module in modules.items()
            for attr_name, obj in vars(module).items()
            if (inspect.isclass(obj)
                and issubclass(obj, PipelineElement)
                and obj is not PipelineElement
                and obj.__module__ == module_name)
        ]

    def _unload(self):
        """Forget all loaded local plugins and module entry points, and drop them from sys.modules."""
        for module_name in self._local_files:
            sys.modules.pop(module_name, None)
        for name in self._local_packages:
            for module_name in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
                sys.modules.pop(module_name, None)
        self._local_modules.clear()
        self._local_files.clear()
        self._local_packages.clear()
        self._entry_point_modules.clear()
        self._requirement_cache.clear()
        self._loaded = False

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
            'hint': 'local', 'package' and 'path' (the directory) for a transformer from a named
            local plugin package; a dict with 'hint': 'local' and 'path' (the file) for one from
            an unnamed :attr:`PLUGIN_PATH_ENV` directory; or None if it could not be resolved at
            all (e.g. the class is defined in a script or notebook that is neither installed nor
            on the plugin path)
        """
        if module_name in self._requirement_cache:
            return self._requirement_cache[module_name]

        self.load_local_plugins()

        result = None
        top_level = module_name.split(".")[0]
        local_file = self._local_files.get(module_name)
        if top_level in self._local_packages:
            result = {"hint": "local", "package": top_level, "path": str(self._local_packages[top_level])}
        elif local_file is not None:
            result = {"hint": "local", "path": str(local_file)}
        else:
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

    @staticmethod
    def plugin_package(module_name: str) -> str:
        """
        The plugin package a module belongs to, i.e. its top-level package - e.g. 'acme' for
        'acme.transformers', or the package name of a named local plugin directory. Plugin
        transformers are exposed per plugin package, as ``damast.plugins.<package>.<class>``.
        """
        return module_name.split(".")[0]

    def _entry_points(self) -> tuple[list, list]:
        """:return: (class entry-points, module entry-points) of :attr:`ENTRY_POINT_GROUP`"""
        entry_points = list(importlib.metadata.entry_points(group=self.ENTRY_POINT_GROUP))
        module_entry_points = [ep for ep in entry_points if self.is_module_entry_point(ep)]
        for ep in module_entry_points:
            self._warn_ignored_entry_point_name(ep)
        return ([ep for ep in entry_points if not self.is_module_entry_point(ep)], module_entry_points)

    def _warn_ignored_entry_point_name(self, entry_point):
        """
        Warn (once) if a module entry point's name differs from its plugin package: the namespace
        is always the top-level package, the name of a module entry point is only a label.
        """
        package = self.plugin_package(entry_point.value)
        key = (entry_point.name, entry_point.value)
        if entry_point.name != package and key not in self._warned_entry_points:
            self._warned_entry_points.add(key)
            logger.warning(f"PluginManager: the name '{entry_point.name}' of entry-point"
                           f" '{entry_point.name} = {entry_point.value}' is ignored - its transformers"
                           f" are provided by plugin package '{package}', i.e. damast.plugins.{package}")

    def plugin_packages(self) -> set[str]:
        """Names of all plugin packages - local ones and those of entry-points (not imported here)."""
        packages = {self.plugin_package(module_name) for module_name in self.load_local_plugins()}
        class_entry_points, module_entry_points = self._entry_points()
        packages |= {self.plugin_package(ep.value) for ep in class_entry_points + module_entry_points}
        return packages

    def resolve_plugin(self, package: str, name: str) -> type[PipelineElement]:
        """
        Resolve transformer ``name`` within plugin ``package`` - see :func:`plugin_package`.

        Sources are checked in this order, the first one wins (with a warning, if several
        provide it): local plugin files, class entry-points, then module entry-points - the
        latter are only imported if nothing else provides ``name``.

        :raise AttributeError: If the package provides no transformer ``name``
        """
        matches: dict[str, Callable[[], type]] = {
            f"{module_name}:{attr_name}": (lambda obj=obj: obj)
            for module_name, attr_name, obj in self.pipeline_elements(self.load_local_plugins())
            if self.plugin_package(module_name) == package and attr_name == name
        }
        class_entry_points, module_entry_points = self._entry_points()
        for ep in class_entry_points:
            if self.plugin_package(ep.value) == package and ep.name == name:
                matches.setdefault(ep.value, ep.load)

        if not matches:
            for ep in module_entry_points:
                if self.plugin_package(ep.value) != package:
                    continue
                for module_name, attr_name, obj in self.pipeline_elements(self.scan_module_entry_point(ep)):
                    if attr_name == name:
                        matches.setdefault(f"{module_name}:{attr_name}", lambda obj=obj: obj)

        if not matches:
            raise AttributeError(f"plugin package '{package}' has no transformer '{name}'")

        targets = list(matches)
        if len(targets) > 1:
            logger.warning(f"PluginManager: transformer '{package}.{name}' is provided by more than one"
                           f" source ({', '.join(targets)}) - using '{targets[0]}'")
        return matches[targets[0]]()

    def list_plugins(self) -> dict[str, str]:
        """
        Discover transformer plugins from both the entry-point group and local plugin path.

        This is purely a discovery/documentation aid - :func:`PipelineElement.create_new`
        resolves classes by ``module_name``/``class_name`` regardless of whether they are
        registered here. If the same transformer is provided by more than one source with a
        different target, a warning is logged and the first one wins - see
        :func:`resolve_plugin` for the order.

        :return: Mapping of '<plugin package>.<class name>' (see :func:`plugin_package`) to its
            'module_name:class_name' target
        """
        plugins: dict[str, str] = {}

        def register(qualified_name: str, target: str, source: str) -> None:
            existing = plugins.setdefault(qualified_name, target)
            if existing != target:
                logger.warning(
                    f"PluginManager: plugin '{qualified_name}' is registered by more than one"
                    f" source ('{existing}' and '{target}' from {source}) - keeping the first one"
                )

        for module_name, attr_name, _ in self.pipeline_elements(self.load_local_plugins()):
            register(f"{self.plugin_package(module_name)}.{attr_name}", f"{module_name}:{attr_name}",
                     "a local plugin file")

        class_entry_points, module_entry_points = self._entry_points()
        for ep in class_entry_points:
            register(f"{self.plugin_package(ep.value)}.{ep.name}", ep.value, "an entry-point")

        for ep in module_entry_points:
            for module_name, attr_name, _ in self.pipeline_elements(self.scan_module_entry_point(ep)):
                register(f"{self.plugin_package(module_name)}.{attr_name}", f"{module_name}:{attr_name}",
                         f"module entry-point '{ep.name}'")

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
        if requires and requires.get("hint") == "local" and requires.get("package"):
            package = requires["package"]
            return (f"{cls.__name__}.create_new: could not load '{class_name}' from '{module_name}'."
                    f" It was saved as part of the local plugin package '{package}', originally loaded"
                    f" from '{requires.get('path')}'. Register that directory under the same name, e.g."
                    f" {PluginManager.PLUGIN_PATH_ENV}=\"{package}={requires.get('path')}\""
                    f" (currently: {plugin_path}).")

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
        if not expected_version:
            return

        satisfied, specifier_text = _check_version_requirement(installed_version, expected_version)
        if not satisfied:
            logger.warning(
                f"{cls.__name__}.create_new: '{class_name}' was saved with plugin package"
                f" '{distribution}{specifier_text}', but '{installed_version}' is installed."
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

        :return: Mapping of '<plugin package>.<class name>' to its 'module_name:class_name' target
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
            # input_specs is a dict keyed by label ('df' by default, plus e.g. 'other' for a
            # join operator's second input) - label each one when there is more than one
            input_specs = self.input_specs
            show_labels = len(input_specs) > 1
            for label, specs in input_specs.items():
                if show_labels:
                    data += hspace + DEFAULT_INDENT * 3 + label + ":\n"
                    data += DataSpecification.to_str(specs, indent_level=indent_level + 4)
                else:
                    data += DataSpecification.to_str(specs, indent_level=indent_level + 4)

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
                    (np.sin(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_x")
                )
            clone.lazyframe = clone.lazyframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_y")
                )
        return clone

