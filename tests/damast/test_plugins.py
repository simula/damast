import importlib
import importlib.metadata
import os
import re
import sys
from pathlib import Path

import pytest

import damast.plugins
from damast.core.transformations import PipelineElement, PluginManager, plugin_manager

LOCAL_TRANSFORMER_SOURCE = """
from damast.core.transformations import PipelineElement
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output


class LocalDoubler(PipelineElement):
    @describe("doubles a column")
    @input({"x": {}})
    @output({"x_doubled": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df
"""


def _reset_plugin_manager():
    plugin_manager._unload()
    plugin_manager._registered_packages.clear()


@pytest.fixture
def local_plugin_path(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, str(plugin_dir))

    yield plugin_dir

    _reset_plugin_manager()


def test_getattr_resolves_local_plugin_via_its_package(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    LocalDoubler = damast.plugins.acme_local_transformer.LocalDoubler

    assert LocalDoubler.__name__ == "LocalDoubler"
    assert LocalDoubler.__module__ == "acme_local_transformer"


def test_getattr_resolves_entry_point_plugin_via_its_package(monkeypatch):
    class FakeEntryPoint:
        name = "AcmeTransformer"
        value = "acme_pkg.transformers:AcmeTransformer"

        @staticmethod
        def load():
            return "loaded-acme-transformer"

    def fake_entry_points(*, group):
        assert group == PluginManager.ENTRY_POINT_GROUP
        return [FakeEntryPoint()]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert damast.plugins.acme_pkg.AcmeTransformer == "loaded-acme-transformer"


def test_plain_class_name_is_not_resolvable(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    with pytest.raises(AttributeError, match="no plugin package 'LocalDoubler'"):
        damast.plugins.LocalDoubler
    with pytest.raises(ImportError):
        from damast.plugins import LocalDoubler  # noqa: F401


def test_unknown_plugin_package_raises():
    with pytest.raises(AttributeError, match="no plugin package 'does_not_exist'"):
        damast.plugins.does_not_exist
    with pytest.raises(ModuleNotFoundError, match="No plugin package 'does_not_exist'"):
        importlib.import_module("damast.plugins.does_not_exist")


def test_unknown_transformer_in_known_package_raises(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    with pytest.raises(AttributeError, match="'acme_local_transformer' has no transformer 'DoesNotExist'"):
        damast.plugins.acme_local_transformer.DoesNotExist


def test_same_class_name_in_two_packages_does_not_clash(tmp_path, monkeypatch, caplog):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "acme_transformer_a.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    (dir_b / "acme_transformer_b.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, os.pathsep.join([str(dir_a), str(dir_b)]))
    plugin_manager.reload()

    with caplog.at_level("WARNING"):
        from damast.plugins.acme_transformer_a import LocalDoubler as DoublerA
        from damast.plugins.acme_transformer_b import LocalDoubler as DoublerB

    assert DoublerA.__module__ == "acme_transformer_a"
    assert DoublerB.__module__ == "acme_transformer_b"
    assert not any("more than one source" in record.message for record in caplog.records)

    _reset_plugin_manager()


def test_dir_lists_plugin_packages_and_their_transformers(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    assert "acme_local_transformer" in dir(damast.plugins)
    assert dir(damast.plugins.acme_local_transformer) == ["LocalDoubler"]


def test_from_import_syntax_resolves_local_plugin(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    from damast.plugins.acme_local_transformer import LocalDoubler

    assert LocalDoubler.__module__ == "acme_local_transformer"


# --- named local plugin packages (DAMAST_PLUGIN_PATH="name=path") ---------------------------

NAMED_MAIN_SOURCE = LOCAL_TRANSFORMER_SOURCE + """
from . import helpers
from .sub import VALUE
"""

NAMED_PACKAGE_FILES = {
    "main.py": NAMED_MAIN_SOURCE,
    "helpers.py": "FACTOR = 2\n",
    "_private.py": "raise RuntimeError('must not be imported by the scan')\n",
    "sub/__init__.py": "from .util import VALUE\n",
    "sub/util.py": "VALUE = 42\n",
}


def _write_files(root, files: dict[str, str]):
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


@pytest.fixture
def named_plugin_dir(tmp_path, monkeypatch):
    plugin_dir = _write_files(tmp_path / "transformers", NAMED_PACKAGE_FILES)
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, f"acme_named={plugin_dir}")
    plugin_manager.reload()

    yield plugin_dir

    _reset_plugin_manager()


def test_named_plugin_dir_is_loaded_as_package_with_relative_imports(named_plugin_dir):
    assert PipelineElement.list_plugins()["acme_named.LocalDoubler"] == "acme_named.main:LocalDoubler"

    main = sys.modules["acme_named.main"]
    assert main.VALUE == 42
    # the sibling imported via 'from . import helpers' is the very module the scan registered
    assert main.helpers is plugin_manager.local_modules["acme_named.helpers"]
    assert "acme_named._private" not in sys.modules
    assert plugin_manager.local_packages == {"acme_named": named_plugin_dir}


def test_named_plugin_dir_resolvable_via_damast_plugins_and_create_new(named_plugin_dir):
    from damast.plugins.acme_named import LocalDoubler

    assert LocalDoubler.__module__ == "acme_named.main"

    instance = PipelineElement.create_new(module_name="acme_named.main", class_name="LocalDoubler")
    assert isinstance(instance, LocalDoubler)
    assert dict(instance)["requires"] == {"hint": "local", "package": "acme_named", "path": str(named_plugin_dir)}


def test_named_plugin_dir_with_init_runs_it_as_the_package(tmp_path, monkeypatch):
    plugin_dir = _write_files(tmp_path / "transformers", {"__init__.py": LOCAL_TRANSFORMER_SOURCE})
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, f"acme_named_init={plugin_dir}")
    plugin_manager.reload()

    assert PipelineElement.list_plugins()["acme_named_init.LocalDoubler"] == "acme_named_init:LocalDoubler"
    assert sys.modules["acme_named_init"].__path__ == [str(plugin_dir)]

    _reset_plugin_manager()


def test_missing_named_plugin_package_error_suggests_name_and_path(named_plugin_dir, monkeypatch):
    saved_step = dict(PipelineElement.create_new(module_name="acme_named.main", class_name="LocalDoubler"))

    # simulate loading the pipeline elsewhere, where this package is not registered
    monkeypatch.delenv(PluginManager.PLUGIN_PATH_ENV)
    plugin_manager.reload()

    with pytest.raises(ImportError, match=re.escape(f'DAMAST_PLUGIN_PATH="acme_named={named_plugin_dir}"')):
        PipelineElement.create_new(**saved_step)


def test_register_plugin_package(tmp_path):
    plugin_dir = _write_files(tmp_path / "transformers", NAMED_PACKAGE_FILES)
    plugin_manager.register_plugin_package("acme_registered", plugin_dir)

    assert PipelineElement.list_plugins()["acme_registered.LocalDoubler"] == "acme_registered.main:LocalDoubler"

    # registrations survive a reload
    plugin_manager.reload()
    assert "acme_registered.main" in plugin_manager.local_modules

    _reset_plugin_manager()


def test_register_plugin_package_rejects_invalid_name(tmp_path):
    with pytest.raises(ValueError, match="not a valid plugin package name"):
        plugin_manager.register_plugin_package("acme-registered", tmp_path)


@pytest.mark.parametrize(["name", "message"], [
    ["class", "not a valid package name"],
    ["json", "already importable"],
])
def test_named_plugin_dir_invalid_or_importable_name_is_skipped(name, message, tmp_path, monkeypatch, caplog):
    plugin_dir = _write_files(tmp_path / "transformers", NAMED_PACKAGE_FILES)
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, f"{name}={plugin_dir}")

    with caplog.at_level("WARNING"):
        plugin_manager.reload()

    assert plugin_manager.local_packages == {}
    assert any(message in record.message for record in caplog.records)

    _reset_plugin_manager()


def test_named_plugin_dir_duplicate_name_warns_and_keeps_first(tmp_path, monkeypatch, caplog):
    dir_a = _write_files(tmp_path / "a", NAMED_PACKAGE_FILES)
    dir_b = _write_files(tmp_path / "b", NAMED_PACKAGE_FILES)
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV,
                       os.pathsep.join([f"acme_dup={dir_a}", f"acme_dup={dir_b}"]))

    with caplog.at_level("WARNING"):
        plugin_manager.reload()

    assert plugin_manager.local_packages == {"acme_dup": dir_a}
    assert any("collides with already loaded" in record.message for record in caplog.records)

    _reset_plugin_manager()


def test_named_plugin_dir_reload_picks_up_edits(named_plugin_dir):
    main_file = named_plugin_dir / "main.py"
    main_file.write_text(NAMED_MAIN_SOURCE.replace("LocalDoubler", "LocalTripler"))
    plugin_manager.reload()

    plugins = PipelineElement.list_plugins()
    assert plugins["acme_named.LocalTripler"] == "acme_named.main:LocalTripler"
    assert "acme_named.LocalDoubler" not in plugins


def test_unnamed_plugin_dir_is_deprecated_but_still_loads_alongside_named(tmp_path, monkeypatch, caplog):
    named_dir = _write_files(tmp_path / "named", NAMED_PACKAGE_FILES)
    flat_dir = _write_files(tmp_path / "flat", {"acme_flat.py": LOCAL_TRANSFORMER_SOURCE.replace(
        "LocalDoubler", "FlatDoubler")})
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, os.pathsep.join([f"acme_mixed={named_dir}", str(flat_dir)]))
    plugin_manager._warned_unnamed.clear()

    with caplog.at_level("WARNING"):
        plugins = PipelineElement.list_plugins()

    assert plugins["acme_mixed.LocalDoubler"] == "acme_mixed.main:LocalDoubler"
    assert plugins["acme_flat.FlatDoubler"] == "acme_flat:FlatDoubler"
    assert any("is deprecated" in record.message for record in caplog.records)

    _reset_plugin_manager()


# --- module entry-points ('name = "pkg.module"') -------------------------------------------

MODULE_ENTRY_FILES = {
    "__init__.py": "",
    "a.py": LOCAL_TRANSFORMER_SOURCE.replace("LocalDoubler", "ModuleDoubler"),
    "reexport.py": "from .a import ModuleDoubler\n",
    "broken.py": "raise RuntimeError('broken plugin module')\n",
}


@pytest.fixture
def installed_package(tmp_path, monkeypatch):
    """Create an importable package, standing in for an installed distribution."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    monkeypatch.syspath_prepend(str(site_dir))
    created = []

    def make(name: str, files: dict[str, str]):
        created.append(name)
        _write_files(site_dir / name, files)
        importlib.invalidate_caches()

    yield make

    for module_name in [m for m in sys.modules if m.split(".")[0] in created]:
        sys.modules.pop(module_name)
    _reset_plugin_manager()


@pytest.fixture
def fake_entry_points(monkeypatch):
    entry_points: list[importlib.metadata.EntryPoint] = []

    def add(name: str, value: str):
        entry_points.append(importlib.metadata.EntryPoint(name=name, value=value,
                                                          group=PluginManager.ENTRY_POINT_GROUP))

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: list(entry_points))
    return add


def test_module_entry_point_for_package_registers_classes_of_its_submodules(
        installed_package, fake_entry_points, caplog):
    installed_package("acme_mod_pkg", MODULE_ENTRY_FILES)
    fake_entry_points("acme_mod_pkg", "acme_mod_pkg")

    with caplog.at_level("WARNING"):
        plugins = PipelineElement.list_plugins()

    # defined in 'a', only re-exported by 'reexport' - listed once, without an ambiguity warning
    assert plugins["acme_mod_pkg.ModuleDoubler"] == "acme_mod_pkg.a:ModuleDoubler"
    assert not any("more than one source" in record.message for record in caplog.records)
    assert any("acme_mod_pkg.broken" in record.message for record in caplog.records)


def test_module_entry_point_for_single_module_skips_reexported_classes(installed_package, fake_entry_points):
    installed_package("acme_mod_single", MODULE_ENTRY_FILES)
    fake_entry_points("acme_mod_single", "acme_mod_single.reexport")

    assert "acme_mod_single.ModuleDoubler" not in PipelineElement.list_plugins()


def test_module_entry_point_resolvable_via_damast_plugins(installed_package, fake_entry_points):
    installed_package("acme_mod_lookup", MODULE_ENTRY_FILES)
    fake_entry_points("acme_mod_lookup", "acme_mod_lookup.a")

    from damast.plugins.acme_mod_lookup import ModuleDoubler

    assert ModuleDoubler.__module__ == "acme_mod_lookup.a"


def test_module_entry_point_name_is_only_a_label(installed_package, fake_entry_points, caplog):
    """The namespace is the top-level package; a different entry-point name is warned about, once."""
    installed_package("acme_mod_label", MODULE_ENTRY_FILES)
    fake_entry_points("acme_label", "acme_mod_label")

    with caplog.at_level("WARNING"):
        packages = plugin_manager.plugin_packages()
        plugin_manager.plugin_packages()

    assert "acme_mod_label" in packages
    assert "acme_label" not in packages
    warnings = [r.message for r in caplog.records if "is ignored" in r.message]
    assert len(warnings) == 1
    assert "'acme_label = acme_mod_label'" in warnings[0]
    assert "damast.plugins.acme_mod_label" in warnings[0]

    from damast.plugins.acme_mod_label import ModuleDoubler
    assert ModuleDoubler.__module__ == "acme_mod_label.a"
    with pytest.raises(ModuleNotFoundError, match="No plugin package 'acme_label'"):
        import damast.plugins.acme_label  # noqa: F401


def test_module_entry_point_named_after_its_package_is_not_warned_about(
        installed_package, fake_entry_points, caplog):
    installed_package("acme_mod_named", MODULE_ENTRY_FILES)
    fake_entry_points("acme_mod_named", "acme_mod_named.a")

    with caplog.at_level("WARNING"):
        plugin_manager.plugin_packages()

    assert not any("is ignored" in record.message for record in caplog.records)


def test_class_entry_point_wins_over_module_entry_point_of_same_package(installed_package, fake_entry_points):
    installed_package("acme_mod_class", MODULE_ENTRY_FILES)
    installed_package("acme_mod_lazy", MODULE_ENTRY_FILES)
    fake_entry_points("ModuleDoubler", "acme_mod_class.a:ModuleDoubler")
    fake_entry_points("acme_mod_class", "acme_mod_class")
    fake_entry_points("acme_mod_lazy", "acme_mod_lazy")

    # answered by the class entry-point: no module entry-point package gets scanned/imported
    assert damast.plugins.acme_mod_class.ModuleDoubler.__module__ == "acme_mod_class.a"
    assert "acme_mod_class.reexport" not in sys.modules
    assert "acme_mod_lazy" not in sys.modules

    # both entries point at the very same class - listed once
    assert PipelineElement.list_plugins()["acme_mod_class.ModuleDoubler"] == "acme_mod_class.a:ModuleDoubler"


def test_same_class_name_twice_within_one_package_warns_and_keeps_first(tmp_path, monkeypatch, caplog):
    plugin_dir = _write_files(tmp_path / "transformers", {
        "main.py": LOCAL_TRANSFORMER_SOURCE,
        "main2.py": LOCAL_TRANSFORMER_SOURCE,
    })
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, f"acme_twice={plugin_dir}")
    plugin_manager.reload()

    with caplog.at_level("WARNING"):
        plugins = PipelineElement.list_plugins()
        LocalDoubler = damast.plugins.acme_twice.LocalDoubler

    assert plugins["acme_twice.LocalDoubler"] == "acme_twice.main:LocalDoubler"
    assert LocalDoubler.__module__ == "acme_twice.main"
    assert any("registered by more than one source" in record.message for record in caplog.records)
    assert any("provided by more than one source" in record.message for record in caplog.records)

    _reset_plugin_manager()


def test_pipeline_saved_from_named_plugin_dir_replays_against_installed_package(
        named_plugin_dir, installed_package, fake_entry_points, monkeypatch):
    saved_step = dict(PipelineElement.create_new(module_name="acme_named.main", class_name="LocalDoubler"))

    # elsewhere: the same code is installed as package 'acme_named' instead of a local directory
    monkeypatch.delenv(PluginManager.PLUGIN_PATH_ENV)
    plugin_manager.reload()
    installed_package("acme_named", NAMED_PACKAGE_FILES)
    fake_entry_points("acme_named", "acme_named")

    instance = PipelineElement.create_new(**saved_step)
    assert type(instance).__module__ == "acme_named.main"
    assert "site" in Path(sys.modules["acme_named.main"].__file__).parts
