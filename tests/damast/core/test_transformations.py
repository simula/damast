import importlib.metadata
import os
import sys

import pytest

import damast
from damast.core.transformations import (
    PipelineElement,
    PluginManager,
    plugin_manager,
)
from damast.data_handling.transformers.cycle_transformer import CycleTransformer

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
    for module_name in list(plugin_manager.local_files):
        sys.modules.pop(module_name, None)
    plugin_manager._local_modules.clear()
    plugin_manager._local_files.clear()
    plugin_manager._requirement_cache.clear()
    plugin_manager._loaded = False


@pytest.fixture
def local_plugin_path(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, str(plugin_dir))

    yield plugin_dir

    _reset_plugin_manager()

# CycleTransformer is a built-in PipelineElement whose module lives inside the
# installed 'damast' distribution - used here as a stand-in for a transformer
# provided by any installed (plugin) package.


def test_resolve_requirement_for_installed_package():
    requirement = plugin_manager.resolve_requirement(CycleTransformer.__module__)
    assert requirement == {"distribution": "damast", "version": damast.version.__version__}


def test_resolve_requirement_unresolvable_module():
    assert plugin_manager.resolve_requirement("this_module_does_not_exist_anywhere") is None


def test_pipeline_element_iter_includes_requires():
    data = dict(CycleTransformer(n=1))
    assert data["requires"] == {"distribution": "damast", "version": damast.version.__version__}


def test_create_new_roundtrip_with_matching_requirement():
    step = dict(CycleTransformer(n=1))
    instance = PipelineElement.create_new(**step)
    assert isinstance(instance, CycleTransformer)


def test_create_new_missing_plugin_package_raises_actionable_error():
    step = dict(CycleTransformer(n=1))
    step["requires"] = {"distribution": "acme-damast-plugins", "version": "1.2.3"}

    with pytest.raises(ImportError, match="pip install acme-damast-plugins==1.2.3"):
        PipelineElement.create_new(**step)


def test_create_new_version_mismatch_warns_but_loads(caplog):
    step = dict(CycleTransformer(n=1))
    step["requires"] = {"distribution": "damast", "version": "0.0.0-does-not-match"}

    with caplog.at_level("WARNING"):
        instance = PipelineElement.create_new(**step)

    assert isinstance(instance, CycleTransformer)
    assert any("0.0.0-does-not-match" in record.message for record in caplog.records)


def test_list_plugins_discovers_entry_points(monkeypatch):
    class FakeEntryPoint:
        name = "AcmeTransformer"
        value = "acme_pkg.transformers:AcmeTransformer"

    def fake_entry_points(*, group):
        assert group == PluginManager.ENTRY_POINT_GROUP
        return [FakeEntryPoint()]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert PipelineElement.list_plugins() == {"AcmeTransformer": "acme_pkg.transformers:AcmeTransformer"}


def test_list_plugins_empty_by_default():
    assert PipelineElement.list_plugins() == {}


def test_local_plugin_path_discovered_via_list_plugins(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    PipelineElement.reload_plugins()

    plugins = PipelineElement.list_plugins()
    assert plugins["LocalDoubler"] == "acme_local_transformer:LocalDoubler"


def test_local_plugin_path_resolvable_via_create_new(local_plugin_path):
    (local_plugin_path / "acme_local_transformer2.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    PipelineElement.reload_plugins()

    instance = PipelineElement.create_new(module_name="acme_local_transformer2", class_name="LocalDoubler")
    assert instance.__class__.__name__ == "LocalDoubler"

    # a locally-loaded transformer is not backed by an installable distribution, but is
    # flagged via a 'hint' so it is clear (and traceable) that it came from DAMAST_PLUGIN_PATH
    expected_path = local_plugin_path / "acme_local_transformer2.py"
    assert dict(instance)["requires"] == {"hint": "local", "path": str(expected_path)}


def test_missing_local_plugin_error_mentions_original_path(local_plugin_path):
    plugin_file = local_plugin_path / "acme_local_transformer3.py"
    plugin_file.write_text(LOCAL_TRANSFORMER_SOURCE)
    PipelineElement.reload_plugins()

    step = PipelineElement.create_new(module_name="acme_local_transformer3", class_name="LocalDoubler")
    saved_step = dict(step)
    assert saved_step["requires"] == {"hint": "local", "path": str(plugin_file)}

    # simulate loading the pipeline elsewhere, where this local plugin file is unavailable
    sys.modules.pop("acme_local_transformer3", None)
    plugin_manager._local_modules.pop("acme_local_transformer3", None)
    plugin_manager._local_files.pop("acme_local_transformer3", None)

    with pytest.raises(ImportError, match="originally loaded from"):
        PipelineElement.create_new(**saved_step)


def test_local_plugin_path_missing_directory_warns_but_does_not_crash(tmp_path, monkeypatch, caplog):
    missing_dir = tmp_path / "does-not-exist"
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, str(missing_dir))

    with caplog.at_level("WARNING"):
        PipelineElement.reload_plugins()

    assert PipelineElement.list_plugins() == {}
    assert any("is not a directory" in record.message for record in caplog.records)

    _reset_plugin_manager()


def test_local_plugin_path_name_collision_warns_and_keeps_first(tmp_path, monkeypatch, caplog):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "same_name.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    (dir_b / "same_name.py").write_text(LOCAL_TRANSFORMER_SOURCE)

    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, os.pathsep.join([str(dir_a), str(dir_b)]))

    with caplog.at_level("WARNING"):
        PipelineElement.reload_plugins()

    assert any("collides with" in record.message for record in caplog.records)
    assert plugin_manager.local_files["same_name"] == dir_a / "same_name.py"

    _reset_plugin_manager()


def test_create_new_missing_local_module_error_mentions_plugin_path(monkeypatch):
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, "/some/configured/path")

    with pytest.raises(ImportError, match=PluginManager.PLUGIN_PATH_ENV):
        PipelineElement.create_new(module_name="totally_missing_local_module", class_name="Foo")


def test_plugin_manager_is_a_separate_instantiable_class():
    manager = PluginManager()
    assert manager is not plugin_manager
    assert manager.list_plugins() == {}
    assert manager.local_files == {}
