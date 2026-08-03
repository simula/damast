import importlib.metadata
import os
import sys

import pytest

import damast.plugins
from damast.core.transformations import PluginManager, plugin_manager

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


def test_getattr_resolves_local_plugin(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    LocalDoubler = damast.plugins.LocalDoubler

    assert LocalDoubler.__name__ == "LocalDoubler"
    assert LocalDoubler.__module__ == "acme_local_transformer"


def test_getattr_resolves_entry_point_plugin(monkeypatch):
    class FakeEntryPoint:
        name = "AcmeTransformer"

        @staticmethod
        def load():
            return "loaded-acme-transformer"

    def fake_entry_points(*, group):
        assert group == PluginManager.ENTRY_POINT_GROUP
        return [FakeEntryPoint()]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert damast.plugins.AcmeTransformer == "loaded-acme-transformer"


def test_getattr_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no plugin named 'DoesNotExist'"):
        damast.plugins.DoesNotExist


def test_local_plugin_takes_precedence_over_same_named_entry_point(local_plugin_path, monkeypatch):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    class FakeEntryPoint:
        name = "LocalDoubler"
        value = "acme_pkg.transformers:LocalDoubler"

        @staticmethod
        def load():
            return "should-not-be-used"

    monkeypatch.setattr(importlib.metadata, "entry_points",
                         lambda *, group: [FakeEntryPoint()])

    assert damast.plugins.LocalDoubler.__module__ == "acme_local_transformer"


def test_getattr_warns_on_ambiguous_local_plugins(tmp_path, monkeypatch, caplog):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "acme_transformer_a.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    (dir_b / "acme_transformer_b.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    monkeypatch.setenv(PluginManager.PLUGIN_PATH_ENV, os.pathsep.join([str(dir_a), str(dir_b)]))
    plugin_manager.reload()

    with caplog.at_level("WARNING"):
        LocalDoubler = damast.plugins.LocalDoubler

    assert LocalDoubler.__module__ == "acme_transformer_a"
    assert any("is ambiguous" in record.message for record in caplog.records)

    _reset_plugin_manager()


def test_getattr_warns_on_local_and_entry_point_collision(local_plugin_path, monkeypatch, caplog):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    class FakeEntryPoint:
        name = "LocalDoubler"
        value = "acme_pkg.transformers:LocalDoubler"

        @staticmethod
        def load():
            return "should-not-be-used"

    monkeypatch.setattr(importlib.metadata, "entry_points",
                         lambda *, group: [FakeEntryPoint()])

    with caplog.at_level("WARNING"):
        LocalDoubler = damast.plugins.LocalDoubler

    assert LocalDoubler.__module__ == "acme_local_transformer"
    assert any("is ambiguous" in record.message for record in caplog.records)


def test_dir_lists_discovered_plugins(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    assert "LocalDoubler" in dir(damast.plugins)


def test_from_import_syntax_resolves_local_plugin(local_plugin_path):
    (local_plugin_path / "acme_local_transformer.py").write_text(LOCAL_TRANSFORMER_SOURCE)
    plugin_manager.reload()

    from damast.plugins import LocalDoubler

    assert LocalDoubler.__module__ == "acme_local_transformer"
