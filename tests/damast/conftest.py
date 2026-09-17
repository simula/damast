import sys
from pathlib import Path

import pytest

from damast.core.transformations import PluginManager, plugin_manager


@pytest.fixture
def data_path():
    return Path(__file__).parent / "data"

@pytest.fixture
def isolate_plugins(monkeypatch):
    """
    Isolate from ambient state, rather than assuming a clean environment: DAMAST_PLUGIN_PATH
    may already be set outside this test process, an installed package may register a real
    'damast.transformers' entry-point, and plugin_manager is a process-wide singleton other
    tests may have already populated - see _reset_plugin_manager() in tests/damast/test_plugins.py
    for the same pattern.
    """
    import importlib.metadata as importlib_metadata

    monkeypatch.delenv(PluginManager.PLUGIN_PATH_ENV, raising=False)

    # Only fake out the 'damast.transformers' group - script_runner itself resolves the
    # 'damast' console script via entry_points(group='console_scripts', ...), which must
    # keep working.
    real_entry_points = importlib_metadata.entry_points

    def fake_entry_points(*, group=None, **kwargs):
        if group == PluginManager.ENTRY_POINT_GROUP and not kwargs:
            return []
        return real_entry_points(group=group, **kwargs)

    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)

    for module_name in list(plugin_manager.local_files):
        sys.modules.pop(module_name, None)
    plugin_manager._local_modules.clear()
    plugin_manager._local_files.clear()
    plugin_manager._requirement_cache.clear()
    plugin_manager._loaded = False
