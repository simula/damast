import importlib.util

import polars
import pytest

from damast.core import engine


@pytest.fixture
def restore_affinity():
    affinity = polars.Config.state().get("POLARS_ENGINE_AFFINITY")
    yield
    polars.Config.set_engine_affinity(affinity)


@pytest.mark.parametrize("devices", ["", "-1"])
def test_gpu_hidden_by_cuda_visible_devices(monkeypatch, devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert not engine.gpu_available()


def test_gpu_requires_cudf_polars(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert not engine.gpu_available()

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert engine.gpu_available()


@pytest.mark.parametrize("gpu", [True, False])
def test_configure_engine_affinity(monkeypatch, restore_affinity, gpu):
    monkeypatch.delenv("POLARS_ENGINE_AFFINITY", raising=False)
    monkeypatch.setattr(engine, "gpu_available", lambda: gpu)
    engine.configure_engine_affinity()
    assert polars.Config.state()["POLARS_ENGINE_AFFINITY"] == ("gpu" if gpu else "streaming")


def test_configure_engine_affinity_keeps_explicit_setting(monkeypatch, restore_affinity):
    polars.Config.set_engine_affinity("in-memory")
    monkeypatch.setenv("POLARS_ENGINE_AFFINITY", "in-memory")
    monkeypatch.setattr(engine, "gpu_available", lambda: True)
    engine.configure_engine_affinity()
    assert polars.Config.state()["POLARS_ENGINE_AFFINITY"] == "in-memory"


def test_default_engine(restore_affinity):
    polars.Config.set_engine_affinity("streaming")
    assert engine.default_engine() == "streaming"

    polars.Config.set_engine_affinity("gpu")
    assert isinstance(engine.default_engine(), polars.GPUEngine)
