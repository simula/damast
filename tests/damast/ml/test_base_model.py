import inspect
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from damast.ml import keras
from damast.ml.models.base import BaseModel

supports_splines = "splines" in inspect.signature(keras.utils.plot_model).parameters


class StackedAttentionModel(BaseModel):
    """Two residual self-attention blocks - a graph graphviz' orthogonal edge routing fails on."""
    input_specs = {"a": {"length": 1}, "b": {"length": 1}}
    output_specs = {"a": {"length": 1}, "b": {"length": 1}}

    def _init_model(self):
        inputs = keras.Input(shape=(5, len(self.features)))
        x = keras.layers.LayerNormalization()(keras.layers.Dense(8)(inputs))
        for _ in range(2):
            attention = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)(x, x)
            x = keras.layers.LayerNormalization()(x + attention)
            feed_forward = keras.layers.Dense(16, activation="relu")(x)
            feed_forward = keras.layers.Dropout(0.1)(keras.layers.Dense(8)(feed_forward))
            x = keras.layers.LayerNormalization()(x + feed_forward)
        outputs = keras.layers.Dense(len(self.targets))(keras.layers.GlobalAveragePooling1D()(x))
        self.model = keras.Model(inputs=inputs, outputs=outputs)


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz is not installed")
@pytest.mark.skipif(not supports_splines, reason="this keras version cannot select the graphviz edge style")
def test_plot_stacked_residual_blocks(tmp_path):
    model = StackedAttentionModel(name="stacked", features=["a", "b"], targets=["a", "b"], output_dir=tmp_path)

    assert model.plot().is_file()


@pytest.mark.skipif(shutil.which("dot") is not None, reason="graphviz is installed, so plotting works")
def test_plot_without_graphviz_does_not_raise(tmp_path):
    """The model plot is documentation only - a failing plot must not fail build()/training."""
    model = StackedAttentionModel(name="stacked", features=["a", "b"], targets=["a", "b"], output_dir=tmp_path)

    assert model.plot().name == "stacked.png"


@pytest.mark.parametrize(
    "which, run, expected",
    [
        (lambda _: None, None, False),                                   # graphviz not installed
        (lambda _: "/usr/bin/dot", lambda *a, **kw: MagicMock(returncode=0), True),
        (lambda _: "/usr/bin/dot", lambda *a, **kw: MagicMock(returncode=1), False),
        (lambda _: "/usr/bin/dot",
         lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("dot", 10)), False),
    ],
    ids=["missing", "answers", "fails", "hangs"],
)
def test_graphviz_is_usable(monkeypatch, which, run, expected):
    """A graphviz that does not answer must be reported unusable rather than block the caller -
    `keras.utils.plot_model` probes it with a subprocess that can hang (see damast #windows-ci)."""
    from damast.ml.models import base

    base.graphviz_is_usable.cache_clear()
    monkeypatch.setattr(base.shutil, "which", which)
    if run is not None:
        monkeypatch.setattr(base.subprocess, "run", run)

    try:
        assert base.graphviz_is_usable() is expected
    finally:
        base.graphviz_is_usable.cache_clear()


def test_the_training_history_is_written_after_every_epoch(tmp_path):
    """So that a running training can be followed - and compared with others - before it ends: a
    training that breaks off still leaves the epochs it finished."""
    import numpy as np
    import polars as pl

    from damast.ml.models.base import HISTORY_FILENAME

    class SmallModel(BaseModel):
        input_specs = {"a": {"length": 1}}
        output_specs = {"a": {"length": 1}}

        def _init_model(self):
            inputs = keras.Input(shape=(5, 1))
            self.model = keras.Model(inputs, keras.layers.Dense(1)(keras.layers.Flatten()(inputs)))

    def batches(limit=None):
        rng = np.random.default_rng(0)
        # float32: the torch backend on macOS (MPS) does not support float64
        for k in range(limit or 10**9):
            yield rng.random((4, 5, 1), dtype=np.float32), rng.random((4, 1), dtype=np.float32)
        raise RuntimeError("the data ran out")

    model = SmallModel(name="small", features=["a"], targets=["a"], output_dir=tmp_path)
    model.build()
    history_file = model.model_dir / HISTORY_FILENAME

    # 4 batches per epoch: drawing the 11th batch fails. Keras may read up to 2 batches ahead (jax
    # backend in older keras), so the failure lands in the 3rd epoch with or without read-ahead
    with pytest.raises(RuntimeError, match="Training of small failed") as failure:
        model.train(batches(limit=10), batches(), epochs=5, steps_per_epoch=4, validation_steps=1)
    assert "the data ran out" in str(failure.value.__cause__)
    history = pl.read_csv(history_file)
    assert history["epoch"].to_list() == [0, 1]
    assert "val_loss" in history.columns

    # continued training - from the last best checkpoint, as after a crash - appends to the history.
    # Reloading is also required in-process: the jax backend leaves the model without weights when
    # fit() raises at the start of an epoch
    model.load_best_weights()
    model.train(batches(), batches(), epochs=3, initial_epoch=2, steps_per_epoch=2, validation_steps=1)
    assert pl.read_csv(history_file)["epoch"].to_list() == [0, 1, 2]
