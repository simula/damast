import json

import pytest

import pandas as pd
from damast.data_handling.pipeline import Pipeline
from damast.data_handling.transformers.base import BaseTransformer
from damast.data_handling.transformers.filters import MinMaxFilter
from damast.data_handling.transformers.sorters import GenericSorter


def test_pipeline(tmp_path):
    data = [
        [1, "b"],
        [3, "c"],
        [0, "a"]
    ]
    df = pd.DataFrame(data, columns=["id", "name"])
    pipeline = Pipeline([
        ("sort_by_id", GenericSorter(column_names=["id"])),
        ("min_max_filter", MinMaxFilter(min=1, max=3, column_name="id"))
    ])

    with pytest.raises(RuntimeError) as e:
        pipeline.get_stats()

    pipeline.fit_transform(df)
    stats = pipeline.get_stats()

    assert len(stats) == 2
    assert stats[0][0] == "sort_by_id"
    assert stats[1][0] == "min_max_filter"

    sort_stats = stats[0][1]
    assert BaseTransformer.RUNTIME_IN_S in sort_stats
    assert BaseTransformer.INPUT_COLUMNS in sort_stats
    assert BaseTransformer.INPUT_SHAPE in sort_stats
    assert BaseTransformer.OUTPUT_COLUMNS in sort_stats
    assert BaseTransformer.OUTPUT_SHAPE in sort_stats

    stats_file = tmp_path / "test-pipeline-stats.json"
    pipeline.save_stats(filename=stats_file)
    assert stats_file.exists()

    with open(stats_file, "r") as f:
        loaded_stats = json.loads(f.read())

    for idx, c in enumerate(["sort_by_mmsi", "min_max_filter"]):
        loaded_name, loaded_t_stats = loaded_stats[idx]
        name, t_stats = stats[idx]

        for t in [BaseTransformer.RUNTIME_IN_S,
                  BaseTransformer.INPUT_COLUMNS,
                  BaseTransformer.INPUT_SHAPE,
                  BaseTransformer.OUTPUT_COLUMNS,
                  BaseTransformer.OUTPUT_SHAPE]:
            if type(t_stats[t]) is tuple:
                assert loaded_t_stats[t] == list(t_stats[t])
            else:
                assert loaded_t_stats[t] == t_stats[t]
