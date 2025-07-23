import damast
from damast.core import DataProcessingPipeline
from damast.core.dataframe import AnnotatedDataFrame
from damast.data_handling.transformers import AddDeltaTime, DropMissingOrNan
from damast.domains.maritime.transformers.features import DeltaDistance, Speed


class MyPipeline(DataProcessingPipeline):
    def __init__(self,
                 workdir: str | Path,
                 name: str = "my-pipeline",
                 name_mappings: dict[str, str] = {}):
        super().__init__(name=name,
                         base_dir=workdir,
                         name_mappings=name_mappings)

        self.add("Delta Time",
                 AddDeltaTime(),
                 name_mappings={
                     "group": "mmsi",
                     "time_column": "reception_date"
                })

        self.add("Delta Distance",
                 DeltaDistance(x_shift=True, y_shift=True),
                 name_mappings={
                     "group": "mmsi",
                     "sort": "reception_date",
                     "x": "lat",
                     "y": "lon",
                     "out": "delta_distance",
                })

        self.add("Speed",
                 Speed(),
                 name_mappings={
                     "delta_distance": "delta_distance",
                     "delta_time": "delta_time",
                })

pipeline = MyPipeline(workdir=".")
pipeline.save("pipelines")

