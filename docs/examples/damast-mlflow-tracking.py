from pathlib import Path

import polars
from astropy import units

from damast.core.data_description import MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.metadata import DataSpecification, MetaData, ValidationMode
from damast.integrations.mlflow_tracker import track_pipeline

pipeline = DataProcessingPipeline.load(Path("pipelines") / "my-pipeline.damast.ppl")

df = polars.DataFrame({"lat": [59.9], "lon": [10.7]})
metadata = MetaData(columns=[
    DataSpecification(name="lat", unit=units.deg, value_range=MinMax(-90.0, 90.0)),
    DataSpecification(name="lon", unit=units.deg, value_range=MinMax(-180.0, 180.0)),
])
adf = AnnotatedDataFrame(df, metadata, validation_mode=ValidationMode.UPDATE_METADATA)

# tracking_uri/experiment_name default to mlflow's own defaults (a local ./mlruns dir) if omitted
with track_pipeline(pipeline, run_name="my-pipeline-run") as tracker:
    result = pipeline.transform(adf)
    tracker.log_result(result)
