import os
from pathlib import Path

# DAMAST_PLUGIN_PATH points at the directory holding my_transformers.py, so that
# 'MyTripler' becomes resolvable via damast.plugins - see docs/examples/plugins/.
os.environ["DAMAST_PLUGIN_PATH"] = str(Path(__file__).parent / "plugins")

from damast.core import DataProcessingPipeline
from damast.plugins import MyTripler

pipeline = DataProcessingPipeline(name="my-plugin-pipeline", base_dir=".")
pipeline.add("Triple mmsi",
             MyTripler(),
             name_mappings={"x": "mmsi"})

pipeline.save("pipelines")
