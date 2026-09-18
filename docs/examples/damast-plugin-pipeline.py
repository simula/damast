import os
from pathlib import Path

# DAMAST_PLUGIN_PATH registers the directory holding my_transformers.py as package
# 'my_plugins', so that 'MyTripler' becomes resolvable via damast.plugins.my_plugins - see
# docs/examples/plugins/.
os.environ["DAMAST_PLUGIN_PATH"] = f"my_plugins={Path(__file__).parent / 'plugins'}"

from damast.core import DataProcessingPipeline
from damast.plugins.my_plugins import MyTripler

pipeline = DataProcessingPipeline(name="my-plugin-pipeline", base_dir=".")
pipeline.add("Triple mmsi",
             MyTripler(),
             name_mappings={"x": "mmsi"})

pipeline.save("pipelines")
