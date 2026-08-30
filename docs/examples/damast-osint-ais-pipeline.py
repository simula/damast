"""
Build and save a pipeline with two named datasources - 'df' for AIS pings, 'osint_events' for
OSINT reports - joined by timestamp. Inspired by damast-examples/hozint-ais/ais_osint_fusion.py,
trimmed to a self-contained example using only the damast core API.

Usage:
    python docs/examples/damast-osint-ais-generate-data.py
    python docs/examples/damast-osint-ais-pipeline.py
    damast process --pipeline pipelines/osint_ais_preparation.damast.ppl \\
        --input-data df=docs/examples/data/ais.parquet \\
        --input-data osint_events=docs/examples/data/osint.parquet \\
        --output-file output.parquet
"""
import os
from pathlib import Path

# JoinByTimestamp lives in its own package/file in a real project - see docs/cli.rst >
# Plugins. Here it's a local plugin file, resolved the same way.
os.environ["DAMAST_PLUGIN_PATH"] = str(Path(__file__).parent / "plugins")

from damast.core.dataprocessing import DataProcessingPipeline
from damast.plugins import JoinByTimestamp

pipeline = DataProcessingPipeline(name="osint_ais_preparation",
                                  description="Join AIS pings with OSINT events by timestamp",
                                  base_dir=".") \
    .join("osint_events", JoinByTimestamp(),
          name_mappings={
              "df": {"timestamp": "date_time_utc"},
              "other": {"timestamp": "timestamp", "lat": "latitude", "lon": "longitude"},
          })

pipeline_path = pipeline.save("pipelines")
print(f"Saved pipeline to {pipeline_path}")
