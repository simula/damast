"""
Generate small synthetic AIS and OSINT datasets for the multi-datasource pipeline example -
see damast-osint-ais-pipeline.py and docs/cli.rst > Process > Multiple input datasources.

Inspired by damast-examples/hozint-ais/ais_osint_fusion.py, trimmed to what's needed here and
using only generated/synthetic data - no real datasets are shipped with damast.
"""
from pathlib import Path

import polars

import damast.domains.maritime.ais.data_generator as generator

output_dir = Path(__file__).parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

ais_data = generator.AISTestData(number_of_trajectories=5, min_length=10, max_length=20)
ais_df = ais_data.dataframe
ais_df.write_parquet(output_dir / "ais.parquet")

# two OSINT "events" derived from real AIS pings, so the join in the example pipeline has
# actual matches to show, plus one deliberately unmatched event
sample = ais_df.filter(polars.col("date_time_utc").is_not_null()).sample(2, seed=1)
matched_events = sample.select(
    timestamp="date_time_utc", latitude="lat", longitude="lon",
).with_columns(event_type=polars.Series(["sighting", "distress-call"]))

unrelated_event = polars.DataFrame({
    "timestamp": ["1999-01-01 00:00:00"],
    "latitude": [0.0],
    "longitude": [0.0],
    "event_type": ["unrelated"],
})

osint_df = polars.concat([matched_events, unrelated_event])
osint_df.write_parquet(output_dir / "osint.parquet")

print(f"Wrote {output_dir / 'ais.parquet'} ({len(ais_df)} rows)")
print(f"Wrote {output_dir / 'osint.parquet'} ({len(osint_df)} rows)")
