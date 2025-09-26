__all__ = [
    "DECORATED_ARTIFACT_SPECS",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS",
]
DAMAST_DEFAULT_DATASOURCE = "df"

DECORATED_DESCRIPTION = "_damast_description"
"""Attribute description for :func:`describe`"""

DECORATED_ARTIFACT_SPECS = "_damast_artifact_specs"
"""Attribute description for :func:`artifacts`"""

DECORATED_INPUT_SPECS = "_damast_input_specs"
"""Attribute description for :func:`input`"""

DECORATED_OUTPUT_SPECS = "_damast_output_specs"
"""Attribute description for :func:`output`"""

DAMAST_HDF5_ROOT: str = "/dataframe"
DAMAST_HDF5_COLUMNS: str = "/dataframe/columns"

DAMAST_SPEC_SUFFIX: str = ".spec.yaml"
DAMAST_SUPPORTED_FILE_FORMATS: dict[str, list[str]] = {
    "parquet": [".parquet", ".pq"],
    "netcdf": [".nc", ".netcdf", ".nc4", ".netcdf4"],
    "hdf": [".h5", ".hdf", ".hdf5"],
    "csv": [".csv"],
}

DAMAST_MOUNT_PREFIX: str = "damast-mount"
