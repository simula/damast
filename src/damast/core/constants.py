DAMAST_HDF5_ROOT: str = "/dataframe"
DAMAST_HDF5_COLUMNS: str = "/dataframe/columns"

DAMAST_SPEC_SUFFIX: str = ".spec.yaml"
DAMAST_SUPPORTED_FILE_FORMATS: dict[str, list[str]] = {
    "parquet": [".parquet", ".pq"],
    "netcdf": [".nc", ".netcdf", ".nc4", ".netcdf4"],
    "hdf": [".h5", ".hdf", ".hdf5"],
    "csv": [".csv"],
}
