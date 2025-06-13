Command Line Interface
========================

The command line interface offers a number of workflow simplification that are encapsulated
in sub-commands:

**inspect**
    check metadadata and dataset properties

**convert**
    convert from (zipped) csv, netcdf to parquet (default) or hdf5 (deprecated)

**annotate**
    create metadata file and update dataframe with metadata

**process**
    apply a data pipeline to a dataset


Inspect
--------

.. literalinclude:: ./examples/damast-inspect-help.txt
  :language: none


Inspect allow to identify columns and properties of columns in a given dataset.
The dataset can consists of one or more (zipped) files, either given as list of filenames or using file pattern.

.. highlight:: python

::

    $ damast inspect -f 1.zip

    Subparser: DataInspectParser
    Loading dataframe (1 files) of total size: 0.0 MB
    Creating offset dictionary for /tmp/damast-example/datasets/1.zip ...
    Creating offset dictionary for /tmp/damast-example/datasets/1.zip took 0.00s
    Created mount point at: /tmp/damast-mountqigwlx74/1.zip
    INFO:damast.core.dataframe:Loading parquet: files=[PosixPath('/tmp/damast-mountqigwlx74/1.zip/dataset-1.zst.parquet')]
    WARNING:damast.core.dataframe:/tmp/damast-mountqigwlx74/1.zip/dataset-1.zst.parquet has no (damast) annotations
    INFO:damast.core.dataframe:No metadata provided or found in files - searching now for an existing spec file
    INFO:damast.core.dataframe:Found no candidate for a spec file
    INFO:damast.core.dataframe:Metadata is not available and not required, so inferring annotation
    Extract str and categorical column metadata: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 1092.51it/s]
    Extract numeric column metadata: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:00<00:00, 12767.03it/s]
    INFO:damast.core.dataframe:Metadata inferring completed
    Annotations:
        accuracy:
            is_optional: False
            representation_type: Boolean
        call_sign:
            is_optional: False
            representation_type: String
            value_range: {'ListOfValues': [None, '', 'SIDF9', 'SABD4', 'STDL5', 'STJE3', 'SKCY7', 'XAGBE']}
        cog:
            is_optional: False
            representation_type: Float32
            value_stats: {'mean': 142.0380096435547, 'stddev': 117.50126647949219, 'total_count': 1234, 'null_count': 745}
        corrupted:
            is_optional: False
            representation_type: Boolean
        corrupted_right:
            is_optional: False
            representation_type: Boolean
        destination:
            is_optional: False
            representation_type: String
            value_range: {'ListOfValues': [None, '', 'VILA', 'ES SUR', 'ESICL', 'EBAL>EDGA', 'IT-SEP', 'PLATF ROMA', 'ITL-BREG']}
        dimension_to_bow:
            is_optional: False
            representation_type: UInt16

     ...
         sog:
         is_optional: False
         representation_type: Float32
         value_stats: {'mean': 2.0780696868896484, 'stddev': 4.677201271057129, 'total_count': 1979, 'null_count': 0}
     version:
         is_optional: False
         representation_type: Int64
         value_range: {'MinMax': {'min': 3, 'max': 3, 'allow_missing': True}}
         value_stats: {'mean': 3.0, 'stddev': 0.0, 'total_count': 1979, 'null_count': 0}


     First 10 and last 10 rows:
     shape: (10, 32)
     ┌───────────┬─────────────────────┬──────────┬───────────┬──────┬───┬────────────┬────────────────────┬──────────────────┬───────────────────────┬─────────┐
     │ mmsi      ┆ reception_date      ┆ lon      ┆ lat       ┆ rot  ┆ … ┆ eta        ┆ message_type_right ┆ satellite_static ┆ reception_date_static ┆ version │
     │ ---       ┆ ---                 ┆ ---      ┆ ---       ┆ ---  ┆   ┆ ---        ┆ ---                ┆ ---              ┆ ---                   ┆ ---     │
     │ i32       ┆ datetime[ms]        ┆ f64      ┆ f64       ┆ f32  ┆   ┆ i64        ┆ i64                ┆ str              ┆ datetime[ms]          ┆ i64     │
     ╞═══════════╪═════════════════════╪══════════╪═══════════╪══════╪═══╪════════════╪════════════════════╪══════════════════╪═══════════════════════╪═════════╡
     │ 345080000 ┆ 2020-11-18 33:00:18 ┆ 0.783398 ┆ 40.483513 ┆ 0.0  ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 334015340 ┆ 2020-11-18 33:00:33 ┆ 0.435345 ┆ 40.414097 ┆ null ┆ … ┆ 1735889800 ┆ 5                  ┆ SAT-AA_037       ┆ 2020-11-18 33:07:35   ┆ 3       │
     │ 334088470 ┆ 2020-11-18 33:00:37 ┆ 0.403745 ┆ 40.358495 ┆ null ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 334098970 ┆ 2020-11-18 33:00:39 ┆ 0.88999  ┆ 40.389833 ┆ null ┆ … ┆ 1783310700 ┆ 5                  ┆ SAT-AA_038       ┆ 2020-11-18 33:04:13   ┆ 3       │
     │ 333019738 ┆ 2020-11-18 33:01:18 ┆ 0.80045  ┆ 40.819483 ┆ null ┆ … ┆ null       ┆ 34                 ┆ SAT-AA_038       ┆ 2020-11-18 33:33:51   ┆ 3       │
     │ 353003075 ┆ 2020-11-18 33:01:38 ┆ 0.550948 ┆ 40.571973 ┆ 0.0  ┆ … ┆ null       ┆ 5                  ┆ SAT-AA_037       ┆ 2020-11-18 33:01:13   ┆ 3       │
     │ 345080000 ┆ 2020-11-18 33:01:37 ┆ 0.759477 ┆ 40.481487 ┆ 0.0  ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 334015340 ┆ 2020-11-18 33:01:33 ┆ 0.435338 ┆ 40.414093 ┆ null ┆ … ┆ 1735889800 ┆ 5                  ┆ SAT-AA_037       ┆ 2020-11-18 33:07:35   ┆ 3       │
     │ 334088470 ┆ 2020-11-18 33:01:37 ┆ 0.403743 ┆ 40.358513 ┆ null ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 334098970 ┆ 2020-11-18 33:03:18 ┆ 0.890313 ┆ 40.370833 ┆ null ┆ … ┆ 1783310700 ┆ 5                  ┆ SAT-AA_038       ┆ 2020-11-18 33:04:13   ┆ 3       │
     └───────────┴─────────────────────┴──────────┴───────────┴──────┴───┴────────────┴────────────────────┴──────────────────┴───────────────────────┴─────────┘
     shape: (10, 32)
     ┌───────────┬─────────────────────┬──────────┬───────────┬──────┬───┬────────────┬────────────────────┬──────────────────┬───────────────────────┬─────────┐
     │ mmsi      ┆ reception_date      ┆ lon      ┆ lat       ┆ rot  ┆ … ┆ eta        ┆ message_type_right ┆ satellite_static ┆ reception_date_static ┆ version │
     │ ---       ┆ ---                 ┆ ---      ┆ ---       ┆ ---  ┆   ┆ ---        ┆ ---                ┆ ---              ┆ ---                   ┆ ---     │
     │ i32       ┆ datetime[ms]        ┆ f64      ┆ f64       ┆ f32  ┆   ┆ i64        ┆ i64                ┆ str              ┆ datetime[ms]          ┆ i64     │
     ╞═══════════╪═════════════════════╪══════════╪═══════════╪══════╪═══╪════════════╪════════════════════╪══════════════════╪═══════════════════════╪═════════╡
     │ 335990004 ┆ 2020-11-19 01:59:00 ┆ 0.849883 ┆ 40.937813 ┆ null ┆ … ┆ null       ┆ 34                 ┆ SAT-AA_038       ┆ 2020-11-19 01:31:43   ┆ 3       │
     │ 334015340 ┆ 2020-11-19 03:00:13 ┆ 0.435335 ┆ 40.414083 ┆ null ┆ … ┆ 1735889800 ┆ 5                  ┆ SAT-AA_037       ┆ 2020-11-19 01:19:34   ┆ 3       │
     │ 334088470 ┆ 2020-11-19 03:00:19 ┆ 0.40377  ┆ 40.358493 ┆ null ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 333049539 ┆ 2020-11-19 03:00:31 ┆ 0.80088  ┆ 40.819835 ┆ null ┆ … ┆ null       ┆ 34                 ┆ SAT-AA_038       ┆ 2020-11-19 01:03:01   ┆ 3       │
     │ 334018830 ┆ 2020-11-19 03:00:35 ┆ 0.895348 ┆ 40.897835 ┆ null ┆ … ┆ 1735889800 ┆ 5                  ┆ SAT-AA_038       ┆ 2020-11-19 01:07:38   ┆ 3       │
     │ 333058871 ┆ 2020-11-19 03:00:31 ┆ 0.800105 ┆ 40.819735 ┆ null ┆ … ┆ null       ┆ 34                 ┆ SAT-AA_037       ┆ 2020-11-19 00:59:00   ┆ 3       │
     │ 334098970 ┆ 2020-11-19 03:00:37 ┆ 0.891373 ┆ 40.403033 ┆ null ┆ … ┆ 1783310700 ┆ 5                  ┆ SAT-AA_038       ┆ 2020-11-19 01:04:13   ┆ 3       │
     │ 345080000 ┆ 2020-11-19 03:00:38 ┆ 0.373085 ┆ 40.10578  ┆ 0.0  ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 333041379 ┆ 2020-11-19 03:00:48 ┆ 0.801778 ┆ 40.818448 ┆ null ┆ … ┆ null       ┆ null               ┆ null             ┆ null                  ┆ 3       │
     │ 333048134 ┆ 2020-11-19 03:00:54 ┆ 0.803097 ┆ 40.830813 ┆ null ┆ … ┆ null       ┆ 34                 ┆ SAT-AA_037       ┆ 2020-11-19 01:08:30   ┆ 3       │
     └───────────┴─────────────────────┴──────────┴───────────┴──────┴───┴────────────┴────────────────────┴──────────────────┴───────────────────────┴─────────┘
.. highlight:: none


Examples
^^^^^^^^^

Individual columns can be filtered using a python expression that is compliant with the backend (here: polars) being used.

For instance to extract:

- the time-series for a particular id (mmsi):
.. highlight:: python

::

   damast inspect -f 1.zip --filter 'mmsi == 335990004'

.. highlight:: none

- all data in a time interval:

.. highlight:: python
   
::

    damast inspect -f 1.zip --filter 'reception_date >= dt.datetime.fromisoformat("2020-11-19 00:00:00")' --filter 'reception_date <= dt.datetime.fromisoformat("2020-11-20 00:00:00")'

.. highlight:: none


Convert
--------

.. literalinclude:: ./examples/damast-convert-help.txt
  :language: none

.. highlight:: python

::

    damast convert --help
    usage: damast convert [-h] [-w WORKDIR] [-v] [--loglevel LOGLEVEL] [--logfile LOGFILE] -f FILES [FILES ...] [-m METADATA_INPUT] [-o OUTPUT_FILE] [--output-dir OUTPUT_DIR]
                          [--output-type OUTPUT_TYPE] [--validation-mode {ignore,readonly,update_data,update_metadata}]
                          {} ...

    damast convert - data conversion subcommand called

    positional arguments:
      {}                    sub-command help

    options:
      -h, --help            show this help message and exit
      -w WORKDIR, --workdir WORKDIR
      -v, --verbose
      --loglevel LOGLEVEL   Set loglevel to display
      --logfile LOGFILE     Set file for saving log (default prints to terminal)
      -f FILES [FILES ...], --files FILES [FILES ...]
                            Files or patterns of the (annotated) data file that should be converted
      -m METADATA_INPUT, --metadata-input METADATA_INPUT
                            The metadata input file
      -o OUTPUT_FILE, --output-file OUTPUT_FILE
                            The output file either: .parquet, .hdf5
      --output-dir OUTPUT_DIR
                            The output directory
      --output-type OUTPUT_TYPE
                            The output file type: .parquet (default) or .hdf5
      --validation-mode {ignore,readonly,update_data,update_metadata}
                            Define the validation mode

.. highlight:: none

Examples
^^^^^^^^^

- convert one or more files to parquet (N:N)

.. highlight:: python

::

    damast convert -f 1.zip --output-dir export --output-type .parquet

.. highlight:: none


- convert one or more files to a single parquet file (N:1)

.. highlight:: python

::

    damast convert -f 1.zip --output-file data-1.parquet --output-type .parquet

.. highlight:: none



Annotate
--------

.. literalinclude:: ./examples/damast-annotate-help.txt
  :language: none

Examples
^^^^^^^^

- set the unit for two columns, here *lat* and *lon* to *deg*, and creating a new file in the subfolder *export*

.. highlight:: python

::

    damast annotate -f input.parquet --set-unit lon:deg lat:deg --output-dir export

.. highlight:: none

- set the unit for two columns, here *lat* and *lon* to *deg*, inplace, i.e., change the existing file

.. highlight:: python

::

    damast annotate -f input.parquet --set-unit lon:deg lat:deg --inplace

.. highlight:: none



Process
---------

Once a DataProcessPipeline has been exported and saved, e.g., in the following example as *my-pipeline.damast.ppl*, it can be reapplied to an existing data set.
The dataset needs to comply with the required input columns and metadata requirements, such as units, so that the pipeline can successfully run.
Damast will check these requirements and raise an exception if these requirements are not satisfied.

.. literalinclude:: ./examples/damast-process-pipeline.py
   :language: Python

.. literalinclude:: ./examples/damast-process-help.txt
  :language: none

Examples
^^^^^^^^

.. highlight:: python

::

    damast process --input-data input.parquet --pipeline pipelines/my-pipeline.damast.ppl

.. highlight:: none

