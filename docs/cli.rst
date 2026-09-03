Command Line Interface
========================

The command line interface offers a number of workflow simplifications that are encapsulated
in sub-commands:

**inspect**
    check metadadata and dataset properties

**convert**
    convert from (zipped) csv, netcdf to parquet (default) or hdf5 (deprecated)

**annotate**
    create metadata file and update dataframe with metadata

**process**
    apply a data pipeline to a dataset

**plugins**
    list transformer plugins registered by installed packages or via ``DAMAST_PLUGIN_PATH``

**watch**
    watch directories for completed files and run a configured command on each


Inspect
--------

.. literalinclude:: ./examples/damast-inspect-help.txt
  :language: none


Inspect allows to identify columns and properties of columns in a given dataset.
The dataset can consist of one or more (zipped) files, either given as list of filenames or using file pattern.

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

By default, ``inspect`` runs in ``READONLY`` validation mode: it only checks metadata against the
data, without computing or changing anything.
If a file's stored metadata is missing ``value_range``/``value_stats``, then pass ``--validation-mode update_metadata`` to have them computed from the actual
data instead:

.. highlight:: python
::

    $ damast inspect -f data.parquet --validation-mode update_metadata


.. highlight:: none
To make the information persistent in the parquet file:

.. highlight:: python
::

    $ damast annotate -f data.parquet --apply --inplace

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

.. literalinclude:: ./examples/damast-process-help.txt
  :language: none


Once a DataProcessPipeline has been exported and saved, e.g., in the following example as *my-pipeline.damast.ppl*, it can be reapplied to an existing data set.
The dataset needs to comply with the required input columns and metadata requirements, such as units, so that the pipeline can successfully run.
Damast will check these requirements and raise an exception if these requirements are not satisfied.

.. literalinclude:: ./examples/damast-process-pipeline.py
   :language: Python

Examples
^^^^^^^^

.. highlight:: python

::

    damast process --input-data input.parquet --pipeline pipelines/my-pipeline.damast.ppl

.. highlight:: none

Multiple input datasources
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A pipeline is not limited to a single input - :func:`damast.core.dataprocessing.DataProcessingPipeline.join`
lets a pipeline declare additional, named datasources, e.g., here to fuse AIS ship-position pings
with OSINT (open-source intelligence) event reports by timestamp.
Using damast process with ``--input-data`` allows to specify a datasource with a prefix, e.g.,
``--input-data osint_events=1.parquet 2.parquet``.
A pipeline with only the default datasource (the common case) can omit the prefix ``--input-data 1.parquet``
name needed.

The following trims down the pattern used in
`damast-examples/hozint-ais/ais_osint_fusion.py <https://github.com/simula/damast-examples>`_ to
a minimal, self-contained example - a ``JoinByTimestamp`` plugin transformer (see `Plugins`_
below) that joins AIS pings (the default ``df`` datasource) with OSINT events (a second
datasource, ``osint_events``) wherever their timestamps match:

.. literalinclude:: ./examples/plugins/osint_ais_transformers.py
   :language: Python

.. literalinclude:: ./examples/damast-osint-ais-pipeline.py
   :language: Python

Generating small synthetic datasets to try it on:

.. literalinclude:: ./examples/damast-osint-ais-generate-data.py
   :language: Python

.. highlight:: python

::

    python docs/examples/damast-osint-ais-generate-data.py
    python docs/examples/damast-osint-ais-pipeline.py

    # JoinByTimestamp is a local plugin transformer (see Plugins below) - needed both to
    # build the pipeline above and to load it back for damast process
    export DAMAST_PLUGIN_PATH=docs/examples/plugins
    damast process --pipeline pipelines/osint_ais_preparation.damast.ppl \
        --input-data df=docs/examples/data/ais.parquet \
        --input-data osint_events=docs/examples/data/osint.parquet \
        --output-file output.parquet

.. highlight:: none

Describe a pipeline
^^^^^^^^^^^^^^^^^^^^

``--describe`` prints a saved pipeline's interface: every datasource it requires and the
columns each one must provide, followed by every processing step.
Noe input data or running the pipeline is needed. Loading the pipeline still needs its transformers to be
resolvable, i.e., ``DAMAST_PLUGIN_PATH`` must be set for a local plugin, while installed plugins will automatically be discovered:

.. highlight:: python

::

    export DAMAST_PLUGIN_PATH=docs/examples/plugins
    damast process --pipeline pipelines/osint_ais_preparation.damast.ppl --describe

.. highlight:: none

.. literalinclude:: ./examples/damast-process-describe.txt
  :language: none

The **Interface** section is specifically the requirement for each datasource's *first*
consuming step - once inside the pipeline, later steps consume columns the pipeline itself has
already produced, not the raw datasource, so this is the actual external contract to satisfy
when supplying ``--input-data``.

Experiment tracking
^^^^^^^^^^^^^^^^^^^^

``damast.integrations.mlflow_tracker.track_pipeline`` permits a pipeline to use [mlflow](https://mlflow.org) for reporting.
Part of the reporting is the output ``AnnotatedDataFrame``'s metadata contract, i.e., units, and representation_type as well as runtime stats for per-step timing and row counts.

The feature requires mlflow, which will be installed with the extra ``ml``:
(``uv pip install damast[ml]``).

.. literalinclude:: ./examples/damast-mlflow-tracking.py
   :language: Python


A ``damast.ml.experiments.Experiment`` (see the *Experiments* notebook) combines such a pipeline
with model training and evaluation. ``Experiment.run()`` performs all three internally, so the pipeline-level
contract and the per-epoch training metrics need two separate hooks in the same run: wrap the
call in ``track_pipeline`` for the pipeline (as above), and additionally call
``mlflow.keras.autolog()`` beforehand.

The experiment report that ``Experiment.run()`` writes (training parameters, per-model evaluation
results) isn't part of any ``AnnotatedDataFrame``'s metadata, so it is logged separately, straight
onto the tracker:

.. literalinclude:: ./examples/damast-mlflow-ml-experiment.py
   :language: Python

If other trackers, e.g. W&B, shall be used, an integration can be based on
``damast.integrations.tracking.flatten_metadata``/``flatten_step_stats`` and the
``damast.integrations.tracking.ExperimentTracker`` interface, both are backend-agnostic,



Watch
------

.. literalinclude:: ./examples/damast-watch-help.txt
  :language: none

Some data sources are collected incrementally, e.g. one file per day, appended to for hours
before it is complete. ``watch`` scans one or more directories for such files and, once a file
has not been modified for a configurable *quiet period*, runs a configured command on it -
typically ``damast convert`` with a metadata spec, or ``damast process`` with a pipeline.

``watch`` performs a single scan-and-exit; it is not a daemon. Schedule it periodically with
cron or a systemd timer (see `Deployment`_ below). On success a source file is moved into the
job's ``processed_dir``, on failure into ``failed_dir`` alongside a ``<file>.error.log`` - this
also means a file is only ever handled once, even across repeated invocations.

Configuration
^^^^^^^^^^^^^^

A watch config is a YAML file listing one or more jobs:

.. highlight:: yaml

::

    jobs:
      - name: ais-daily                 # optional; default: source_dir's basename
        source_dir: /data/incoming/ais
        target_dir: /data/processed/ais # optional, default: source_dir; exposed to command as {output_dir}
        pattern: "*.csv"                # optional, default "*.csv"
        quiet_period: 1800              # optional, seconds, default 1800
        processed_dir: /data/incoming/ais/processed   # optional, default {source_dir}/processed
        failed_dir: /data/incoming/ais/failed          # optional, default {source_dir}/failed
        command:                        # required, argv list - no shell
          - damast
          - process
          - --pipeline
          - /pipelines/ais.damast.ppl
          - --input-data
          - "{input}"
          - --output-file
          - "{output_dir}/{stem}.parquet"

      - name: osint-daily
        source_dir: /data/incoming/osint
        command:
          - damast
          - convert
          - -f
          - "{input}"
          - -m
          - /specs/osint.spec.yaml
          - -o
          - "{output_dir}/{stem}.parquet"

.. highlight:: none

Each token of ``command`` is substituted with ``str.format``, so a job's command can use these
placeholders:

============  ==================================================
placeholder   value
============  ==================================================
``{input}``   absolute path of the ready file
``{output_dir}``  the job's ``target_dir``
``{stem}``    the ready file's name without its suffix
``{name}``    the ready file's name with its suffix
============  ==================================================

``--create-config`` builds a config interactively instead of hand-writing the YAML, prompting
for each job's fields (blank answers accept the default shown in brackets) and writing the
result to the file given via ``--config``:

.. highlight:: none

::

    $ damast watch --config watch.yaml --create-config
    --- job 1 ---
    Source directory to watch: /data/incoming/ais
    Job name [ais]:
    Target directory (available to the command as {output_dir}) [/data/incoming/ais]: /data/processed/ais
    File pattern [*.csv]:
    Quiet period in seconds before a file is considered complete [1800]:
    Command to run on each ready file (use {input}/{output_dir}/{stem}/{name}): damast process --pipeline /pipelines/ais.damast.ppl --input-data {input} --output-file {output_dir}/{stem}.parquet
    Add another job? [y/N]: n
    Wrote 1 job(s) to 'watch.yaml'

The command line is split the same way a shell would (via ``shlex.split``) into the argv list
the config stores - no shell is ever invoked. Running ``--create-config`` against a file that
already exists asks whether to append the new job(s) to it or overwrite it.

Environment variables and home directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``source_dir``, ``target_dir``, ``processed_dir``, ``failed_dir`` and every token of
``command`` accept ``${VARNAME}`` and ``~``/``{home}``, expanded to the named environment
variable and the current user's home directory respectively:

.. highlight:: yaml

::

    jobs:
      - name: ais-daily
        source_dir: ${DATA_ROOT}/incoming/ais
        target_dir: ~/damast-watch/ais
        command:
          - damast
          - convert
          - -f
          - "{input}"
          - -m
          - "{home}/specs/ais.spec.yaml"
          - -o
          - "{output_dir}/{stem}.parquet"

.. highlight:: none

Expansions run at different times, which matters for how a missing
variable shows up:

- ``source_dir``/``target_dir``/``processed_dir``/``failed_dir`` are expanded once, when the
  job is loaded - raising an exception when encountering an unset variable
- ``command`` tokens are expanded for every ready file.

Progress
^^^^^^^^

While a job runs, ``watch`` shows a live ``tqdm`` progress bar naming the job, the file
currently being processed, and the log to check on it:

.. highlight:: none

::

    [ais-daily] 2026-08-30.csv:  45%|####5     | 5/11 [00:03<00:04,  1.32file/s, log: /data/processed/ais/2026-08-30.log]

The command's own stdout/stderr is streamed live into that ``<target_dir>/<stem>.log`` file -
``tail -f`` it to follow a long-running command - and mirrored to the ``damast.core.watch``
logger at ``DEBUG`` level. Neither is printed to the console at the default ``INFO`` level, so
the progress bar stays a single, clean line; pass ``--log-level DEBUG`` to see that output
inline instead. On failure, the same log path is named in the raised error and, for the moved
file, duplicated into ``<failed_dir>/<file>.error.log`` alongside the traceback.

Examples
^^^^^^^^

.. highlight:: python

::

    damast watch --config watch.yaml

    # restrict to one job, e.g. for a tighter cron schedule
    damast watch --config watch.yaml --job ais-daily

    # list which files are ready without running anything
    damast watch --config watch.yaml --dry-run

.. highlight:: none

A non-zero exit code means at least one job or file failed - useful for cron/systemd failure
alerting. ``watch`` does not interpret the configured command's output; check a job's
``failed_dir`` for the moved input and its ``.error.log`` (captured stdout/stderr and exit code).

Deployment
^^^^^^^^^^^

Since ``damast watch`` performs a single scan, recurring execution is the deployment's
responsibility. A crontab entry, scanning every 15 minutes:

.. highlight:: none

::

    */15 * * * * damast watch --config /etc/damast/watch.yaml

Or a systemd service/timer pair::

    # /etc/systemd/system/damast-watch.service
    [Unit]
    Description=damast watch

    [Service]
    Type=oneshot
    ExecStart=/usr/local/bin/damast watch --config /etc/damast/watch.yaml

    # /etc/systemd/system/damast-watch.timer
    [Unit]
    Description=Run damast-watch periodically

    [Timer]
    OnCalendar=*:0/15
    Persistent=true

    [Install]
    WantedBy=timers.target

Enable with ``systemctl enable --now damast-watch.timer``; ``OnFailure=`` on the service can
point at an alerting unit, since the non-zero exit code on any job/file failure is preserved.


Plugins
---------

.. literalinclude:: ./examples/damast-plugins-help.txt
  :language: none

A pipeline can use transformers that are not part of the ``damast`` package itself. Two ways of
registering such plugin transformers are supported - see :class:`damast.core.transformations.PluginManager`
for the full API:

- installable packages that declare their :class:`damast.core.transformations.PipelineElement` subclasses via the
  ``damast.transformers`` entry-point group in their own ``pyproject.toml``::

      [project.entry-points."damast.transformers"]
      MyTransformer = "acme_pkg.transformers:MyTransformer"

- local, ad-hoc ``*.py`` files that are not part of any installed package, made discoverable by
  pointing the ``DAMAST_PLUGIN_PATH`` environment variable at the directory (or directories,
  separated with ``os.pathsep``) that contains them

Regardless of which of the two a transformer comes from, it is resolvable in code the same way,
via the ``damast.plugins`` namespace::

    from damast.plugins import MyTransformer

``damast.plugins`` resolves names lazily on first access, so nothing beyond the requested class
is ever imported - see :mod:`damast.plugins` for details.

Example: a local plugin transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A local, ad-hoc transformer is just a :class:`damast.core.transformations.PipelineElement`
subclass in a loose ``*.py`` file, written like any other transformer:

.. literalinclude:: ./examples/plugins/my_transformers.py
   :language: Python

With ``DAMAST_PLUGIN_PATH`` pointing at the directory containing that file, ``damast plugins``
lists it without requiring any further Python code:

.. highlight:: none

::

    $ export DAMAST_PLUGIN_PATH=./examples/plugins
    $ damast plugins

    MyTripler: my_transformers:MyTripler

``MyTripler`` is now resolvable via ``damast.plugins`` and can be used in a pipeline like any
other transformer:

.. literalinclude:: ./examples/damast-plugin-pipeline.py
   :language: Python

The resulting pipeline can be applied like any other, e.g. via ``damast process`` (see `Process`_
above), as long as ``DAMAST_PLUGIN_PATH`` is still set to a directory containing
``my_transformers.py``:

.. highlight:: none

::

    damast process --input-data data.parquet --pipeline pipelines/my-plugin-pipeline.damast.ppl

Pipelines saved with a plugin transformer record where it came from under ``requires`` (the
installed distribution and version, or the original local file path), so that loading the pipeline
elsewhere fails with an actionable message - naming the missing package to ``pip install``, or the
``DAMAST_PLUGIN_PATH`` directory to add - instead of a bare import error.

