"""
Module containing decorators and classes to model data processing pipelines
"""
from __future__ import annotations

import copy
import importlib
import re
import tempfile
from datetime import datetime, timezone
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

import damast.version
from damast.core.transformations import PipelineElement
from damast.core.processing_graph import Node, ProcessingGraph
import traceback as tc

from .transformations import PipelineElement
from .dataframe import AnnotatedDataFrame
from .formatting import DEFAULT_INDENT
from .metadata import MetaData, DataSpecification

__all__ = [
    "artifacts",
    "input",
    "output",
    "describe",
    "DataProcessingPipeline",
    "PipelineElement",
]

_log: Logger = getLogger(__name__)

DAMAST_PIPELINE_SUFFIX: str = ".damast.ppl"
"""Suffix of :class:`DataProcessingPipeline` files created in :func:`DataProcessingPipeline.save`
and used by :func:`DataProcessingPipeline.load`"""

VAEX_STATE_SUFFIX: str = ".vaex-state.json"
"""Suffix of :class:`DataProcessingPipeline` files created in :func:`DataProcessingPipeline.save_state`
and used by :func:`DataProcessingPipeline.load_state`"""


class DataProcessingPipeline(PipelineElement):
    """
    A data-processing pipeline for a sequence of transformers

    :param name: Name of the pipeline
    :param base_dir: Base directory towards which transformer output which be relative
    :param steps: Steps that form the basis for this pipeline
    :param inplace_transformation: If true, the input :class:`damast.core.dataframe.AnnotatedDataFrame` is not
        copied when calling :func:`transform`. Else input data-frame is untouched
    :param name_mappings: Name mappings that should be applied to individual transformations.

    :raises ValueError: If any of the transformer names are `None`
    :raises AttributeError: If the transformer is missing the :func:`transform` function
    :raises AttributeError: If transformer is missing input or output decoration
    :raises RuntimeError: If the sequence of transformers does not satisfy the sequential requirements
    """

    #: Name of the processing pipeline
    name: str
    description: str

    #: Base path (which is forwarded to transformers, when calling
    #: transform)
    base_dir: Path

    #: The output specs - as specified by decorators
    _output_specs: List[DataSpecification]

    #: Check if the pipeline is ready to be run
    is_ready: bool
    # The processing graph that define this pipeline
    processing_graph: ProcessingGraph 

    _inplace_transformation: bool
    _name_mappings: Dict[str, str]
    _processing_stats: Dict[str, Dict[str, Any]]

    _meta: Dict[str, str]

    def __init__(self, *,
                 name: str,
                 description: str = "",
                 base_dir: Union[str, Path] = tempfile.gettempdir(),
                 processing_graph: List[Tuple[str, Union[Dict[str, Any], PipelineElement]]] | ProcessingGraph = None,
                 inplace_transformation: bool = False,
                 name_mappings: Dict[str, str] = {},
                 meta: Dict[str, str] | None = None,
                 ):
        super().__init__()

        self.name = name
        self.description = description
        self.base_dir = Path(base_dir)

        self._output_specs = []
        self._inplace_transformation = inplace_transformation

        self._name_mappings = name_mappings
        self._processing_stats = {}

        self.processing_graph = ProcessingGraph()
        if processing_graph:
            if isinstance(processing_graph, ProcessingGraph):
                self.processing_graph = processing_graph
            elif isinstance(processing_graph, dict):
                self.processing_graph = ProcessingGraph.from_dict(processing_graph)
            elif isinstance(processing_graph, list):
                for step in processing_graph:
                    name, instance = step
                    if isinstance(instance, PipelineElement):
                        self.processing_graph.add(Node(
                                name=name,
                                transformer=instance
                            )
                        )
                    elif isinstance(instance, dict):
                        self.processing_graph.add(Node(
                                name=name,
                                transformer=PipelineElement.create_new(**instance)
                            )
                        )
                    else:
                        raise ValueError(f"{self.__class__.__name__}.__init__: could not instantiate PipelineElement"
                                     f" from {type(step)}")
        self.is_ready = False

        if meta is not None:
            self._meta = meta
        else:
            self._meta = {
                'damast_version': damast.version.__version__
            }

    @property
    def output_specs(self):
        if not self.is_ready:
            raise RuntimeError(
                f"{self.__class__.__name__}.output_specs: pipeline is not yet ready to run. "
                f"Please call 'prepare()' to set the correct output specs"
            )

        return self._output_specs

    def add(
            self,
            name: str,
            transformer: PipelineElement,
            name_mappings: Optional[Dict[str, str]] = None,
    ) -> DataProcessingPipeline:
        """
        Add a pipeline step

        :param name: Name of the step
        :param transformer: The transformer that shall be executed
        :param name_mappings: Allow to define a name mapping for this pipeline element instance
        :return:
        """
        transformer.set_parent(pipeline=self)
        if name_mappings is not None:
            transformer._name_mappings = name_mappings

        self.processing_graph.add(
                Node(name=name,
                     transformer=transformer
                     )
        )
        self.is_ready = False
        return self

    def join(
            self,
            name: str,
            operator: PipelineElement, 
            data_source: DataProcessingPipeline | None = None,
            name_mappings: Optional[Dict[str, str]] = None,
    ) -> DataProcessingPipeline:
        """
        Add a pipeline step

        :param name: Name of the step
        :param transformer: The transformer that shall be executed
        :param name_mappings: Allow to define a name mapping for this pipeline element instance
        :return:
        """
        operator.set_parent(pipeline=self)
        if name_mappings is not None:
            operator._name_mappings = name_mappings

        self.processing_graph.join(
                name=name,
                processing_graph=data_source.processing_graph if data_source else None,
                operator=operator
        )
        self.is_ready = False
        return self

    @classmethod
    def validate(
            cls, processing_graph: ProcessingGraph, metadata: MetaData
    ) -> Dict[str, Any]:
        """
        Validate the existing pipeline and collect the minimal input and output data specification.

        :param steps: processing steps
        :param metadata: the input metadata for this pipeline
        :return: The minimal output specification for this pipeline
        """
        # Keep track of the expected (minimal) specs at each step in the pipeline
        current_specs: Optional[List[DataSpecification]] = copy.deepcopy(
            metadata.columns
        )
        for node in processing_graph.nodes():
            if node.name is None:
                raise ValueError(
                    f"{cls.__name__}.validate: missing name processing step"
                )

            if not hasattr(node.transformer, "transform"):
                raise AttributeError(
                    f"{cls.__name__}.validate: processing step '{node.name}' does not fulfill the"
                    f" TransformerMixin requirements - no method 'fit_transform' found"
                )

            input_specs = node.transformer.input_specs
            output_specs = node.transformer.output_specs

            md = MetaData(columns=current_specs, annotations=[])
            fulfillment = md.get_fulfillment(expected_specs=input_specs)
            if not fulfillment.is_met():
                raise RuntimeError(
                    f"{cls.__name__}.validate: Input requirements are not fulfilled. "
                    f"Current input available (at step '{node.name}'): {fulfillment}"
                )

            current_specs = DataSpecification.merge_lists(current_specs, input_specs)
            current_specs = DataSpecification.merge_lists(current_specs, output_specs)

        return {"processing_graph": processing_graph, "output_spec": current_specs}

    def save(self, dir: Union[str, Path]) -> Path:
        """
        Save the processing pipeline

        :param dir: directory where to save this pipeline
        """
        base_dir = Path(dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = base_dir / f"{self.name}{DAMAST_PIPELINE_SUFFIX}"

        with open(filename, "w") as f:
            yaml.dump(dict(self), f)
        return filename

    def __iter__(self):
        yield "name", self.name
        yield "description", self.description
        yield "meta", self._meta
        yield "base_dir", str(self.base_dir)
        yield "processing_graph", dict(self.processing_graph)

    def save_state(self,
                   df: AnnotatedDataFrame,
                   dir: Union[str, Path]) -> Path:
        """
        Save the processing pipeline

        :param dir: directory where to save this pipeline
        """
        filename = Path(dir) / f"{self.name}{VAEX_STATE_SUFFIX}"
        #df._dataframe.state_write(file=filename)
        df._dataframe.serialize(filename)
        return filename

    @classmethod
    def load(cls, path: Union[str, Path], name: str = "*") -> DataProcessingPipeline:
        """
        Load a :class:`DataProcessingPipeline` from file (without suffix :attr:`DAMAST_PIPELINE_SUFFIX`)

        :param path: Directory containing pipeline(s) (with fiven suffix) or pipeline filename
        :name: Name of file(s) without suffix. Can be a Regex expression
        """
        basename = f"{name}{DAMAST_PIPELINE_SUFFIX}"
        path = Path(path)
        if path.is_dir():
            files = list(path.glob(basename))
        elif path.is_file():
            files = [path]
        else:
            raise RuntimeError("{self.__class__.__name__}.load: {path}"
                               " is neither directory nor file")

        if len(files) == 1:
            filename = files[0]
        elif len(files) == 0:
            raise FileNotFoundError(
                f"{cls.__name__}.load: could not find '{basename}' in {dir}"
            )
        else:
            raise RuntimeError(
                f"{cls.__name__}.load: multiple pipeline available, pls be more specific:"
                f" {','.join([x.name for x in files])}"
            )

        with open(filename, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        return cls(**data)

    @classmethod
    def load_state(
            cls, df: AnnotatedDataFrame, dir: Union[str, Path], name: str = "*"
    ) -> AnnotatedDataFrame:
        """
        Load a ``vaex`` state (from file) to a :class:`damast.core.AnnotatedDataFrame`.

        :param df: The data-frame
        :param dir: Directory of vaex state file
        :param name: Regex describing vaex state file (without :attr:`VAEX_STATE_SUFFIX`)
        :raises FileNotFoundError: If no files found
        :raises RuntimeError: If multiple vaex-states satisfies the ``name`` requirement.
        :return: The data-frame with applied state
        """
        basename = f"{name}{VAEX_STATE_SUFFIX}"
        dir_path = Path(dir)

        files = list(dir_path.glob(basename))
        if len(files) == 1:
            filename = files[0]
        elif len(files) == 0:
            raise FileNotFoundError(
                f"{cls.__name__}.load: could not find '{basename}' in {dir_path}"
            )
        else:
            raise RuntimeError(
                f"{cls.__name__}.load: multiple pipeline available, pls be more specific:"
                f" {','.join([x.name for x in files])}"
            )

        #df.dataframe.state_load(file=filename)
        df._dataframe = df.dataframe.deserialize(filename)
        return df

    def prepare(self, **dataframes) -> DataProcessingPipeline:
        """
        Prepare the pipeline by applying necessary name mapping and validating all steps that define the pipeline.

        A :class:`DataProcessingPipeline` must be prepared before execution.

        :returns: This instance of data processing pipeline
        """
        if not dataframes:
            raise RuntimeError("DataProcessingPipeline.prepare: missing dataframes")

        # At this stage, ensure that the dataframes conforms to their metadata
        for name, df in dataframes.items():
            try:
                df.validate_metadata()
            except ValueError as e:
                raise RuntimeError(
                    f"{self.__class__.__name__}.prepare: specification of the provided AnnotatedDataFrame '{name}'"
                    f" does not match its data. If you modified the frame after construction ensure"
                    f" consistency between 'data' and 'metadata' by using "
                    f" AnnotatedDataFrame.validate_metadata()"
                    f" -- {e}"
                ) from e

        # The pipeline will define a name mapping (only for items in its interface)
        for k, v in self._name_mappings.items():
            for node in self.processing_graph:
                pipeline_element = node.transformer

                input_columns = [x.name for x in pipeline_element.input_specs]
                output_columns = [x.name for x in pipeline_element.output_specs]

                columns = input_columns + output_columns
                for x in columns:
                    if x == k:
                        pipeline_element.name_mappings[x] = v

        validation_result = self.validate(processing_graph=self.processing_graph,
                                          metadata=df.metadata)
        self.is_ready = True

        self.processing_graph = validation_result["processing_graph"]
        self._output_specs: List[DataSpecification] = validation_result["output_spec"]

        return self

    def transform(self, df: AnnotatedDataFrame, **kwargs) -> AnnotatedDataFrame:
        """
        Apply pipeline on given annotated dataframe

        .. note::
            If any filters are applied by the pipeline, these are executed at the end of the
            transformation.

        :param df: The input dataframe
        :returns: The transformed dataframe
        """
        dataframes = { '__default__': df }
        for x in self.processing_graph.get_joins():
            if x.name not in kwargs:
                raise RuntimeError("DataProcessingPipeline.transform: "
                                   "missing data source argument"
                                   f" for join '{x}' - found {kwargs.keys()}")
            dataframes[x.name] = kwargs[x.name]

        for label, dataframe in dataframes.items():
            if not isinstance(df, AnnotatedDataFrame):
                raise TypeError(
                    f"{self.__class__.__name__}.transform"
                    f" expected an annotated dataframe for processing ({label=}), "
                    f"got {type(df)}"
                )

        if not self.is_ready:
            self.prepare(**dataframes)

        for name, df in dataframes.items():
            if df.is_empty():
                raise RuntimeError(
                    f"{self.__class__.__name__}.transform: there is no data available to transform using datasource '{name}'"
            )

        if self._inplace_transformation:
            in_dataframes = dataframes
        else:
            in_dataframes = {x: copy.deepcopy(y) for x,y in dataframes.items()}

        adf = self._run(in_dataframes)
        assert isinstance(adf, AnnotatedDataFrame)

        adf.validate_metadata(validation_mode=damast.core.ValidationMode.UPDATE_METADATA)
        return adf

    def _run(self, dataframes: dict[str, AnnotatedDataFrame]):
        df = dataframes["__default__"]
        for idx, node in enumerate(self.processing_graph.nodes(), start=1):
            try:
                if node.name in dataframes:
                    df = node.transformer.fit_transform(df, dataframes[node.name])
                else:
                    df = node.transformer.fit_transform(df)
                AnnotatedDataFrame.ensure_type(df)
            except Exception as e:
                msg = ''.join(tc.format_exception(e)[-2:])
                raise RuntimeError(f"Step #{idx} in pipeline ({node}) failed: name_mappings: {node.transformer.name_mappings}\n\
                        {msg}")
        return df


    def on_transform_start(self,
                           step: PipelineElement,
                           adf: AnnotatedDataFrame):
        """
        Default implementation of the on_transform_start callback.

        This output is a message in INFO log level
        """
        if hasattr(step, "transform_start"):
            return

        _log.info(f"[transform] start: {step.__class__.__name__} - {step.name_mappings}")
        start_time = datetime.now(timezone.utc)
        setattr(step, "transform_start", start_time)

        step_name = self.processing_graph[step.uuid].name
        self._processing_stats[step_name] = {
            "input_dataframe_length": adf.shape[0],
            "start_time": start_time
        }

    def on_transform_end(self,
                         step: PipelineElement,
                         adf: AnnotatedDataFrame):
        """
        Default implementation of the on_transform_end callback.

        This outputs a message in INFO log level
        """
        if not hasattr(step, "transform_start"):
            return

        start = getattr(step, "transform_start")
        end_time = datetime.now(timezone.utc)
        delta = (end_time - start).total_seconds()
        delattr(step, "transform_start")
        _log.info(f"[transform] end: {step.__class__.__name__} - {step.name_mappings}: "
                  f"{delta} seconds, {adf.shape[0]} remaining rows)")

        step_name = self.processing_graph[step.uuid].name
        self._processing_stats[step_name] = {
            "processing_time_in_s": delta,
            "output_dataframe_length": adf.shape[0],
            "end_time": end_time
        }

    def __repr__(self) -> str:
        """
        Create the representation string for this :class:`DataProcessingPipeline`.

        :returns: String representation
        """
        return self.to_str()

    def to_str(self, indent_level: int = 0) -> str:
        return self.processing_graph.to_str(indent_level=indent_level)

def polar_data_type_constructor(loader, tag_suffix, node):
    module_whitelist = ["polars.datatypes.classes"]
    for m in module_whitelist:
        if tag_suffix.startswith(m):
            module = importlib.import_module(m)
            object_name = tag_suffix.replace(m + ".","")
            return getattr(module, object_name)

    raise ValueError(f"damast.core.dataprocessing: no constructor registered for {tag_suffix=}")

yaml.SafeLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", polar_data_type_constructor)
