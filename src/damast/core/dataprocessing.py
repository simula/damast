"""
Module containing decorators and classes to model data processing pipelines
"""
from __future__ import annotations

import copy
import importlib
import inspect
import os
import re
import tempfile
import traceback as tc
from collections import OrderedDict
from datetime import datetime, timezone
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import yaml
from tqdm import tqdm

import damast.version
from damast.core.processing_graph import Node, ProcessingGraph
from damast.core.transformations import PipelineElement

from .constants import DAMAST_DEFAULT_DATASOURCE
from .dataframe import AnnotatedDataFrame
from .formatting import DEFAULT_INDENT
from .metadata import DataSpecification, MetaData
from .transformations import PipelineElement

__all__ = [
    "artifacts",
    "input",
    "output",
    "describe",
    "DataProcessingPipeline",
    "PipelineElement",
]

logger: Logger = getLogger(__name__)

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
    _name_mappings: Dict[str, Dict[str, str]]
    _processing_stats: Dict[str, Dict[str, Any]]

    _meta: Dict[str, str]

    def __init__(self, *,
                 name: str,
                 description: str = "",
                 base_dir: Union[str, Path] = tempfile.gettempdir(),
                 processing_graph: List[Tuple[str, Union[Dict[str, Any], PipelineElement]]] | ProcessingGraph = None,
                 inplace_transformation: bool = False,
                 name_mappings: Dict[str, Dict[str, str]] = { DAMAST_DEFAULT_DATASOURCE: {}},
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
            if len(transformer.input_specs) == 1 and DAMAST_DEFAULT_DATASOURCE not in name_mappings:
                transformer._name_mappings = { DAMAST_DEFAULT_DATASOURCE: name_mappings.copy() }

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
            operator._name_mappings = name_mappings.copy()

        self.processing_graph.join(
                name=name,
                processing_graph=data_source.processing_graph if data_source else None,
                operator=operator
        )
        self.is_ready = False
        return self

    @classmethod
    def validate(
            cls, processing_graph: ProcessingGraph, metadata: dict[str, MetaData]
    ) -> Dict[str, Any]:
        """
        Validate the existing pipeline and collect the minimal input and output data specification.

        :param steps: processing steps
        :param metadata: the input metadata for this pipeline as dictionary of input datasource to metadata
        :return: The minimal output specification for this pipeline
        """
        processing_graph.clear_state()

        current_specs: dict[list[DataSpecification]] = {}
        for ds_name, ds_metadata in metadata.items():
            # Keep track of the expected (minimal) specs at each step in the pipeline
            current_specs[ds_name] = copy.deepcopy(
                ds_metadata.columns
            )

        current_node_output_spec = {} # key -> node uuid
        for idx, node in enumerate(processing_graph.nodes(), start=1):
            if node.name is None:
                raise ValueError(
                    f"{cls.__name__}.validate: missing name processing step"
                )

            if not hasattr(node.transformer, "transform"):
                raise AttributeError(
                    f"{cls.__name__}.validate: processing step '{node.name}' does not fulfill the"
                    f" TransformerMixin requirements - no method 'fit_transform' found"
                )

            try:
                logger.info("#{idx} validate {node}")
                if node.name in current_specs and node.is_datasource():
                    datasource = node.name
                    md = MetaData(columns=current_specs[datasource], annotations=[])
                    node_input_specs = list(node.transformer.input_specs.values())[0]
                    fulfillment = md.get_fulfillment(expected_specs=node_input_specs)
                    if not fulfillment.is_met():
                        raise RuntimeError(
                            f"{cls.__name__}.validate: Input requirements are not fulfilled (for datasource '{node.transformer}'). "
                            f"Current input available (at step '{node.name}'): {fulfillment}"
                        )
                    node.validation_output_spec = DataSpecification.merge_lists(node_input_specs,
                                                                                current_specs[datasource])
                    current_node_output_spec = node.validation_output_spec
                else:
                    for i in node.inputs():
                        for from_node,to_node,data in processing_graph._graph.in_edges(node, data=True):
                            slot = data['slot']

                            md = MetaData(columns=from_node.validation_output_spec, annotations=[])
                            fulfillment = md.get_fulfillment(expected_specs=node.transformer.input_specs[slot])
                            if not fulfillment.is_met():
                                raise RuntimeError(
                                    f"{cls.__name__}.validate: Input requirements are not fulfilled {node} (for {slot=}). "
                                     f"Current input available (at step '{node.name}'): {fulfillment}"
                                )

                            if not node.validation_output_spec:
                                node.validation_output_spec = DataSpecification.merge_lists(from_node.validation_output_spec, node.transformer.output_specs)
                            else:
                                node.validation_output_spec = DataSpecification.merge_lists(node.transformer.output_specs, node.validation_output_spec)

                            current_node_output_spec = node.validation_output_spec
            except Exception as e:
                msg = ''.join(tc.format_exception(e)[-2:])
                raise RuntimeError(f"Validation of step #{idx} in pipeline ({node}) failed: name_mappings: {node.transformer.name_mappings}"
                                   f"{msg}")

        return {"processing_graph": processing_graph, "output_spec": node.validation_output_spec}

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

    def prepare(self,
                name_mappings: dict[str, dict[str, any]] | None = None,
                **dataframes) -> DataProcessingPipeline:
        """
        Prepare the pipeline by applying necessary name mapping and validating all steps that define the pipeline.

        A :class:`DataProcessingPipeline` must be prepared before execution.

        :returns: This instance of data processing pipeline
        """
        if not dataframes:
            raise RuntimeError("DataProcessingPipeline.prepare: missing dataframes")

        pipeline = copy.deepcopy(self)

        # Update the pipeline mapping with a temporary overlay
        if name_mappings:
            for df_name, df_name_mapping in name_mappings.items():
                if df_name in pipeline.name_mappings:
                    pipeline.name_mappings[df_name].update(name_mappings[df_name])
                else:
                    pipeline.name_mappings[df_name] = name_mappings[df_name]

        # At this stage, ensure that the dataframes conforms to their metadata
        metadata = OrderedDict()
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
            if name in pipeline.name_mappings:
                for k, v in pipeline.name_mappings[name].items():
                    for node in pipeline.processing_graph.nodes():
                        pipeline_element = node.transformer

                        for dataset, mapping in pipeline_element.name_mappings.items():
                            for from_val, to_val in mapping.items():
                                if to_val == k:
                                    pipeline_element.name_mappings[dataset][from_val] = v

            metadata[name] = df.metadata

        validation_result = pipeline.validate(processing_graph=pipeline.processing_graph,
                                          metadata=metadata)

        pipeline.processing_graph = validation_result["processing_graph"]
        pipeline._output_specs: List[DataSpecification] = validation_result["output_spec"]

        pipeline.is_ready = True
        return pipeline

    def transform(self,
                  df: AnnotatedDataFrame,
                  name_mappings: dict[str, dict[str,any]] | None = None,
                  verbose: bool = False,
                  **kwargs) -> AnnotatedDataFrame:
        """
        Apply pipeline on given annotated dataframe

        .. note::
            If any filters are applied by the pipeline, these are executed at the end of the
            transformation.

        :param df: The input dataframe
        :returns: The transformed dataframe
        """
        dataframes = { DAMAST_DEFAULT_DATASOURCE: df }
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

        pipeline = self.prepare(name_mappings=name_mappings, **dataframes)

        for name, df in dataframes.items():
            if df.is_empty():
                raise RuntimeError(
                    f"{self.__class__.__name__}.transform: there is no data available to transform using datasource '{name}'"
            )

        if pipeline._inplace_transformation:
            in_dataframes = dataframes
        else:
            in_dataframes = {x: copy.deepcopy(y) for x,y in dataframes.items()}

        adf = pipeline._run(in_dataframes, verbose=verbose)
        assert isinstance(adf, AnnotatedDataFrame)

        adf.validate_metadata(validation_mode=damast.core.ValidationMode.UPDATE_METADATA)
        return adf

    def _run(self,
             dataframes: dict[str, AnnotatedDataFrame],
             verbose: bool = True) -> AnnotatedDataFrame:
        iterator = enumerate(self.processing_graph.nodes(), start=1)
        if not verbose:
            iterator = tqdm(iterator, desc="Step ", total=len(self.processing_graph))

        for idx, node in iterator:
            # ensure clean state
            if node.result:
                self.processing_graph.clear_state()

            try:
                logger.info("#{idx} run {node}")
                if node.name in dataframes and node.is_datasource():
                    node.result = node.transformer.fit_transform(dataframes[node.name])
                    AnnotatedDataFrame.ensure_type(node.result)
                else:
                    node.result = self.processing_graph.execute(node)

                if verbose:
                    print(f"Preview step #{idx} {node} (1 row)")
                    print(f"{node.result.head(1).collect()}")
            except Exception as e:
                msg = ''.join(tc.format_exception(e)[-2:])
                for slot, df in self.processing_graph.get_current_inputs(node).items():
                    msg += "{slot=} "
                    msg += "     {df.head(1).collect)}"

                if 'DAMAST_INTERACTIVE' in os.environ:
                    if str(os.environ['DAMAST_INTERACTIVE']).lower() == "true":
                        breakpoint()

                raise RuntimeError(f"Step #{idx} in pipeline ({node}) failed: name_mappings: {node.transformer.name_mappings}\n\
                        {msg}")
        return node.result


    def on_transform_start(self,
                           step: PipelineElement,
                           adf: AnnotatedDataFrame):
        """
        Default implementation of the on_transform_start callback.

        This output is a message in INFO log level
        """
        if hasattr(step, "transform_start"):
            return

        logger.info(f"[transform] start: {step.__class__.__name__} - {step.name_mappings}")
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
        logger.info(f"[transform] end: {step.__class__.__name__} - {step.name_mappings}: "
                  f"{delta} seconds, {adf.shape[0]} remaining rows)")

        step_name = self.processing_graph[step.uuid].name
        self._processing_stats[step_name].update({
            "processing_time_in_s": delta,
            "output_dataframe_length": adf.shape[0],
            "end_time": end_time
        })

    def __repr__(self) -> str:
        """
        Create the representation string for this :class:`DataProcessingPipeline`.

        :returns: String representation
        """
        return self.to_str()

    def to_str(self, indent_level: int = 0) -> str:
        return self.processing_graph.to_str(indent_level=indent_level)

    def __deepcopy__(self, memo) -> DataProcessingPipeline:
        pipeline = copy.copy(self)

        pipeline._name_mappings = copy.deepcopy(self._name_mappings)
        pipeline._processing_stats = copy.deepcopy(self._processing_stats)
        pipeline.processing_graph = copy.deepcopy(self.processing_graph)
        pipeline._output_specs = [copy.deepcopy(x) for x in self._output_specs]
        return pipeline

def polar_data_type_constructor(loader, tag_suffix, node):
    module_whitelist = ["polars.datatypes.classes"]
    for m in module_whitelist:
        if tag_suffix.startswith(m):
            module = importlib.import_module(m)
            object_name = tag_suffix.replace(m + ".","")
            return getattr(module, object_name)

    raise ValueError(f"damast.core.dataprocessing: no constructor registered for {tag_suffix=}")

yaml.SafeLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", polar_data_type_constructor)
