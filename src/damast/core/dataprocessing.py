"""
Module containing decorators and classes to model data processing pipelines
"""
from __future__ import annotations

import copy
import functools
import importlib
import inspect
import re
import tempfile

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import vaex.ml
from vaex.ml.transformations import Transformer

from .dataframe import AnnotatedDataFrame
from .formatting import DEFAULT_INDENT
from .metadata import ArtifactSpecification, DataSpecification, MetaData

__all__ = [
    "artifacts",
    "input",
    "output",
    "describe",
    "DataProcessingPipeline",
    "DECORATED_ARTIFACT_SPECS",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS",
    "PipelineElement",
]


DECORATED_DESCRIPTION = "_damast_description"
"""Attribute description for :func:`describe`"""


DECORATED_ARTIFACT_SPECS = "_damast_artifact_specs"
"""Attribute description for :func:`artifacts`"""


DECORATED_INPUT_SPECS = "_damast_input_specs"
"""Attribute description for :func:`input`"""

DECORATED_OUTPUT_SPECS = "_damast_output_specs"
"""Attribute description for :func:`output`"""

DAMAST_PIPELINE_SUFFIX: str = ".damast.ppl"
"""Suffix of :class:`DataProcessingPipeline` files created in :func:`DataProcessingPipeline.save`
and used by :func:`DataProcessingPipeline.load`"""

VAEX_STATE_SUFFIX: str = ".vaex-state.json"
"""Suffix of :class:`DataProcessingPipeline` files created in :func:`DataProcessingPipeline.save_state`
and used by :func:`DataProcessingPipeline.load_state`"""


def _get_dataframe(*args, **kwargs) -> AnnotatedDataFrame:
    """
    Extract the dataframe from positional or keyword arguments

    :param args: positional arguments
    :param kwargs: keyword arguments
    :return: The annotated data frame
    :raise KeyError: if a positional argument does not exist and keyword :code:`df` is missing
    :raise TypeError: if the kwargs 'df' is not an AnnotatedDataFrame
    """
    _df: AnnotatedDataFrame
    if len(args) >= 2 and isinstance(args[1], AnnotatedDataFrame):
        _df = args[1]
    elif "df" not in kwargs:
        raise KeyError("Missing keyword argument 'df' to define the AnnotatedDataFrame")
    elif not isinstance(kwargs["df"], AnnotatedDataFrame):
        raise TypeError("Argument 'df' is not an AnnotatedDataFrame")
    else:
        _df = kwargs["df"]
    return _df


def describe(description: str):
    """
    Specify the description for the transformation for the decorated function.

    The decorated function must return :class:`damast.core.AnnotatedDataFrame`.

    :param description: description of the action
    """

    def decorator(func):
        setattr(func, DECORATED_DESCRIPTION, description)
        return func

    return decorator


def input(requirements: Dict[str, Any]):
    """
    Specify the input for the decorated function.

    The decorated function must return :class:`damast.core.AnnotatedDataFrame`.

    :param requirements: List of input requirements
    """

    required_input_specs = DataSpecification.from_requirements(
        requirements=requirements
    )

    def decorator(func):
        setattr(func, DECORATED_INPUT_SPECS, required_input_specs)

        @functools.wraps(func)
        def check(*args, **kwargs):
            _df: AnnotatedDataFrame = _get_dataframe(*args, **kwargs)
            # Ensure that name mapping are applied correctly
            assert isinstance(args[0], PipelineElement)
            fulfillment = _df.get_fulfillment(expected_specs=args[0].input_specs)
            if fulfillment.is_met():
                return func(*args, **kwargs)
            else:
                raise RuntimeError(
                    "Input requirements are not fulfilled:" f" -- {fulfillment}"
                )

        return check

    return decorator


def output(requirements: Dict[str, Any]):
    """
    Specify the output for the decorated function.

    The decorated function must return :class:`damast.core.AnnotatedDataFrame`.

    :param requirements: List of input requirements
    """
    required_output_specs = DataSpecification.from_requirements(
        requirements=requirements
    )

    def decorator(func):
        return_type = inspect.signature(func).return_annotation
        if return_type != AnnotatedDataFrame:
            raise RuntimeError(
                "output: decorator requires 'AnnotatedDataFrame' to be returned by function"
            )

        setattr(func, DECORATED_OUTPUT_SPECS, required_output_specs)

        @functools.wraps(func)
        def check(*args, **kwargs) -> AnnotatedDataFrame:
            setattr(func, DECORATED_OUTPUT_SPECS, required_output_specs)

            _df: AnnotatedDataFrame = _get_dataframe(*args, **kwargs)
            input_columns = [c for c in _df.column_names]

            adf: AnnotatedDataFrame = func(*args, **kwargs)
            if adf is None:
                raise RuntimeError(
                    f"output: decorated function {func} must return 'AnnotatedDataFrame',"
                    f" but was 'None'"
                )
            elif not isinstance(adf, AnnotatedDataFrame):
                raise RuntimeError(
                    f"output: decorated function {func} must return 'AnnotatedDataFrame', but was '"
                    f"{type(adf)}"
                )

            for c in input_columns:
                if c not in adf._dataframe.column_names:
                    raise RuntimeError(
                        f"output: column '{c}' was removed by decorated function."
                        f" Only adding of columns is permitted."
                    )

            # Ensure that name mapping are applied correctly
            assert isinstance(args[0], PipelineElement)

            # Ensure that metadata is up to date with the dataframe
            adf.update(expectations=args[0].output_specs)
            return adf

        return check

    return decorator


def artifacts(requirements: Dict[str, Any]):
    """
    Specify the output for the decorated function.

    The decorated function must return :class:`damast.core.AnnotatedDataFrame`.

    :param requirements: List of input requirements
    """
    required_artifact_specs = ArtifactSpecification(requirements=requirements)

    def decorator(func):
        return_type = inspect.signature(func).return_annotation
        if return_type != AnnotatedDataFrame:
            raise RuntimeError(
                "artifacts: decorator requires 'AnnotatedDataFrame' to be returned by function"
            )

        setattr(func, DECORATED_ARTIFACT_SPECS, required_artifact_specs)

        # When a pipeline does generate artifacts, then it might not provide any output, but serves only as
        # passthrough element. Hence, per default set an empty output spec if there is none.
        if not hasattr(func, DECORATED_OUTPUT_SPECS):
            setattr(
                func,
                DECORATED_OUTPUT_SPECS,
                DataSpecification.from_requirements(requirements={}),
            )

        @functools.wraps(func)
        def check(*args, **kwargs) -> AnnotatedDataFrame:
            result = func(*args, **kwargs)

            if not isinstance(args[0], PipelineElement):
                raise RuntimeError(
                    f"{args[0].__class__.__name__} must inherit from PipelineElement"
                )

            # Validate the spec with respect to the existing parent pipeline
            try:
                instance = args[0]
                required_artifact_specs.validate(
                    base_dir=instance.parent_pipeline.base_dir
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"artifacts: {func} is expected to generate an artifact. "
                    f" Pipeline element ran as part of pipeline: '{instance.parent_pipeline.name}'"
                    f" -- {e}"
                )

            return result

        return check

    return decorator


class PipelineElement(Transformer):
    """
    Allow to get the reference to a parent pipeline
    """

    #: Pipeline in which context this processor will be run
    parent_pipeline: DataProcessingPipeline

    #: Map names of input and outputs for a particular pipeline
    _name_mappings: Dict[str, str]

    def set_parent(self, pipeline: DataProcessingPipeline):
        """
        Sets the parent pipeline for this pipeline element

        :param pipeline: Parent pipeline
        """
        self.parent_pipeline = pipeline

    @property
    def name_mappings(self):
        if not hasattr(self, "_name_mappings"):
            self._name_mappings = {}
        return self._name_mappings

    def get_name(self, name: str) -> Any:
        """
        Add the fully resolved input/output name for this key.

        :param name: Name as used in the input spec, or pattern "{{x}}_suffix" in order to create a dynamic
                     output based an existing and renameable input
        :return: Name for this input after resolving name mappings and references
        """
        if name in self.name_mappings:
            return self.name_mappings[name]

        # Allow to use patterns, so that an existing input
        # reference can be reused for dynamic labelling
        while re.match("{{\\w}}", name):
            for match in re.finditer("{{\\w}}", name):
                resolved_name = match.group()[2:-2]
                if resolved_name in self.name_mappings:
                    resolved_name = self.name_mappings[resolved_name]

                name = name.replace(match.group(), resolved_name)
        return name

    @property
    def input_specs(self) -> List[DataSpecification]:
        if not hasattr(self.transform, DECORATED_INPUT_SPECS):
            raise AttributeError(
                f"{self.__class__.__name__}.validate: missing input specification"
                f" for processing step '{self.__class__.__name__}'"
            )

        generic_spec = getattr(self.transform, DECORATED_INPUT_SPECS)
        specs = copy.deepcopy(generic_spec)
        for spec in specs:
            spec.name = self.get_name(spec.name)

        return specs

    @property
    def output_specs(self) -> List[DataSpecification]:
        if not hasattr(self.transform, DECORATED_OUTPUT_SPECS):
            raise AttributeError(
                f"{self.__class__.__name__}.validate: missing output specification"
                f" for processing step '{self.__class__.__name__}'"
            )

        generic_spec = getattr(self.transform, DECORATED_OUTPUT_SPECS)
        specs = copy.deepcopy(generic_spec)
        for spec in specs:
            spec.name = self.get_name(spec.name)

        return specs

    @classmethod
    def create_new(cls,
                   module_name: str,
                   class_name: str,
                   name_mappings: Dict[str, Any] = {}):
        if module_name is None:
            raise ValueError(f"{cls.__name__}.create_new: missing 'module_name'")

        if class_name is None:
            raise KeyError(f"{cls.__name__}.create_new: missing 'class_name'")

        p_module = importlib.import_module(module_name)
        if hasattr(p_module, class_name):
            klass = getattr(p_module, class_name)
        else:
            raise ImportError(f"{cls.__name__}.create_new: could not load '{class_name}' from '{p_module}'")

        instance = klass()
        instance._name_mappings = name_mappings
        return instance

    def __iter__(self):
        yield "module_name", f"{self.__class__.__module__}"
        yield "class_name", f"{self.__class__.__qualname__}"
        yield "name_mappings", self.name_mappings

    def __eq__(self, other):
        return dict(self) == dict(other)


class DataProcessingPipeline(PipelineElement):
    """
    A data-processing pipeline for a sequence of transformers

    :param name: Name of the pipeline
    :param base_dir: Base directory towards which transformer output which be relative

    :raises ValueError: If any of the transformer names are `None`
    :raises AttributeError: If the transformer is missing the :func:`transform` function
    :raises AttributeError: If transformer is missing input or output decoration
    :raises RuntimeError: If the sequence of transformers does not satisfy the sequential requirements
    """

    #: Name of the processing pipeline
    name: str

    #: Base path (which is forwarded to transformers, when calling
    #: transform)
    base_dir: Path

    #: The output specs - as specified by decorators
    _output_specs: List[DataSpecification]

    #: Check if the pipeline is ready to be run
    is_ready: bool
    # The processing steps that define this pipeline
    steps: List[Tuple[str, PipelineElement]]

    def __init__(self,
                 name: str,
                 base_dir: Union[str, Path] = None,
                 steps: List[Tuple[str, Union[Dict[str, Any], PipelineElement]]] = None
                 ):
        self.name = name
        if base_dir is None:
            base_dir = tempfile.gettempdir()

        self.base_dir = Path(base_dir)

        self._output_specs = None
        if steps is None or len(steps) == 0:
            self.steps = []
        else:
            name, instance = steps[0]
            if isinstance(instance, PipelineElement):
                self.steps = steps
            elif isinstance(instance, dict):
                self.steps = []
                for step in steps:
                    self.steps.append([step[0], PipelineElement.create_new(**step[1])])
            else:
                raise ValueError(f"{self.__class__.__name__}.__init__: could not instantiate PipelineElement"
                                 f" from {type(steps[1])}")
        self.is_ready = False

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
        name_mappings: Dict[str, str] = None,
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

        self.steps.append([name, transformer])
        self.is_ready = False
        return self

    @classmethod
    def validate(
        cls, steps: List[Tuple[str, PipelineElement]], metadata: MetaData
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
        for name, transformer in steps:
            if name is None:
                raise ValueError(
                    f"{cls.__name__}.validate: missing name processing step"
                )

            if not hasattr(transformer, "transform"):
                raise AttributeError(
                    f"{cls.__name__}.validate: processing step '{name}' does not fulfill the"
                    f" TransformerMixin requirements - no method 'fit_transform' found"
                )

            input_specs = transformer.input_specs
            output_specs = transformer.output_specs

            md = MetaData(columns=current_specs, annotations=[])
            fulfillment = md.get_fulfillment(expected_specs=input_specs)
            if not fulfillment.is_met():
                raise RuntimeError(
                    f"{cls.__name__}.validate: Input requirements are not fulfilled. "
                    f"Current input available (at step '{name}'): {fulfillment}"
                )

            current_specs = DataSpecification.merge_lists(current_specs, input_specs)
            current_specs = DataSpecification.merge_lists(current_specs, output_specs)

        return {"steps": steps, "output_spec": current_specs}

    def to_str(self, indent_level: int = 0) -> str:
        """
        Output pipeline as string.

        :param indent_level: Indentation per step.
            It is multiplied by :attr:`damast.core.formatting.DEFAULT_INDENT`.
        :returns: The pipeline in string representation
        """
        hspace = DEFAULT_INDENT * indent_level

        data = hspace + self.__class__.__name__ + "\n"
        for step in self.steps:
            name, transformer = step
            data += hspace + DEFAULT_INDENT + name + ":\n"

            if hasattr(transformer.transform, DECORATED_DESCRIPTION):
                description = getattr(transformer.transform, DECORATED_DESCRIPTION)
                data += (
                    hspace + DEFAULT_INDENT * 2 + "description: " + description + "\n"
                )

            data += hspace + DEFAULT_INDENT * 2 + "input:\n"
            data += DataSpecification.to_str(
                transformer.input_specs, indent_level=indent_level + 4
            )

            data += hspace + DEFAULT_INDENT * 2 + "output:\n"
            data += DataSpecification.to_str(
                transformer.output_specs, indent_level=indent_level + 4
            )

        return data

    def save(self, dir: Union[str, Path]) -> Path:
        """
        Save the processing pipeline

        :param dir: directory where to save this pipeline
        """
        filename = dir / f"{self.name}{DAMAST_PIPELINE_SUFFIX}"
        with open(filename, "w") as f:
            yaml.dump(dict(self), f)
        return filename

    def __iter__(self):
        yield "name", self.name
        yield "base_dir", str(self.base_dir)
        yield "steps", [[step_name, dict(pipeline_element)] for step_name, pipeline_element in self.steps]

    def save_state(self,
                   df: AnnotatedDataFrame,
                   dir: Union[str, Path]) -> Path:
        """
        Save the processing pipeline

        :param dir: directory where to save this pipeline
        """
        filename = dir / f"{self.name}{VAEX_STATE_SUFFIX}"
        df._dataframe.state_write(file=filename)
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
            files = [x for x in path.glob(basename)]
        elif path.is_file():
            files = [path]

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

        return cls(data_dict=data)

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
        dir = Path(dir)

        files = [x for x in dir.glob(basename)]
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

        df._dataframe.state_load(file=filename)
        return df

    def prepare(self, df: AnnotatedDataFrame) -> DataProcessingPipeline:
        """
        Prepare the pipeline by validating all step that define the pipeline.

        A :class:`DataProcessingPipeline` must be prepared before execution.

        :returns: This instance of data processing pipeline
        """

        # At this stage, ensure that the dataframe conforms to the metadata
        try:
            df.validate_metadata()
        except ValueError as e:
            raise RuntimeError(
                f"{self.__class__.__name__}.prepare: specification of the provided AnnotatedDataFrame"
                f" does not match its data. If you modified the frame after construction ensure"
                f" consistency between 'data' and 'metadata' by using "
                f" AnnotatedDataFrame.validate_metadata()"
                f" -- {e}"
            )

        validation_result = self.validate(steps=self.steps, metadata=df._metadata)
        self.is_ready = True

        self.steps: List[Tuple[str, Transformer]] = validation_result["steps"]
        self._output_specs: List[DataSpecification] = validation_result["output_spec"]

        return self

    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Apply pipeline on given annotated dataframe

        .. note::
            If any filters are applied by the pipeline, these are executed at the end of the
            transformation.

        :param df: The input dataframe
        :returns: The transformed dataframe
        """
        if not isinstance(df, AnnotatedDataFrame):
            raise TypeError(
                f"{self.__class__.__name__}.transform"
                f" expected an annotated dataframe for processing, "
                f"got {type(df)}"
            )

        if not self.is_ready:
            self.prepare(df=df)

        steps = [t for _, t in self.steps]
        pipeline = vaex.ml.Pipeline(steps)

        if df.is_empty():
            raise RuntimeError(
                f"{self.__class__.__name__}.transform: there is no data available to transform"
            )

        adf = pipeline.transform(dataframe=df)
        assert isinstance(adf, AnnotatedDataFrame)

        # Remove all rows that have been filtered from the data-frame
        adf._dataframe = adf._dataframe.extract()
        return adf

    def __repr__(self) -> str:
        """
        Create the representation string for this :class:`DataProcessingPipeline`.

        :returns: String representation
        """
        return self.to_str()
