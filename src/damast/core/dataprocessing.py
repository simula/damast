"""
Module containing decorators and classes to model data processing pipelines
"""
import functools
import inspect
from typing import Any, Dict, List, Tuple, Optional

import vaex.ml
from vaex.ml.transformations import Transformer

from .dataframe import AnnotatedDataFrame
from .formatting import DEFAULT_INDENT
from .metadata import DataSpecification, MetaData

__all__ = [
    "input",
    "output",
    "describe",
    "DataProcessingPipeline",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS"
]

DECORATED_DESCRIPTION = '_damast_description'
DECORATED_INPUT_SPECS = '_damast_input_specs'
DECORATED_OUTPUT_SPECS = '_damast_output_specs'


def _get_dataframe(*args, **kwargs) -> AnnotatedDataFrame:
    """
    Extract the dataframe from positional or keyword arguments

    :param args: positional arguments
    :param kwargs: keyword arguments
    :return: The annotated data frame
    :raise KeyError: if a positional argument does not exist and keyword 'df' is missing
    raise TypeError: if the kwargs 'df' is not an AnnotatedDataFrame
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

    The decorated function must return :class:`AnnotatedDataFrame`.

    :param description: description of the action
    """

    def decorator(func):
        setattr(func, DECORATED_DESCRIPTION, description)
        return func

    return decorator


def input(requirements: Dict[str, Any]):
    """
    Specify the input for the decorated function.

    The decorated function must return :class:`AnnotatedDataFrame`.

    :param requirements: List of input requirements
    """

    required_input_specs = DataSpecification.from_requirements(requirements=requirements)

    def decorator(func):
        setattr(func, DECORATED_INPUT_SPECS, required_input_specs)

        @functools.wraps(func)
        def check(*args, **kwargs):
            _df: AnnotatedDataFrame = _get_dataframe(*args, **kwargs)

            fulfillment = _df.get_fulfillment(expected_specs=required_input_specs)
            if fulfillment.is_met():
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Input requirements are not fulfilled:"
                                   f" -- {fulfillment}")

        return check

    return decorator


def output(requirements: Dict[str, Any]):
    """
    Specify the output for the decorated function.

    The decorated function must return :class:`AnnotatedDataFrame`.

    :param requirements: List of input requirements
    """
    required_output_specs = DataSpecification.from_requirements(requirements=requirements)

    def decorator(func):
        return_type = inspect.signature(func).return_annotation
        if return_type != AnnotatedDataFrame:
            raise RuntimeError("output: decorator requires 'AnnotatedDataFrame' to be returned by function")

        setattr(func, DECORATED_OUTPUT_SPECS, required_output_specs)

        @functools.wraps(func)
        def check(*args, **kwargs) -> AnnotatedDataFrame:
            _df: AnnotatedDataFrame = _get_dataframe(*args, **kwargs)
            input_columns = [c for c in _df.column_names]

            adf: AnnotatedDataFrame = func(*args, **kwargs)
            if adf is None:
                raise RuntimeError("output: decorated function must return 'AnnotatedDataFrame', but was 'None'")

            for c in input_columns:
                if c not in adf._dataframe.column_names:
                    raise RuntimeError(f"output: column '{c}' was removed by decorated function."
                                       f" Only adding of columns is permitted.")

            # Ensure that metadata is up to date with the dataframe
            adf.update(expectations=required_output_specs)
            return adf

        return check

    return decorator


class DataProcessingPipeline(Transformer):
    """
    A data-processing pipeline for a sequence of transformers

    :param steps: A list of tuples (name_of_transformer, :class:`Transformer`)
    :raises ValueError: If any of the transformer names are `None`
    :raises AttributeError: If the transformer is missing the :func:`transform` function
    :raises AttributeError: If transformer is missing input or output decoration
    :raises RuntimeError: If the sequence of transformers does not satisfy the sequential requirements
    """

    #: The output specs - as specified by decorators
    output_specs: List[DataSpecification]

    # The processing steps that define this pipeline
    steps: List[Tuple[str, Transformer]]

    def __init__(self, steps: List[Tuple[str, Transformer]]):
        validation_result = self.validate(steps=steps)

        self.steps = validation_result["steps"]
        self.output_specs = validation_result["output_spec"]

    @classmethod
    def validate(cls, steps: List[Tuple[str, Transformer]]) -> Dict[str, Any]:
        """
        Validate the existing pipeline and collect the final (minimal output) data specification

        .. todo::

            Document output

        :param steps: processing steps
        :return:
        """
        # Keep track of the expected (minimal) specs at each step in the pipeline
        current_specs: Optional[List[DataSpecification]] = None
        for name, transformer in steps:

            if name is None:
                raise ValueError(f"{cls.__name__}.validate: missing name processing step")

            if not hasattr(transformer, "transform"):
                raise AttributeError(f"{cls.__name__}.validate: processing step '{name}' does not fulfill the"
                                     f" TransformerMixin requirements - no method 'fit_transform' found")

            if hasattr(transformer.transform, DECORATED_INPUT_SPECS):
                input_specs = getattr(transformer.transform, DECORATED_INPUT_SPECS)
            else:
                raise AttributeError(
                    f"{cls.__name__}.validate: missing input specification for processing step '{name}'")

            if hasattr(transformer.transform, DECORATED_OUTPUT_SPECS):
                output_specs = getattr(transformer.transform, DECORATED_OUTPUT_SPECS)
            else:
                raise AttributeError(
                    f"{cls.__name__}.validate: missing output specification for processing step '{name}'")

            if current_specs is None:
                current_specs = input_specs
            else:
                md = MetaData(columns=current_specs, annotations={})
                fulfillment = md.get_fulfillment(expected_specs=input_specs)
                if not fulfillment.is_met():
                    raise RuntimeError(f"{cls.__name__}.validate: insufficient output declared"
                                       f" from previous step to step '{name}': {fulfillment}")

                current_specs = DataSpecification.merge_lists(current_specs, input_specs)
            current_specs = DataSpecification.merge_lists(current_specs, output_specs)

        return {"steps": steps, "output_spec": current_specs}

    def to_str(self, indent_level: int = 0) -> str:
        """
        Output pipeline as string

        :param indent_level: Indentation per step. It is multiplied by :attr:`DEFAULT_INDENT`.
        :returns: The pipeline in string representation
        """
        hspace = DEFAULT_INDENT * indent_level

        data = hspace + self.__class__.__name__ + "\n"
        for step in self.steps:
            name, transformer = step
            data += hspace + DEFAULT_INDENT + name + ":\n"

            if hasattr(transformer.transform, DECORATED_DESCRIPTION):
                description = getattr(transformer.transform, DECORATED_DESCRIPTION)
                data += hspace + DEFAULT_INDENT * 2 + "description: " + description + "\n"

            data += hspace + DEFAULT_INDENT * 2 + "input:\n"
            input_specs = getattr(transformer.transform, DECORATED_INPUT_SPECS)
            data += DataSpecification.to_str(input_specs, indent_level=indent_level + 4)

            data += hspace + DEFAULT_INDENT * 2 + "output:\n"
            output_specs = getattr(transformer.transform, DECORATED_OUTPUT_SPECS)
            data += DataSpecification.to_str(output_specs, indent_level=indent_level + 4)

        return data

    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Apply pipeline on given annotated dataframe

        :param df: The input dataframe
        :returns: The transformed dataframe
        """
        steps = [t for _, t in self.steps]
        pipeline = vaex.ml.Pipeline(steps)
        return pipeline.transform(dataframe=df)

    def __repr__(self) -> str:
        return self.to_str()
