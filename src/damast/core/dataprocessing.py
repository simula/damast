"""
Module containing decorators and classes to model data processing pipelines
"""
import functools
import inspect
from typing import Any, Dict, List, Tuple

from sklearn.base import TransformerMixin

from .dataframe import AnnotatedDataFrame
from .metadata import DataSpecification, MetaData

__all__ = [
    "input",
    "output",
    "DataProcessingPipeline",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS"
]

DECORATED_INPUT_SPECS = '_damast_input_specs'
DECORATED_OUTPUT_SPECS = '_damast_output_specs'


def input(requirements: List[Dict[str, Any]]):
    """
    Specify the input for the decorated function.

    The decorated function must return 'AnnotatedDataFrame'.

    :param requirements: List of input requirements
    """

    required_input_specs = DataSpecification.from_requirements(requirements=requirements)

    def decorator(func):
        setattr(func, DECORATED_INPUT_SPECS, required_input_specs)

        @functools.wraps(func)
        def check(*args, **kwargs):
            if "df" not in kwargs:
                raise KeyError("Missing keyword argument 'df' to define the AnnotatedDataFrame")
            assert isinstance(kwargs["df"], AnnotatedDataFrame)

            _df: AnnotatedDataFrame = kwargs["df"]
            fulfillment = _df.get_fulfillment(expected_specs=required_input_specs)
            if fulfillment.is_met():
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Input requirements are not fulfilled:"
                                   f" -- {fulfillment}")

        return check

    return decorator


def output(requirements: List[Dict[str, Any]]):
    """
    Specify the output for the decorated function.

    The decorated function must return 'AnnotatedDataFrame'.

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
            if "df" not in kwargs:
                raise KeyError("output: decorated function misses keyword argument"
                               " 'df' to define the AnnotatedDataFrame")
            assert isinstance(kwargs["df"], AnnotatedDataFrame)

            _df = kwargs["df"]
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


class DataProcessingPipeline:
    #: The output specs - as specified by decorators
    output_specs: List[DataSpecification] = None

    # The processing steps that define this pipeline
    steps: List[TransformerMixin] = None

    def __init__(self, steps: List[Tuple[str, TransformerMixin]]):
        validation_result = self.validate(steps=steps)

        self.steps: List[Tuple[str, TransformerMixin]] = validation_result["steps"]
        self.output_specs: List[DataSpecification] = validation_result["output_spec"]

    @classmethod
    def validate(cls, steps: List[Tuple[str, TransformerMixin]]) -> Dict[str, Any]:
        """
        Validate the existing pipeline and collect the final (minimal output) data specification

        :param steps: processing steps
        :return: Tul
        """
        # Keep track of the expected (minimal) specs at each step in the pipeline
        current_specs: List[DataSpecification] = None
        for step in steps:
            name, transformer = step

            if name is None:
                raise RuntimeError(f"{cls.__name__}.validate: missing name processing step")

            if getattr(transformer, "transform") is None:
                raise RuntimeError(f"{cls.__name__}.validate: processing step '{name}' does not fulfill the"
                                   f" TransformerMixin requirements - no method 'fit_transform' found")

            input_specs = getattr(transformer.transform, DECORATED_INPUT_SPECS)
            if input_specs is None:
                raise RuntimeError(f"{cls.__name__}.validate: missing input specification for processing step '{name}'")

            output_specs = getattr(transformer.transform, DECORATED_OUTPUT_SPECS)
            if output_specs is None:
                raise RuntimeError(
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
