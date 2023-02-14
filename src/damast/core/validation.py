import functools
import inspect
from typing import Any, Dict, List

from .dataframe import AnnotatedDataFrame

__all__ = [
    "input",
    "output"
]

from .metadata import DataSpecification, ExpectedDataSpecification


def _get_requirements(requirements: List[Dict[str, Any]]) -> List[ExpectedDataSpecification]:
    required_specs = []
    for requirement in requirements:
        for k, v in requirement.items():
            kwargs = v
            kwargs[DataSpecification.Key.name.value] = k
            if DataSpecification.Key.category.value not in kwargs:
                kwargs[DataSpecification.Key.category.value] = None

            required_spec = ExpectedDataSpecification(**kwargs)
            required_specs.append(required_spec)
    return required_specs


def input(requirements: List[Dict[str, Any]]):
    required_specs = _get_requirements(requirements=requirements)

    def decorator(func):
        @functools.wraps(func)
        def check(*args, **kwargs):
            if "df" not in kwargs:
                raise KeyError("Missing keyword argument 'df' to define the AnnotatedDataFrame")
            assert isinstance(kwargs["df"], AnnotatedDataFrame)

            _df: AnnotatedDataFrame = kwargs["df"]
            fulfillment = _df.get_fulfillment(expected_specs=required_specs)
            if fulfillment.is_met():
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Input requirements are not fulfilled:"
                                   f" -- {fulfillment}")

        return check

    return decorator


def output(requirements: List[Dict[str, Any]]):
    required_output_specs = _get_requirements(requirements=requirements)

    def decorator(func):
        return_type = inspect.signature(func).return_annotation
        if return_type != AnnotatedDataFrame:
            raise RuntimeError("output: decorator requires 'AnnotatedDataFrame' to be returned by function")

        @functools.wraps(func)
        def check(*args, **kwargs) -> AnnotatedDataFrame:
            if "df" not in kwargs:
                raise KeyError("output: decorated function misses keyword argument"
                               " 'df' to define the AnnotatedDataFrame")
            assert isinstance(kwargs["df"], AnnotatedDataFrame)

            adf: AnnotatedDataFrame = func(*args, **kwargs)
            if adf is None:
                raise RuntimeError("output: decorated function must return 'AnnotatedDataFrame', but was 'None'")

            # Ensure that metadata is up to date with the dataframe
            adf.update(expectations=required_output_specs)
            return adf

        return check

    return decorator
