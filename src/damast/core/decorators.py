import functools
import inspect
from collections import OrderedDict

from damast.core.dataframe import AnnotatedDataFrame

from .constants import (
    DAMAST_DEFAULT_DATASOURCE,
    DECORATED_ARTIFACT_SPECS,
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    )
from .metadata import ArtifactSpecification, DataSpecification
from .transformations import PipelineElement


def _get_dataframe(*args, **kwargs) -> AnnotatedDataFrame:
    """
    Extract the dataframe from positional or keyword arguments
    :param datasource: the expected name of the datasource
    :param args: positional arguments
    :param kwargs: keyword arguments
    :return: The annotated data frame
    :raise KeyError: if a positional argument does not exist and keyword :code:`df` is missing
    :raise TypeError: if the kwargs 'df' is not an AnnotatedDataFrame
    """
    _df: AnnotatedDataFrame

    arguments = kwargs.copy()
    if 'datasource_argname' in arguments:
        argname = arguments['datasource_argname']
    else:
        argname = DAMAST_DEFAULT_DATASOURCE

    arg_index = 0
    for parameter in inspect.signature(args[0].transform).parameters:
        if parameter not in kwargs:
            arg_index += 1
            arguments[parameter] = args[arg_index]

    datasource = arguments[argname]
    if not isinstance(datasource, AnnotatedDataFrame):
        raise TypeError(f"Argument {argname} is not an AnnotatedDataFrame")

    return datasource

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


def input(requirements: dict[str, any], label: str | None = None):
    """
    Specify the input for the decorated function.

    The decorated function must return :class:`damast.core.AnnotatedDataFrame`.

    :param requirements: List of input requirements
    """

    if label is None:
        label = DAMAST_DEFAULT_DATASOURCE

    required_input_specs = DataSpecification.from_requirements(
        requirements=requirements
    )

    def decorator(func):
        if not hasattr(func, DECORATED_INPUT_SPECS):
            setattr(func, DECORATED_INPUT_SPECS, OrderedDict())

        input_specs = getattr(func, DECORATED_INPUT_SPECS)
        if label in input_specs:
            raise KeyError(f"input for '{label}' has already been specified")

        parameters = list(inspect.signature(func).parameters.keys())
        if label not in parameters:
            for x,_ in parameters:
                if x == label:
                    break

                if x not in input_specs:
                    raise KeyError(f"input for '{x}' unknown, but needs to be before '{label}' - please validate decorator order")

        input_specs[label] = required_input_specs

        @functools.wraps(func)
        def check(*args, **kwargs):
            _df: AnnotatedDataFrame = _get_dataframe(*args, datasource_argname=label, **kwargs)
            # Ensure that name mapping are applied correctly
            assert isinstance(args[0], PipelineElement)

            pipeline_element = args[0]
            fulfillment = _df.get_fulfillment(expected_specs=pipeline_element.input_specs[label])
            if fulfillment.is_met():
                if hasattr(pipeline_element, "parent_pipeline"):
                    # if this is the last datasource parameter, this is also the last input decorators
                    # so the transform can start
                    if label == parameters[-1]:
                        getattr(pipeline_element, "parent_pipeline").on_transform_start(pipeline_element, adf=_df)
                return func(*args, **kwargs)

            raise RuntimeError(
                "Input requirements are not fulfilled:" f" -- {fulfillment}"
            )

        return check

    return decorator


def output(requirements: dict[str, any]):
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
        if return_type != AnnotatedDataFrame and return_type != 'AnnotatedDataFrame':
            raise RuntimeError(
                f"output: decorator requires 'AnnotatedDataFrame' to be returned by function - but was '{return_type}'"
            )

        setattr(func, DECORATED_OUTPUT_SPECS, required_output_specs)

        @functools.wraps(func)
        def check(*args, **kwargs) -> AnnotatedDataFrame:
            setattr(func, DECORATED_OUTPUT_SPECS, required_output_specs)

            _df: AnnotatedDataFrame = _get_dataframe(*args, **kwargs)
            input_columns = list(_df.column_names)

            adf: AnnotatedDataFrame = func(*args, **kwargs)
            if adf is None:
                raise RuntimeError(
                    f"output: decorated function {func} must return 'AnnotatedDataFrame',"
                    f" but was 'None'"
                )

            if not isinstance(adf, AnnotatedDataFrame):
                raise RuntimeError(
                    f"output: decorated function {func} must return 'AnnotatedDataFrame', but was '"
                    f"{type(adf)}"
                )

            for c in input_columns:
                if c not in adf.column_names:
                    raise RuntimeError(
                        f"output: column '{c}' was removed by decorated function."
                        f" Only adding of columns is permitted."
                    )

            # Ensure that name mapping are applied correctly
            assert isinstance(args[0], PipelineElement)
            pipeline_element = args[0]

            parent_pipeline = None
            if hasattr(pipeline_element, "parent_pipeline"):
                parent_pipeline = getattr(pipeline_element, "parent_pipeline")

            try:
                # Ensure that metadata is up to date with the dataframe
                adf.update(expectations=pipeline_element.output_specs)
            except RuntimeError as e:
                txt = f"Failed to update metadata in pipeline element: {pipeline_element}"
                if parent_pipeline:
                    txt += f" in pipeline '{parent_pipeline.name}' ({type(parent_pipeline)})"
                txt += f" -- {e}"

                raise RuntimeError(txt) from e

            if parent_pipeline:
                parent_pipeline.on_transform_end(pipeline_element, adf=adf)

            return adf

        return check

    return decorator


def artifacts(requirements: dict[str, any]):
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
            instance = args[0]
            try:
                required_artifact_specs.validate(
                    base_dir=instance.parent_pipeline.base_dir
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"artifacts: {func} is expected to generate an artifact. "
                    f" Pipeline element ran as part of pipeline: '{instance.parent_pipeline.name}'"
                ) from e

            return result

        return check

    return decorator

