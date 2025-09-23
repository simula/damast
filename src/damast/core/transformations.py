from __future__ import annotations

from abc import abstractmethod
import copy
import inspect
import importlib
import numpy as np
import polars
import re

from damast.core.dataframe import AnnotatedDataFrame
from .formatting import DEFAULT_INDENT
from .constants import (
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    DECORATED_ARTIFACT_SPECS
)

class Transformer:
    uuid: str

    def set_uuid(self, uuid: str):
        self.uuid = uuid

    def fit(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame | None = None):
        pass

    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame | None = None):
        return df

    def fit_transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame | None = None):
        if not other:
            self.fit(df=df)
            return self.transform(df=df)
        else:
            self.fit(df, other)
            return self.transform(df,other)

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
    def name_mappings(self) -> Dict[str, str]:
        """
        Get current name mappings for this instance
        """
        if not hasattr(self, "_name_mappings"):
            self._name_mappings = {}
        return self._name_mappings

    @property
    def parameters(self) -> Dict[str, str]:
        """
        Get the current list of parameter for initialization
        """
        if not hasattr(self, "_parameters"):
            self._parameters = None

        self.prepare_parameters()

        return self._parameters

    def prepare_parameters(self):
        parameters = {}
        for p in inspect.signature(self.__init__).parameters:
            parameter_name = p
            if not parameter_name in ['args', 'kwargs']:
                # only process named parameters
                try:
                    parameters[parameter_name] = getattr(self, parameter_name)
                except AttributeError as e:
                    raise ValueError(f"PipelineElement: please ensure that {self.__class__.__name__}.__init__ keyword arguments"
                            f" are saved in an attribute of the same name: missing '{parameter_name}'")

        self._parameters = parameters


    def get_name(self, name: str) -> Any:
        """
        Add the fully resolved input/output name for this key.

        :param name: Name as used in the input spec, or pattern "{{x}}_suffix" in order to create a dynamic
                     output based an existing and renameable input
        :return: Name for this input after resolving name mappings and references
        """
        if not isinstance(name, str):
            raise TypeError(f"{self.__class__.__name__}.get_name: provided transformer label is not a string: {name}")

        if name in self.name_mappings:
            # allow multiple levels of name resolution, e.g.,
            # x -> y, y -> z --> x -> z
            mapped_name = self.name_mappings[name]
            if mapped_name == name:
                return name

            return self.get_name(mapped_name)

        # Allow to use patterns, so that an existing input
        # reference can be reused for dynamic labelling
        while re.search("{{\\w+}}", name):
            for match in re.finditer("{{\\w+}}", name):
                resolved_name = match.group()[2:-2]
                if resolved_name in self.name_mappings:
                    resolved_name = self.name_mappings[resolved_name]

                name = name.replace(match.group(), resolved_name)
        return name

    @abstractmethod
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Default transform implementation
        """

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
                   name_mappings: Optional[Dict[str, Any]] = None,
                   parameters: Optional[Dict[str, Any]] = {}) -> PipelineElement:
        """
        Create a new PipelineElement Subclass instance dynamically

        :param module_name: Name of the module for the PipelineElement class
        :param class_name: Name of the PipelineElement subclass
        :param name_mappings: Dictionary of name mappings that should apply
        :return: Instance for the PipelineElement instance

        :raise ValueError: If module or class with given name is not specified
        :raise ImportError: If class could not be loaded
        """
        if module_name is None:
            raise ValueError(f"{cls.__name__}.create_new: missing 'module_name'")

        if class_name is None:
            raise ValueError(f"{cls.__name__}.create_new: missing 'class_name'")

        p_module = importlib.import_module(module_name)
        if hasattr(p_module, class_name):
            klass = getattr(p_module, class_name)
        else:
            raise ImportError(f"{cls.__name__}.create_new: could not load '{class_name}' from '{p_module}'")
        if parameters:
            instance = klass(**parameters)
        else:
            instance = klass()

        if name_mappings is None:
            name_mappings = {}
        instance._name_mappings = name_mappings
        return instance

    def __iter__(self):
        yield "module_name", f"{self.__class__.__module__}"
        yield "class_name", f"{self.__class__.__qualname__}"
        yield "parameters", self.parameters
        yield "name_mappings", self.name_mappings

    def __eq__(self, other):
        return dict(self) == dict(other)

    @classmethod
    def get_types(cls) -> List[PipelineElement]:
        """
        Get all available PipelineElement implementations

        :return: List of PipelineElement classes
        """
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(subclass._subclasses())
        return klasses

    @classmethod
    def _subclasses(cls) -> List[PipelineElement]:
        """
        Generate the list of subclasses for the calling class
        """
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(subclass._subclasses())
        return klasses

    @classmethod
    def generate_subclass_documentation(cls) -> str:
        """
        Generate the documentation for all subclasses of ::class::`PipelineElement`
        """
        implementations = sorted(cls.get_types(), key=str)
        txt = ""
        for k in implementations:
            txt += "=" * 80
            txt += f"\n{k.__name__} -- from {k.__module__}\n"
            txt += k.__doc__
            txt += '\n'
        return txt

    def to_str(self, indent_level: int = 0) -> str:
        hspace = DEFAULT_INDENT * indent_level
        data = hspace + self.__class__.__name__ + "\n"
        if hasattr(self.transform, DECORATED_DESCRIPTION):
             description = getattr(self.transform, DECORATED_DESCRIPTION)
             data += (
                     hspace + DEFAULT_INDENT * 2 + "description: " + description + "\n"
                )
    
        data += hspace + DEFAULT_INDENT * 2 + "input:\n"
        if hasattr(self.transform, DECORATED_INPUT_SPECS):
            data += DataSpecification.to_str(
                self.input_specs, indent_level=indent_level + 4
            )

        data += hspace + DEFAULT_INDENT * 2 + "output:\n"
        if hasattr(self.transform, DECORATED_OUTPUT_SPECS):
            data += DataSpecification.to_str(
                self.output_specs, indent_level=indent_level + 4
            )
        return data

class CycleTransformer(PipelineElement):
    def __init__(self, features: list[str], n: int):
        self.features = features
        self.n = n

    def transform(self, df: AnnotatedDataFrame):
        if type(df) != AnnotatedDataFrame:
            raise ValueError(f"Transformer requires 'AnnotatedDataFrame',"
                    f" but got '{type(df)}")
        clone = df.copy()

        for feature in self.features:
            clone._dataframe = clone._dataframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_x")
                )
            clone._dataframe = clone._dataframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_y")
                )
        return clone
