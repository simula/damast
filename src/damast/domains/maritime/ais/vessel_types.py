"""
Module to encode the class hierarchy of the global fishing watch
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Union


class VesselType:
    """
    The base class for all vessel types defined by the global fishing watch.
    """

    _all_types: Optional[List[VesselType]] = None

    @classmethod
    def typename(cls) -> str:
        """
        Get the representation name.

        :return: The typename in lower case and snake case
        """
        snake_case_name = cls.__name__
        snake_case_name = re.sub('([A-Z]+)', r'_\1', snake_case_name).lower()
        snake_case_name = re.sub('^_', '', snake_case_name)
        return snake_case_name

    @classmethod
    def get_types(cls) -> List[VesselType]:
        """
        Get all available vessel types.

        :return: List of vessel types
        """
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(cls._subclasses(subclass))
        return klasses

    @classmethod
    def get_types_as_str(cls) -> List[VesselType]:
        return [x.typename() for x in cls.get_types()]

    @staticmethod
    def _subclasses(cls) -> List[VesselType]:
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(cls._subclasses(subclass))
        return klasses

    @classmethod
    def get_values(cls) -> List[int]:
        """
        Get the int representations for this class

        :return: List of values
        """
        cls._initialize_types()
        values: List[int] = []

        for klass in cls._all_types:
            values.append(VesselType.to_id(klass=klass))

        return values

    @classmethod
    def _initialize_types(cls):
        if cls._all_types is None:
            cls._all_types = cls.get_types()

    @classmethod
    def by_id(cls, *,
              identifier: int) -> VesselType:
        cls._initialize_types()

        return cls._all_types[identifier]

    @classmethod
    def to_id(cls, *,
              klass: Union[str, VesselType] = None) -> int:
        """
        Get the id for a klass name or class type of VesselType.

        :param klass:
        :return: id for a particular vessel class
        """
        VesselType._initialize_types()
        if klass is None:
            klass = cls

        if type(klass) is str:
            klass = cls.by_name(name=klass)

        if issubclass(klass, VesselType):
            return cls._all_types.index(klass)

        raise ValueError(f"VesselType.by_id: failed to identify '{klass}'")

    @classmethod
    def by_name(cls,
                name: str) -> VesselType:
        """
        Get the VesselType by given name

        :param name: Name (representation) of the type
        :return: VesselType Class Object
        """
        cls._initialize_types()

        for k in cls._all_types:
            if k.typename() == name:
                return k

        raise KeyError(f"VesselType.by_name: failed to identify '{name}'")

    def __class_getitem__(cls, name: str) -> int:
        """
        Allow an Enum-like interface to the class index values

        :param name: Name of the class
        :return: int representation of the class
        """
        klass = cls.by_name(name=name)
        return cls.to_id(klass=klass)

    @classmethod
    def get_mapping(cls) -> Dict[str, int]:
        """
        Compute the mapping from vessel typename to integer

        :return: Dictionary representing the mapping
        """
        mapping = {}
        for t in cls.get_types():
            mapping[t.typename()] = cls.to_id(klass=t)
        return mapping


class Unspecified(VesselType):
    pass


class Cargo(VesselType):
    pass


class Passenger(VesselType):
    pass


class Pleasure(VesselType):
    pass


class Specialcraft(VesselType):
    pass


class Tanker(VesselType):
    pass


class Tug(VesselType):
    pass


class Fishing(VesselType):
    pass

# region Global Fishing Watch Types


class SquidJigger(Fishing):
    pass


class DriftingLonglines(Fishing):
    pass


class PoleAndLine(Fishing):
    pass


class Trollers(Fishing):
    pass


class FixedGear(Fishing):
    pass


class Trawlers(Fishing):
    pass


class DredgeFishing(Fishing):
    pass


class Seiners(Fishing):
    pass


class PurseSeines(Seiners):
    pass


class OtherSeines(Seiners):
    pass


class TunaPurseSeines(PurseSeines):
    pass


class OtherPurseSeines(PurseSeines):
    pass


class PotsAndTraps(FixedGear):
    pass


class SetLonglines(FixedGear):
    pass


class SetGillnets(FixedGear):
    pass

# endregion
