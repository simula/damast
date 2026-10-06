"""
Module to encode the class hierarchy of the global fishing watch
"""
from __future__ import annotations

import re

from damast.domains.maritime.ais.labels import normalise_label


class ShipType:
    """
    Encode the ITU-R M.1371 standard for mapping between ship types and codes.

    The AIS 'type of ship and cargo type' field holds a code from 0 to 99. Six of the ten
    decades are one category subdivided by cargo hazard - 80 is a tanker, 81 to 84 a tanker
    carrying hazardous cargo A to D, 89 a tanker with no further information - while the 30s
    and 50s are distinct types code by code.

    Sources spell the same type differently, so :func:`to_code` resolves the standard's name
    and the common source spellings alike, ignoring case and punctuation. This is what lets a
    source whose ship type is a label rather than a code be aligned onto the standard, instead
    of its value being discarded.

    Example:

        .. highlight:: python
        .. code-block:: python

            ShipType.by_code(81)             # 'tanker_hazardous_a'
            ShipType.to_code("SAR")          # 51
            ShipType.category(84)            # 80 - the tanker decade
            ShipType.codes("tanker")         # [80, ..., 89], e.g. to filter on
    """

    #: Types that stand alone, i.e. are not subdivided by cargo hazard
    DISTINCT_TYPES: dict[int, str] = {
        0: "not_available",
        30: "fishing",
        31: "towing",
        32: "towing_large",
        33: "dredging_or_underwater_ops",
        34: "diving_ops",
        35: "military_ops",
        36: "sailing",
        37: "pleasure_craft",
        50: "pilot_vessel",
        51: "search_and_rescue",
        52: "tug",
        53: "port_tender",
        54: "anti_pollution_equipment",
        55: "law_enforcement",
        56: "spare_local_vessel_1",
        57: "spare_local_vessel_2",
        58: "medical_transport",
        59: "non_combatant",
    }

    #: Categories occupying a whole decade, subdivided by cargo hazard - base code to name
    SUBDIVIDED_TYPES: dict[int, str] = {
        20: "wing_in_ground",
        40: "high_speed_craft",
        60: "passenger",
        70: "cargo",
        80: "tanker",
        90: "other",
    }

    #: Name used where the standard reserves a code for future use
    RESERVED: str = "reserved"

    #: How sources spell a type, normalised, mapped onto the standard's name
    ALIASES: dict[str, str] = {
        "undefined": "not_available",
        "unspecified": "not_available",
        "wig": "wing_in_ground",
        "hsc": "high_speed_craft",
        "towing_long_wide": "towing_large",
        "dredging": "dredging_or_underwater_ops",
        "diving": "diving_ops",
        "military": "military_ops",
        "pleasure": "pleasure_craft",
        "pilot": "pilot_vessel",
        "sar": "search_and_rescue",
        "port_tender": "port_tender",
        "anti_pollution": "anti_pollution_equipment",
        "spare_1": "spare_local_vessel_1",
        "spare_2": "spare_local_vessel_2",
        "medical": "medical_transport",
        "not_party_to_conflict": "non_combatant",
    }

    #: Cargo hazard subdivisions, as the offset from a category's base code
    _HAZARD_SUFFIXES: tuple[str, ...] = ("hazardous_a", "hazardous_b", "hazardous_c", "hazardous_d")

    @classmethod
    def by_code(cls, code: int) -> str:
        """
        Get the ship type for a code.

        :param code: The code as the AIS message carries it
        :return: Name of the type; :attr:`RESERVED`, or ``'<category>_reserved'``, where the
            standard reserves the code
        :raise ValueError: If the code is outside 0 to 99
        """
        if not isinstance(code, int) or isinstance(code, bool) or not 0 <= code <= 99:
            raise ValueError(f"ShipType.by_code: '{code}' is not a code between 0 and 99")

        if code in cls.DISTINCT_TYPES:
            return cls.DISTINCT_TYPES[code]

        base = code - code % 10
        if base not in cls.SUBDIVIDED_TYPES:
            # 1-19, and the gaps in the 30s - reserved for future use
            return cls.RESERVED

        name = cls.SUBDIVIDED_TYPES[base]
        offset = code - base
        if offset == 0 or offset == 9:
            # the category itself, and 'no additional information', are the plain type
            return name
        if offset <= len(cls._HAZARD_SUFFIXES):
            return f"{name}_{cls._HAZARD_SUFFIXES[offset - 1]}"

        return f"{name}_{cls.RESERVED}"

    @classmethod
    def to_code(cls, name: str) -> int:
        """
        Get the code for a ship type, by the standard's name or a source's spelling.

        :param name: Name of the type, e.g. ``"tanker"``, ``"SAR"`` or ``"Towing long/wide"``
        :return: The type's code - the base code for a category that spans a decade
        :raise KeyError: If the name is not a known type, which includes
            :attr:`RESERVED`, as the standard reserves a range rather than one code
        """
        normalised = normalise_label(name)
        normalised = cls.ALIASES.get(normalised, normalised)

        for code, type_name in cls.DISTINCT_TYPES.items():
            if type_name == normalised:
                return code

        for code, type_name in cls.SUBDIVIDED_TYPES.items():
            if type_name == normalised:
                return code
            for offset, suffix in enumerate(cls._HAZARD_SUFFIXES, start=1):
                if f"{type_name}_{suffix}" == normalised:
                    return code + offset

        raise KeyError(f"ShipType.to_code: '{name}' is not a known ship type")

    @classmethod
    def category(cls, code: int) -> int | None:
        """
        Get the code that identifies a code's category, so that the hazard subdivisions of a
        type collapse onto the type itself.

        :param code: The code as the AIS message carries it
        :return: The category's code - the base of the decade for a subdivided category, the
            code itself for a distinct type, and None where the standard reserves the code
        :raise ValueError: If the code is outside 0 to 99
        """
        if cls.by_code(code) == cls.RESERVED:
            return None

        if code in cls.DISTINCT_TYPES:
            return code

        return code - code % 10

    @classmethod
    def codes(cls, name: str) -> list[int]:
        """
        Get every code belonging to a ship type, e.g. to filter a dataframe on it.

        :param name: Name of the type, by the standard or a source's spelling
        :return: The codes, ascending - a whole decade for a subdivided category, otherwise
            the single code
        :raise KeyError: If the name is not a known type
        """
        code = cls.to_code(name)
        if code in cls.DISTINCT_TYPES:
            return [code]

        base = code - code % 10
        return list(range(base, base + 10))

    @classmethod
    def get_mapping(cls) -> dict[str, int]:
        """
        Compute the mapping from every recognised spelling to its code, e.g. to translate a
        column of labels into codes in one step.

        :return: Dictionary representing the mapping, excluding :attr:`RESERVED`, which names
            a range rather than one code
        """
        mapping = {name: code for code, name in cls.DISTINCT_TYPES.items()}
        for code, name in cls.SUBDIVIDED_TYPES.items():
            mapping[name] = code
            for offset, suffix in enumerate(cls._HAZARD_SUFFIXES, start=1):
                mapping[f"{name}_{suffix}"] = code + offset

        for alias, name in cls.ALIASES.items():
            mapping[alias] = mapping[name]

        return mapping


class VesselType:
    """
    The base class for all vessel types defined by the global fishing watch.
    """

    _all_types: list[VesselType] | None = None

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
    def get_types(cls) -> list[VesselType]:
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
    def get_types_as_str(cls) -> list[VesselType]:
        return [x.typename() for x in cls.get_types()]

    @staticmethod
    def _subclasses(cls) -> list[VesselType]:
        klasses = []
        for subclass in cls.__subclasses__():
            klasses.append(subclass)
            klasses.extend(cls._subclasses(subclass))
        return klasses

    @classmethod
    def get_values(cls) -> list[int]:
        """
        Get the int representations for this class

        :return: List of values
        """
        cls._initialize_types()
        values: list[int] = []

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
              klass: str | VesselType = None) -> int:
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
    def get_mapping(cls) -> dict[str, int]:
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
