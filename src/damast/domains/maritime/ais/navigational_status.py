from enum import IntEnum

from damast.domains.maritime.ais.labels import normalise_label

#: How sources spell a status, normalised, where it does not reduce to a member's name. Codes 9
#: and 10 reserve an amendment for high speed craft and wing-in-ground craft respectively, which
#: the member names do not say. Kept outside the enum, whose every plain attribute is a member.
NAVIGATIONAL_STATUS_ALIASES: dict[str, str] = {
    "unknown_value": "Undefined",
    "reserved_for_future_amendment_hsc": "_RESERVED_FOR_FUTURE__HAZARDOUS_GOODS_0",
    "reserved_for_future_amendment_wig": "_RESERVED_FOR_FUTURE__HAZARDOUS_GOODS_1",
    "ais_sart_active": "AIS_SART__MOB_AIS__EPIRB_AIS",
}


class AISNavigationalStatus(IntEnum):
    """
    The AIS Navigational Status:

    :see https://help.marinetraffic.com/hc/en-us/articles/203990998-What-is-the-significance-of-the-AIS-Navigational-Status-Values-

    """
    UnderWayUsingEngine = 0
    AtAnchor = 1
    NotUnderCommand = 2
    RestrictedManeuverability = 3
    ConstrainedByHerDraught = 4
    Moored = 5
    Aground = 6
    EngagedInFishing = 7
    UnderWaySailing = 8
    _RESERVED_FOR_FUTURE__HAZARDOUS_GOODS_0 = 9
    _RESERVED_FOR_FUTURE__HAZARDOUS_GOODS_1 = 10
    Power_DrivenVesselTowingAstern = 11
    Power_DrivenVesselPushingAheadOrTowingAlongside = 12
    _RESERVED_FOR_FUTURE_USE = 13
    AIS_SART__MOB_AIS__EPIRB_AIS = 14
    Undefined = 15

    @classmethod
    def get_values(cls) -> list[int]:
        return [e.value for e in AISNavigationalStatus]

    @classmethod
    def get_mapping(cls) -> dict[str, int]:
        """
        Compute the mapping from every recognised spelling of a status to its code.

        A source may report the status as text rather than the code, so the member names and
        the spellings in :data:`NAVIGATIONAL_STATUS_ALIASES` both resolve, ignoring case and
        punctuation.

        :return: Dictionary representing the mapping
        """
        mapping = {normalise_label(status.name): status.value for status in cls}
        for alias, name in NAVIGATIONAL_STATUS_ALIASES.items():
            mapping[alias] = cls[name].value

        return mapping

    @classmethod
    def to_code(cls, name: str) -> int:
        """
        Get the code for a navigational status, by its name or a source's spelling.

        :param name: Name of the status, e.g. ``"Under way using engine"`` or ``"AIS-SART (active)"``
        :return: The status' code
        :raise KeyError: If the name is not a known status
        """
        try:
            return cls.get_mapping()[normalise_label(name)]
        except KeyError:
            raise KeyError(f"{cls.__name__}.to_code: '{name}' is not a known navigational status")
