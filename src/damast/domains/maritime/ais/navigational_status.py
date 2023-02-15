from enum import IntEnum
from typing import List


class AISNavigationalStatus(IntEnum):
    """
    The AIS Navigational Status:

    :see https://help.marinetraffic.com/hc/en-us/articles/203990998-What-is-the-significance-of-the-AIS-Navigational-Status-Values-

    """  # noqa: E501
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
    def get_values(cls) -> List[int]:
        return [e.value for e in AISNavigationalStatus]
