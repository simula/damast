from pydantic import (
    BaseModel,
    Field
)
from enum import Enum
from pathlib import Path

import datetime as dt
import requests

from argparse import ArgumentParser
import logging

logger = logging.getLogger(__name__)

COPERNICUS_CATALOG = "https://catalogue.dataspace.copernicus.eu/resto/api"

class Satellites(str, Enum):
    SENTINEL2 = "Sentinel2"

class Query(BaseModel):
    cloudCover: int = Field(None, ge=0, le=10)

class MinMax(BaseModel):
    lower_bound: int | float
    upper_bound: int | float

    def __repr__(self):
        return f"[{self.lower_bound},{self.upper_bound}]"


#class CopernicusResponse:
#    type: str
#    features: list[any]


def search(satellite: str, 
        start_date: str | None = None,
        end_date: str | None = None,
        cloud_cover: MinMax | None = None,
        max_records: int = 100,
        lat: float | None = None,
        lon: float | None = None,
        radius: float | None = None
        ):
    search_path = f"{COPERNICUS_CATALOG}/collections/{satellite}/search.json"
        
    params = {}

    now = dt.datetime.utcnow()
    if start_date:
        params["startDate"] = dt.datetime.fromisoformat(start_date)
    #else:
    #    params["startDate"] =  now - dt.timedelta(days=7)

    if end_date:
        params["completionDate"] = dt.datetime.fromisoformat(end_date)
    #else:
    #    params["completionDate"] = now

    for d in ["startDate", "completionDate"]:
        if d in params:
            params[d] = params[d].isoformat()

    if cloud_cover:
        params["cloudCover"] = repr(cloud_cover)

    if lat:
        params["lat"] = repr(lat)

    if lon:
        params["lon"] = repr(lon)

    if radius:
        params["radius"] = radius


    params["sortParam"] = "startDate"
    params["maxRecords"] = max_records


    logger.info(f"Searching {search_path} with {params=}")
    response = requests.get(search_path, params=params)
    return response.json()


if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    logging.basicConfig()

    parser = ArgumentParser()
    parser.add_argument("--satellite",
            type=str,
            default="Sentinel2",
    )

    range_parameters = ["cloud_cover"]

    parser.add_argument("--start-date",
        type=str,
        default=None
    )
    parser.add_argument("--end-date",
        type=str,
        default=None
    )

    parser.add_argument("--lat",
        type=float,
        default=None
    )
    parser.add_argument("--lon",
        type=float,
        default=None
    )

    parser.add_argument("--radius",
            type=float,
            default=50*1000,
            help="Radius in m, around lat/lon coordinate"
    )

    for p in range_parameters:
        parser.add_argument(f"--{p}-min",
            type=int,
            default=None,
            help=f"Lower bound for {p}",
        )
        parser.add_argument(f"--{p}-max",
            type=int,
            default=None,
            help=f"Upper bound for {p}"
        )

    args, options = parser.parse_known_args()
    
    kwargs = {}

    if args.start_date:
        kwargs["start_date"] = args.start_date

    if args.end_date:
        kwargs["end_date"] = args.end_date

    for p in range_parameters:
        lb = None
        ub = None
        if hasattr(args, f"{p}_min"):
            lb = getattr(args, f"{p}_min")
        if hasattr(args, f"{p}_max"):
            ub = getattr(args, f"{p}_max")

        if lb and ub:
            kwargs[p] = MinMax(lower_bound=lb, upper_bound=ub)
        elif lb or ub:
            raise ValueError(f"Please provide {p}-min and {p}-max")

    if args.lat and args.lon and args.radius:
        kwargs["lat"] = args.lat
        kwargs["lon"] = args.lon
        kwargs["radius"] = args.radius

    results = search(args.satellite,
            **kwargs
    )

    properties = results["properties"]
    features = results["features"]

    minStartDate = None
    maxStartDate = None
    for f in features:
        startDate = f['properties']['startDate']
        completionDate = f['properties']['completionDate']

        if minStartDate is None:
            minStartDate = startDate
        elif minStartDate > startDate:
            minStartDate = startDate

        if maxStartDate is None:
            maxStartDate = startDate
        elif maxStartDate < startDate:
            maxStartDate = startDate

    count = len(features)
    if count:
        print(f"Results found: {count} - {minStartDate=}  {maxStartDate=}")
    else:
        print(f"No matching records")

    #print(results)
