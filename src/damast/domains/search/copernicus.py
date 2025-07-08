from pydantic import (
    BaseModel,
    Field
)
from enum import Enum
from pathlib import Path

import datetime as dt
import os
import re
import requests

from argparse import ArgumentParser
import logging

import pystac
import xarray
from tqdm import tqdm

logger = logging.getLogger(__name__)

COPERNICUS_CATALOG = "https://catalogue.dataspace.copernicus.eu/resto/api"

# https://documentation.dataspace.copernicus.eu/APIs/OpenSearch.html
# https://sentiwiki.copernicus.eu/web/s1-products
class CopernicusCollections(str, Enum):
    SENTINEL1 = "SENTINEL-1"
    SENTINEL2 = "SENTINEL-2"
    SENTINEL3 = "SENTINEL-3"

    SENTINEL5 = "SENTINEL-5P"
    SENTINEL6 = "SENTINEL-6"
    SENTINEL1RTC = "SENTINEL-1-RTC"

    GLOBAL_MOSAICS = "GLOBAL-MOSAICS"
    SMOS = "SMOS"
    ENVISAT = "ENVISAT"
    Landsat5 = "LANDSAT-5"
    Landsat7 = "LANDSAT-7"
    Landsat8 = "LANDSAT-8"

    # Copernicus DEM
    COP_DEM = "COP-DEM"

    TERRAAQUA = "TERRAAQUA"
    S2GLC = "S2GLC"

    def names() -> list[str]:
        return [x.value for x in CopernicusCollections]

class Query(BaseModel):
    cloudCover: int = Field(None, ge=0, le=10)

class MinMax(BaseModel):
    lower_bound: int | float
    upper_bound: int | float

    def __repr__(self):
        return f"[{self.lower_bound},{self.upper_bound}]"

# https://documentation.dataspace.copernicus.eu/APIs/OpenSearch.html
def search(satellite: str | None = None, 
        start_date: str | None = None,
        end_date: str | None = None,
        cloud_cover: MinMax | None = None,
        max_records: int = 100,
        lat: float | None = None,
        lon: float | None = None,
        radius: float | None = None
        ):

    search_path = f"{COPERNICUS_CATALOG}/collections/search.json"
    if satellite:
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
    json = response.json()


def search_catalogue(
        bbox: list[float],
        collections: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
    ):

    search_path = f"https://catalogue.dataspace.copernicus.eu/stac/search" 

    params = {}
    if start_date or end_date:
        params["datetime"] = f"{start_date if start_date else ''}/{end_date if end_date else ''}"

    if bbox:
        params["bbox"] = f"{bbox}"

    if collections:
        params["collections"] = ','.join(collections)

    logger.info(f"Search catalogue: {search_path} {params=}")
    response = requests.get(search_path, params=params)
    return response.json()


if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    logging.basicConfig()

    parser = ArgumentParser()
    parser.add_argument("--catalogue",
            action="store_true",
            default=False,
            help="Search the catalogue of items"
    )
    parser.add_argument("--collection",
            nargs="+",
            default=None,
    )

    parser.add_argument("--satellite",
            type=str,
            default=None,
            help="Use satellite from Sentinel[1,2,3]",
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

    parser.add_argument("--bbox",
        type=str,
        default=None,
        help="Bbox (bottom_lon,bottom_lat, top_lon,top_lat)"
    )

    parser.add_argument("--max-pages",
        type=int,
        default=100,
        help="Max number of page(d) results"
    )

    parser.add_argument("--download-dir",
        type=str,
        default="copernicus-downloads",
        help="Download folder"
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


    if not args.catalogue:
        results = search(args.satellite,
                **kwargs
        )
        if not results:
            raise ValueError("No search results")

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

    else:
        if args.bbox:
            kwargs['bbox'] = args.bbox

        def get_by_relation(relation: str, f: list[dict[str, any]]):
            for x in f:
                if x['rel'] == relation:
                    return x

        collection_names = CopernicusCollections.names()
        if args.collection:
            for x in args.collection:
                if x not in collection_names:
                    raise ValueError(f"Collection {x} does not exist")
            collection_names = args.collection

        collections = {}
        for collection in collection_names:
            for page in range(0, args.max_pages):
                kwargs["page"] = page
                kwargs["collections"] = [collection]

                results = search_catalogue(**kwargs)

                if 'links' in results:
                    links = results['links']
                    next_ref = [x['href'] for x in links if x['rel'] == 'next']

                    if 'features' in results:
                        features = results['features']

                        for f in features:
                            collection = get_by_relation('collection', f['links'])['href']
                            item = get_by_relation('self', f['links'])['href']
                            if collection not in collections:
                                collections[collection] = [item]
                            else:
                                collections[collection].append(item)

                if not next_ref:
                    break

        print(f"Collections:")
        base_dir = Path(args.download_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        for k,v in tqdm(collections.items(), desc=f"Collection:"):
            print(f"  {k}:")
            print(f"    items:")
            print(f"      count: {len(v)}")
            print(f"      example: {v[0]}")

            m = re.search(r'[^/]+$',k)
            collection_folder = base_dir / m.group(0)
            collection_folder.mkdir(exist_ok=True)


            for filename in tqdm(v, desc="Images"):
                item = pystac.Item.from_file(filename)
                # https://sentiwiki.copernicus.eu/web/safe-format
                if item.id.endswith(".SAFE"):
                    if 'QUICKLOOK' in item.assets:
                        download_url = item.assets['QUICKLOOK'].href
                        #--header \"Authorization: Bearer $ACCESS_TOKEN\" 
                        os.system(f"curl -L '{download_url}' --location-trusted --output {collection_folder}/{item.id}.jpeg")
                elif item.id.endswith(".SEN3"):
                    if 'QUICKLOOK' in item.assets:
                        download_url = item.assets['QUICKLOOK'].href
                        #--header \"Authorization: Bearer $ACCESS_TOKEN\" 
                        os.system(f"curl -L '{download_url}' --location-trusted --output {collection_folder}/{item.id}.jpeg")
                elif item.id.endswith(".SEN6"):
                    if 'PRODUCT' in item.assets:
                        download_url = item.assets['PRODUCT'].href
                        logger.warning("Download of SEN6 requires special token")
                        #--header \"Authorization: Bearer $ACCESS_TOKEN\" 
                        #os.system(f"curl -L '{download_url}' --location-trusted --output {collection_folder}/{item.id}.jpeg")
                elif item.id.endswith(".nc"):
                    if 'PRODUCT' in item.assets:
                        download_url = item.assets['PRODUCT'].href
                        #--header \"Authorization: Bearer $ACCESS_TOKEN\" 
                        os.system(f"curl -L '{download_url}' --location-trusted --output {collection_folder}/{item.id}.jpeg")
                else:
                    try:
                        ds = xarray.open_dataset(item)
                    except ValueError as e:
                        breakpoint()
