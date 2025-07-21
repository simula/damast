# Copernicus Catalogue Access

Check https://dataspace.copernicus.eu/analyse/apis/catalogue-apis

https://documentation.dataspace.copernicus.eu/APIs/OpenSearch.html

Using plain search - not automated downloading of images yet
``` 
    python copernicus.py --start-date 2020-10-23T00:00:00 --end-date 2020-10-30T00:00:00 --lat 13 --lon -12 --radius 100 --max-pages 100 --satellite SENTINEL-2 --download-dir search-results --download-thumbnails
```

``` 
    python copernicus.py --start-date 2025-07-21T00:00:00 --end-date 2025-07-21T23:00:00 --satellite SENTINEL-2 --download-thumbnails
``` 

Using catalogue search:
``` 
    python copernicus.py --catalogue --start-date 20201023 --end-date 20201030 --bbox=13,-12.2,12,-12 --max-pages 100 --collection SENTINEL-2 --download-dir case-1
```
