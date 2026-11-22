from damast.domains.maritime.ais.catalog import AISDataLake

lake = AISDataLake("/opt/workspace/github/damast-icaart/test-data/datalake")

lake.refresh()

#qv = lake.by_mmsi_quickview()
#print(len(qv))


data = lake.get_track(992572500)

breakpoint()
