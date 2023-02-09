# Maritime Domain: AIS Anomaly Detection

## Context-aware Autoencoders for Anomaly Detection in Maritime Surveillance

Autoencoders are a popular auto-supervised ML method for anomaly detection in time series.
However, the relevance of detected collective and contextual anomalies is often hindered by the difficulty of reconstructing samples. 
Additionally, some anomalies are context-dependent and cannot be detected by standard, context-unaware autoencoders.
We propose ML methods based on context-aware autoencoders to improve the overall anomaly detection quality and apply them to a maritime surveillance case study to detect anomalies in fishing status.
In maritime vessel traffic surveillance, vessels self-report their positions through AIS messages captured by satellites, and anomalous behaviour is highly dependent on the vessel type and activity.
Our methods integrate context-specific thresholds and identify the most relevant contexts to reduce the computational cost of a large number of contexts. 
For the case study evaluation, we consider three variants of context-aware autoencoders as well as the standard autoencoder; the results show that considering the context improves the quality of detected anomalies.

## Installation

To Be Defined.
A docker container to install and run the project is in preparation.

## Parametrization
Set 'params.yaml' file

### RAW Data

- available libraries for usage:
  - https://pypi.org/project/pyais/

Example to decode '.nmea' files.
To make this example work it requires the latest version of pyais library:
```
from pyais.stream import FileReaderStream
from pyais.decode import NMEASentence, decode_nmea_line
import tempfile
from pathlib import Path
from typing import Union
import logging

logging.basicConfig()
LOG: logging.Logger = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def create_example_nmea_file() -> Path:
    messages = [b"\c:1556260129,s:Sat_A,i:<S>S</S><O>XNS</O><T>A:1556264827 F:+3044000</T>*3D\!AIVDM,1,1,,B,15B<J<0P1qF`kTKs86p=PgwR1PR=,0*77",
                b"\c:1556266429,s:Sat_A,i:<S>S</S><O>XNS</O><T>A:1556270597 F:+3744000</T>*3A\!AIVDM,1,1,,A,C1MjQv03wk?8mP=18D3Q3whHPBL?0`2C0HNL?1ccKV30?081110W,0*4C",
                b"\c:1559524187,s:Stat_B,i:<S>S</S><O>XNS</O><T>A:1559529703 F:-2484000</T>*36\!AIVDM,1,1,,D,KmB<J<0@3tCkC0Bl,0*4F"]

    tmpdir = tempfile.gettempdir()
    tmpfile = Path(tmpdir) / "example.nmea"
    with open(tmpfile, "wb") as f:
        for msg in messages:
            f.write(msg)
            f.write(b"\n")
    return tmpfile


def read_nmea_file(filename: Union[str, Path]) -> None:
    LOG.info(f"Reading nmea data from: {filename}")
    with open(filename, "rb") as f:
        while True:
            line = f.readline()
            if line == b'':
                break

            try:
                msg: NMEASentence = decode_nmea_line(line)
                print(msg.decode())
                if msg.tag_block is not None:
                    msg.tag_block.init()
                    print(msg.tag_block.asdict())
            except Exception as e:
                LOG.warning(f"Failed to decode: {line} - {e}")



if __name__ == "__main__":
    filename = create_example_nmea_file()
    read_nmea_file(filename=filename)

```


#### Example message:
```
\c:1559524187,s:AnySat_1,i:<S>S</S><O>XNS</O><T>A:1559529703 F:-2484000</T>*36\!AIVDM,1,1,,D,KmB<J<0@3tCkC0Bl,0*4F
```

| **Field / Column Name**                                         | **Description**                                                                                                                                                                                                                |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| \                                                               | Comment block is used to transport more information                                                                                                                                                                            |
| c:<int>                                                         | UTC timestamp                                                                                                                                                                                                                  |
| s:<string>                                                      | source                                                                                                                                                                                                                         |
| d:<string>                                                      | destination                                                                                                                                                                                                                    |
|||
| i: <information>                                                | as documented in the following                                                                                                                                                                                                 | 
| \<S>S\</S>                                                      | Enumerated list to identify system generating the information<br/>Content:<br/>S - Satellite AIS<br/>A - Terrestial AIS<br/>...                                                                                                     |
| \<O>XNS\</O>                                                    | Providing information on the data originator Format, RCC.SUB.SUB, where<br/> RCC Region or Country Code,<br/> XNS -> North Sea                                                                                                 |
| \<T>A:1559529703 F:-2484000\</T>                                | Values separated by a space, Ground station aquisition timestamp (A:<timestamp>),<br/> data centre ingestion timestamp (I:<timestamp>)<br/>Data Centre delivery timestamp (D:<timestamp>),<br/> Satellite ID (L:<string>)<br/> |
|                                                                 | Ground station ID (G:), frequency shift of arrival with respect to the centre of the AIS channel in MHz (F:)                                                                                                                   |                                                                         |
|                                                                 | precise time of arrival within detection second given after (C:<microsecs>)                                                                                                                                                    |                                                                         | 
| \<E>\<E/>                                                       | Specify basic enrichment (only if, status in I: must be "valid")                                                                                                                                                               |                                                                                                                                                              |
| \<I>\<string>\<I/>                                              | Status <V or T or N>; IMO<7chars>; MMSI<9chars>;...                                                                                                                                                                            |                                                                                                                                                                                                                      |
| \<N> \</N>                                                      | Accuracy of the accuracy/reliability of a position                                                                                                                                                                             |
|||
| x:\<int>                                                        ||
| g:\<sentence-number>-\<total-number-of-sentences>-\<identifier> | Example: g:1-2-1234                                                                                                                                                                                                            |
|||
| **Reserved Characters**                                         ||
| *                                                               | checksum field delimiter, e.g., *36                                                                                                                                                                                            |
| !                                                               | Start of encapsulation sentence delimiter                                                                                                                                                                                      |




For T the following regex is specified:
```
 ^(?:(?:(?:[AID]:\d+)|(?:[FT]:[+-]?\d+)|(?:[LG]:"?[\w-]+"?)) +){1,7}
```

Example:
```
<T>A:123456789 I:123456789 D:123456789 L:"AIS- SAT1" G:"Svalbard-5" F:+975000 T:+123000</T>
```





### AIS Data

#### Test data

Sample AIS data used for tests comes from [Geonorge's map catalog](https://www.geonorge.no). The data is distributed by the Norwegian Coastal Agency under [Norwegian Licence for Open Government Data (NLOD) 2.0](https://data.norge.no/nlod/en/2.0).
Similar AIS datasets can be downloaded [here](https://kartkatalog.geonorge.no/metadata/automatisk-identifikasjonssystem-ais-shipsposisjoner-nedlasting-12nm-fra-grunnlinja/7997fd76-83f9-4e94-bfe7-f4677a6cd787).

#### Main data

- columns: mmsi;lon;lat;date_time_utc;sog;cog;true_heading;nav_status;rot;message_nr;source
- MMSI, LON, LAT, SOG, COG, Heading, Status, timestamp, rot, MessageType, source

See references: 
- [marinetraffic.com](https://help.marinetraffic.com/hc/en-us/articles/205426887-What-kind-of-information-is-AIS-transmitted-)
- [fleetmon.com](https://help.fleetmon.com/en/articles/4476744-ais-navigational-status)
- [Mapping of MID to Country ISO Codes](https://raw.githubusercontent.com/michaeljfazio/MIDs/master/mids.json)

| Field/Column Name                              | Description                                                                                                                                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Maritime Mobile Service Identify number (MMSI) | unique identification number for each vessel station  <br/><a href="https://www.fcc.gov/wireless/bureau-divisions/mobility-division/maritime-mobile/ship-radio-stations/maritime-mobile"> MMSI Components</a> |
| status | [AIS Navigational Status](https://help.marinetraffic.com/hc/en-us/articles/203990998-What-is-the-significance-of-the-AIS-Navigational-Status-Values-)                                                         |
 | Rate of Turn | right or left (0 to 720 degrees per minute)                                                                                                                                                                   |
| sog | Speed over Ground: 0 to 102 knots (0.1-knot resolution)                                                                                                                                                       |
| lat/lon | Position Coordinates: up to 0.0001 minutes accuracy                                                                                                                                                           |
| cog | Course over Ground: up to 0.1 deg relative to true north                                                                                                                                                      |
| heading | 0 to 359 degrees                                                                                                                                                                                              |
| bearing | bearing at own position: 0 to 359 degrees                                                                                                                                                                     |
| timestamp | UTC seconds                                                                                                                                                                                                   |


### Data processing
Parameters set in under 'data_processing':

| Parameter name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| useless_columns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | will be dropped during compression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| columns_brut                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | use the columns listed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| columns_default_value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | replaces NaN with this default value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| columns_type                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | set default column type                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| unused_columns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | will be dropped during compression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|||
| **do_filter_area**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| do_filter_area>lat_min                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | min latitude                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| do_filter_area>lat_max                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | min latitude                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| do_filter_area>lon_min                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | min longitude                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| do_filter_area>lon_max                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | max longitude                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|||
| **MMSI**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| MMSI > min                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | minimum MMSI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| MMSI > max                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | maximum MMSI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| min_number_msg_by_mmsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | minimum number of messages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|||
| MMSI_to_filter_file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | vessel type data, e.g.,   data/mmsi_with_vesseltype.csv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                    |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | columns: MMSI, vesseltype                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
 | `data/fishing-vessels-2.csv `                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | hardcoded fishing vessel data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| | columns: mmsi,flag_ais,flag_registry,flag_gfw, <br/> vessel_class_inferred,vessel_class_inferred_score,<br/>vessel_class_registry,vessel_class_gfw,<br/>self_reported_fishing_vessel,<br/>length_m_inferred,length_m_registry,length_m_gfw,<br/>engine_power_kw_inferred,engine_power_kw_registry,engine_power_kw_gfw,<br/>tonnage_gt_inferred,tonnage_gt_registry,tonnage_gt_gfw,<br/>registries_listed,<br/>fishing_hours_2012,fishing_hours_2013,fishing_hours_2014,<br/>fishing_hours_2015,fishing_hours_2016,fishing_hours_2017,fishing_hours_2018,<br/>fishing_hours_2019,fishing_hours_2020 |
|||
| anchorage_file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | list of anchorages and their lat/lon locations (in m), e.g.,  data/anchorages.csv                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | columns: s2id,latitude,longitude,label,sublabel,iso3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |


Data processing will merge vessel type data with fishing type data.
Adds columns:
- HISTORIC_SIZE
- HISTORIC_SIZE_REVERSED


Input path:
- data/<month>/*.zip

Output pathdata_types:
- data/processed/*.h5


#### Vesseltype data.
References:
- https://globalfishingwatch.org/datasets-and-code-vessel-identity/

Requires:
- vesseltype (as category code)

#### Fishing vessel type
[Global Fishing Watch (GFW)](https://globalfishingwatch.org):
- available columns: mmsi,flag_ais,flag_registry,flag_gfw,vessel_class_inferred,vessel_class_inferred_score,vessel_class_registry,vessel_class_gfw,self_reported_fishing_vessel,length_m_inferred,length_m_registry,length_m_gfw,engine_power_kw_inferred,engine_power_kw_registry,engine_power_kw_gfw,tonnage_gt_inferred,tonnage_gt_registry,tonnage_gt_gfw,registries_listed,fishing_hours_2012,fishing_hours_2013,fishing_hours_2014,fishing_hours_2015,fishing_hours_2016,fishing_hours_2017,fishing_hours_2018,fishing_hours_2019,fishing_hours_2020
- used input columns: mssi, vessel_class_gfw
- renamed columns:
   - mmsi: MMSI
   - vessel_class_gfw: fishing_type
  


## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

1. Get the datasets and put them into the data folder so that is looks as follows: 
```
    data/<month>/<day>.zip
```

2. Run the data processing to generate a data/processed/<month>.h5, e.g. for month 1:
```
    python -m ais_anomaly_detection.data_handling.data_processing -m 1 -M 1 -A
```
   | Option | Description                    |
   |--------------------------------|-------------|
   | -m 1   | to generate from month 1       |
   | -M 1   | to generate till month 1       |
   | -A     | compute the anchorage distance |

3. Run the "datasets creation":
```
    python -m ais_anomaly_detection.data_handling.datasets_creation -m 1 -M 1
```

4. Run the "datasets example creation", which requires the data of month 1 (hardcoded):
```
    python -m ais_anomaly_detection.data_handling.datasets_examples_creation 
```

5. Run the training, e.g. for GCAE
   ```
   python -m src.training -t combination -m 1 -E CNN -D CNN -e 200 -b 256 -l 75 -w 50 -s 100_000 -R 50_000 --prefix aaai_test_ -WgG --do_global_evaluation --do_use_cache
   ```

    | Option            | Description                                                                                                        |
    |-------------------|--------------------------------------------------------------------------------------------------------------------|
    | -t combination    | task: combination                                                                                                  |
    | -m 1              | month: individual month (1-12) or "year" for all, which search for data/datasets/train_trajectories_<m-value>.npy  |
    | -E CNN            | encoder layer: CNN                                                                                                 |
    | -D CNN            | decoder layer: CNN                                                                                                 |
    | -e 200            | number of epochs                                                                                                   |
    | -b 256            | batch size                                                                                                         |
    | -l 75             | dimension of latent layer                                                                                          |
    | -w 50             | windows size                                                                                                       |
    | -s 100_000        | max samples per class                                                                                              |
    | -R 50_000         | do reduce overrepresented labels                                                                                   |
    | --prefix          | number of epochs                                                                                                   |
    | -WgW              | do sample weight (-W), do training GCAE                                                                            |
   
   Once the training is running it can be monitored using tensorboard:
   ```
       tensorboard --logdir=outputs/training/experiments/
   ```
  
## Contributing
This project is open to contributions. In order to collaborate, fork the project and create a merge request with your desired changes.
Adhere to any existing code style and accompany your changes a corresponding tests. Check also if the documentation needs to be updated
as result of your changes.

## Authors and Acknowledgment
Pierre Bernabé<sup>1</sup><sup>2</sup>, Antoine Chevrot<sup>2</sup>, Helge Spieker<sup>1</sup>, Arnaud Gotlieb<sup>1</sup> and Bruno Legeard<sup>2</sup>

<sup>1</sup> Simula Research Laboratory, Oslo, Norway

<sup>2</sup> Institut FEMTO-ST, Université de Bourgogne Franche-Comté, Besançon, France

## License
This project has not been published. A license has to be defined for publishing.

## Project status
This work has been performed as part of the [T-SAR project](https://www.simula.no/research/projects/t-sar)
The Code is currently under review and will likely received a significant restructuring in the month, i.e. in 2023-Q1

