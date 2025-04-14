[![Supported Python Versions](https://img.shields.io/pypi/pyversions/damast)](https://pypi.org/project/damast/)
![test workflow](https://github.com/simula/damast/actions/workflows/test.yml/badge.svg)
![docs workflow](https://github.com/simula/damast/actions/workflows/gh-pages.yml/badge.svg)

# damast: Creation of reproducible data processing pipelines

The main purpose of this library is to faciliate the reusability of data and data processing pipelines.
For this, damast introduces a means to associate metadata with data frames and enables consistency checking.

To ensure semantic consistency, transformation steps in a pipeline can be annotated with
allowed data ranges for inputs and outputs, as well as units.

```
class LatLonTransformer(PipelineElement):
    """
    The LatLonTransformer will consume a lat(itude) and a lon(gitude) column and perform
    cyclic normalization. It will add four columns to a dataframe, namely lat_x, lat_y, lon_x, lon_y.
    """
    @damast.core.describe("Lat/Lon cyclic transformation")
    @damast.core.input({
        "lat": {"unit": units.deg},
        "lon": {"unit": units.deg}
    })
    @damast.core.output({
        "lat_x": {"value_range": MinMax(-1.0, 1.0)},
        "lat_y": {"value_range": MinMax(-1.0, 1.0)},
        "lon_x": {"value_range": MinMax(-1.0, 1.0)},
        "lon_y": {"value_range": MinMax(-1.0, 1.0)}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat_cyclic_transformer = CycleTransformer(features=["lat"], n=180.0)
        lon_cyclic_transformer = CycleTransformer(features=["lon"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        df._dataframe = _df
        return df
```

For detailed examples, check the documentation at: https://simula.github.io/damast

## Installation and Development Setup

Firstly, you will want to create you an isolated development environment for Python, that being conda or venv-based.
The following will go through a venv based setup.

Let us assume you operate with a 'workspace' directory for this project:

```
    cd workspace
```

Here, you will create a virtual environment.
Get an overview over venv (command):

```
    python -m venv --help
```

Create your venv and activate it:
```
    python -m venv damast-venv
    source damast-venv/bin/activate
```

Clone the repo and install:

```
    git clone https://github.com/simula/damast
    cd damast
    pip install -e ".[test,dev]"
```

or alternatively:
```
    pip install damast[test,dev]
```

## Docker Container

If you prefer to work or start with a docker container you can build it using the provided [Dockerfile](https://github.com/simula/damast/blob/main/Dockerfile)
```
    docker build -t damast:latest -f Dockerfile .
```

To enter the container:
```
    docker run -it --rm damast:latest /bin/bash
```

## Usage

To get the usage documentation it is easiest to check the published documentation [here](https://simula.github.io/damast/README.html).

Otherwise, you can also locally generate the latest documentation once you installed the package:
```
    tox -e build_docs
```
Then open the documentation with a browser:
```
    <yourbrowser> _build/html/index.html
```


## Testing

Install the project and use the predefined default test environment:

    tox -e py

## Contributing

This project is open to contributions. For details on how to contribute please check the [Contribution Guidelines](https://github.com/simula/damast/blob/main/CONTRIBUTING.md)

## License
This project is licensed under the [BSD-3-Clause License](https://github.com/simula/damast/blob/main/LICENSE).

## Copyright

Copyright (c) 2023-2025 [Simula Research Laboratory, Oslo, Norway](https://www.simula.no/research/research-departments)

## Acknowledgments

This work has been derived from work that is part of the [T-SAR project](https://www.simula.no/research/projects/t-sar)
Some derived work is mainly part of the specific data processing for the 'maritime' domain.

The development of this library is part of the EU-project [AI4COPSEC](https://ai4copsec.eu) which receives funding
 from the Horizon Europe framework programme under Grant Agreement N. 101190021.
