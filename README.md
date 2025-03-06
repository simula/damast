# damast: Creation of reproducible data processing pipelines

Documentation at: https://simula.github.io/damast

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

## Docker Container

If you prefer to work or start with a docker container you can build it using the provided [Dockerfile](Dockerfile)
```
    docker build -t damast:latest -f Dockerfile .
```

To enter the container:
```
    docker run -it --rm damast:latest /bin/bash
```

## Usage

Once you installed the package you can locally generate the documentation:
```
    tox -e build_docs
```
You can then open the documentation with a browser:
```
    <yourbrowser> _build/html/index.html
```

Otherwise you will find API and usage documentation [here](https://simula-srl.gitlab.io/damast/README.html).




## Testing

Install the project and use the predefined default test environment:

    tox -e py

## Contributing

This project is open to contributions. For details on how to contribute please check the [Contribution Guidelines](CONTRIBUTING.md)

## License
This project is licensed under the [BSD-3-Clause License](LICENSE).

## Copyright

Copyright (c) 2023-2025 [Simula Research Laboratory, Oslo, Norway](https://www.simula.no/research/software-engineering)

## Acknowledgments

This work has been derived from work that is part of the [T-SAR project](https://www.simula.no/research/projects/t-sar)
Some derived work is mainly part of the specific data processing for the 'maritime' domain.

The development of this library is part of the EU-project [AI4COPSEC](https://ai4copsec.eu) which receives funding
 from the Horizon Europe framework programme under Grant Agreement N. 101190021
