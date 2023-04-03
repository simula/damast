# Contribution Guidelines

Great see that you are interested in contributing to 'damast'.

To make the process as smooth as possible for all involed parties, you will find some instructions
below.  In case there is some information missing, or you feel lost, do not
hesitate to ask or [open an issue](https://gitlab.com/simula-srl/damast/-/issues/new)
 on this project.

## Development Setup

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
    git clone https://gitlab.com/simula-srl/damast
    cd damast
    pip install -e ".[test,dev]"

```

## Code and development style

Please adhere to the existing code style.
To run a lint checking on the project:

```
    tox -e lint
```

Have a look at the existing module structure and consider where to put your additions to the project.
Are they domain specific, are there general changes? Will your change affect existing code.

In all cases add testcases under 'tests' in a module specific subfolder.
This project uses 'pytest'.
To get familiar with pytest (pytest --help or visit https://docs.pytest.org).
For a quick start, you can run:

```
    pytest
```

to run all tests.

Or:

```
    pytest tests/damast/core/test_metadata.py -k test_change -s
```

to run a particular test case (-k) and redirecting the output to stdout (-s).



## Merge Requests

Please fork the repository and push your code change into a branch.
Then create a [new merge request](https://gitlab.com/simula-srl/damast/-/merge_requests/new) from this branch.

Generally, read through the documentation first to understand the main ideas and some underlying assumptions of this project.
If you intend to change some of the behavior, please try to validate by reading the documentation or opening an issue whether 
there are reasons for a particular setup.
Some change might have minor or major and sometimes unforeseen side-effects.
So make sure, that in all cases you describe your intentions and arguments of a change in the merge request.

Otherwise, please ensure that:

1. your fork and changes are based on damast's latest state of the 'main' branch.
1. `pytest tests` runs without any errors
1. `tox -e lint` runs without any errors


Again, if you require help with any of the above. Please contact the developers.




