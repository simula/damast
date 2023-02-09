FROM python:3.10-slim

RUN apt update
RUN apt install -y \
            graphviz

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Use login shell (-l) in case we update .bashrc
SHELL ["/bin/bash", "-l", "-c"]
