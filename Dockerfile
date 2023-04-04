FROM python:3.10-slim

RUN apt update
RUN apt upgrade -y

ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y \
            git graphviz tzdata locales sudo
RUN echo "Europe/Oslo" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV SHELL /bin/bash
RUN locale-gen $LANG
RUN dpkg-reconfigure locales

RUN useradd -ms /bin/bash docker
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER docker
WORKDIR /home/docker

RUN git config --global user.email ""
RUN git config --global user.name "Damast CI User"

ENV PATH=/home/docker/.local/bin:$PATH

RUN git clone https://gitlab.com/simula-srl/damast
RUN pip install -e "damast[test,dev]"
RUN damast --help

# Use login shell (-l) in case we update .bashrc
SHELL ["/bin/bash", "-l", "-c"]
