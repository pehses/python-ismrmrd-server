FROM ubuntu:20.04 AS mrd_converter

LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --no-install-suggests --yes \
       wget apt-utils apt-transport-https gnupg2 python3-dev python3-pip ca-certificates libhdf5-serial-dev h5utils cmake cmake-curses-gui \
        libboost-all-dev libfftw3-dev git build-essential gfortran libpng-dev liblapacke-dev libpng-dev libopenblas-dev libxml2-dev libxslt1-dev 

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone -b pulseq https://github.com/pehses/python-ismrmrd-server.git

# ISMRMRD library
RUN cd /opt/code \
    && git clone https://github.com/ismrmrd/ismrmrd.git \
    && cd ismrmrd \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j $(nproc) \
    && make install

# siemens_to_ismrmrd converter
RUN cd /opt/code \
    && git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git \
    && cd siemens_to_ismrmrd \
    && cp /opt/code/python-ismrmrd-server/parameter_maps/* parameter_maps/ \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j $(nproc) \
    && make install

# compile & install the bart MRI toolbox
RUN  cd /opt/code \
     && wget https://github.com/mrirecon/bart/archive/v0.7.00.tar.gz \
     && tar xzvf v0.7.00.tar.gz \
     && cd bart-0.7.00 \
     && echo PARALLEL=1 >> Makefile.local \
     && make \
     && make install

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM ubuntu:20.04
LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy siemens_to_ismrmrd from last stage and re-add necessary dependencies
COPY --from=mrd_converter /usr/local/lib/libismrmrd*        /usr/local/lib/
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd
COPY --from=mrd_converter /usr/local/bin/bart /usr/local/bin/bart

# create pythonpath for single file packages (without requiring setup.py or __init__.py)
RUN mkdir -p /opt/code/pythonpath
ENV PYTHONPATH "${PYTHONPATH}:/opt/code/pythonpath"

# copy bart python interface to pythonpath
COPY --from=mrd_converter /opt/code/bart-0.7.00/python/* /opt/code/pythonpath/

# Dependencies for Python MRD server
RUN  apt-get update --quiet \
     && apt-get dist-upgrade --yes \
     && apt-get install --no-install-recommends --no-install-suggests --yes \
        ca-certificates nano git gcc python3 python3-pip h5utils libxslt1.1 libfftw3-3 liblapacke \
     && pip3 install --no-cache-dir h5py scikit-image ismrmrd

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone -b mrm_paper https://github.com/pehses/python-ismrmrd-server.git

# Cleanup files not required after compiling
RUN  apt-get remove --yes gcc git python3-pip
RUN  apt-get autoremove --yes 
RUN  apt-get autoclean --yes

# add ismrmrd server start script
RUN echo "#!/bin/bash \n/opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/share/debug/python-ismrmrd-server.log" >> /usr/bin/start_server
RUN chmod +x /usr/bin/start_server


CMD ["/usr/bin/start_server"]
