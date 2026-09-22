FROM ubuntu:24.04 AS mrd_converter

LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --no-install-suggests --yes \
     wget apt-transport-https gnupg2 ca-certificates \
     libfftw3-dev git build-essential gfortran libpng-dev liblapacke-dev libopenblas-dev


# install mkl
COPY ./mkl.sh /tmp/
RUN /tmp/mkl.sh


RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone -b pulseq https://github.com/pehses/python-ismrmrd-server.git

# Compile the BART MRI toolbox.
RUN  cd /opt/code \
     && git clone https://codeberg.org/mrirecon/bart.git \
     && cd bart \
     && git checkout v1.0.01 \
     && echo MKL=1 >> Makefile.local \
     && echo MKL_BASE=/opt/intel/mkl >> Makefile.local \
     && echo PARALLEL=1 >> Makefile.local \
     && make \
     && make install

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM ubuntu:24.04
LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy bart
COPY --from=mrd_converter /usr/local/bin/bart /usr/local/bin/bart

# Copy MKL
COPY --from=mrd_converter /opt/intel/mkl/lib /opt/intel/mkl/lib/
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so  \
    libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3  \
    libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   \
    liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 \
    liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf && \
  echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf && \
  ldconfig && \
  echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

# Dependencies for Python MRD server
RUN  apt-get update --quiet \
     && apt-get install --no-install-recommends --no-install-suggests --yes \
        ca-certificates git python3-dev python3-venv libfftw3-double3 libfftw3-single3 liblapacke libpng16-16

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir h5py intel-openmp scipy scikit-image dipy psutil ismrmrd despike julia

RUN  mkdir -p /opt/code
COPY --from=mrd_converter /opt/code/python-ismrmrd-server /opt/code/python-ismrmrd-server
COPY --from=mrd_converter /opt/code/bart /opt/code/bart

RUN pip3 install --no-cache-dir /opt/code/bart/pkg/python

# Cleanup files not required after compiling
RUN  apt-get remove --yes git python3-pip
RUN  apt-get autoremove --yes 
RUN  apt-get autoclean --yes

# add ismrmrd server start script
RUN echo "#!/bin/bash \n/opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/share/debug/python-ismrmrd-server.log" >> /usr/bin/start_server
RUN chmod +x /usr/bin/start_server


CMD ["/usr/bin/start_server"]
