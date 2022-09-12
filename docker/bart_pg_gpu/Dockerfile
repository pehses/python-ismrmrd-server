FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS mrd_converter

LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --no-install-suggests --yes \
       wget apt-utils apt-transport-https gnupg2 python3-dev python3-pip ca-certificates libhdf5-serial-dev h5utils \
       libfftw3-dev git build-essential gfortran liblapacke-dev libpng-dev libopenblas-dev libboost-all-dev libxml2-dev libxslt1-dev libssl-dev curl

# Install Latest CMake
RUN  apt install software-properties-common --no-install-recommends --no-install-suggests --yes \
     && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
     && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
     && apt update \
     && apt install cmake-data=3.21.1-0kitware1ubuntu20.04.1 --no-install-recommends --no-install-suggests --yes \
     && apt install cmake=3.21.1-0kitware1ubuntu20.04.1 --no-install-recommends --no-install-suggests --yes

# Install mkl
COPY ./mkl.sh /tmp/
RUN /tmp/mkl.sh

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone -b pulseq_dev https://github.com/pehses/python-ismrmrd-server.git

# install spiraltraj for trajectory calculation
RUN  cd /opt/code \
     && git clone https://github.com/mrphysics-bonn/spiraltraj.git \
     && cd spiraltraj \
     && pip3 wheel .
     
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
    && git clone https://github.com/mavel101/siemens_to_ismrmrd.git \
    && cd siemens_to_ismrmrd \
    && git checkout skope_ecc \
    && cp /opt/code/python-ismrmrd-server/parameter_maps/* parameter_maps/ \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j $(nproc) \
    && make install

# compile & install the bart MRI toolbox
RUN  cd /opt/code \
     && git clone https://github.com/mrirecon/bart \
     && cd bart \
     && echo MKL=1 >> Makefile.local \
     && echo MKL_BASE=/opt/intel/mkl >> Makefile.local \
     && echo PARALLEL=1 >> Makefile.local \
     && echo CUDA=1 >> Makefile.local \
     && echo CUDA_BASE=/usr/local/cuda >> Makefile.local \
     && echo CUDA_LIB=/lib64 >> Makefile.local \
     && make \
     && make install

### Dependencies for PowerGrid ###

# Download and install NVIDIA HPC SDK
RUN  wget https://developer.download.nvidia.com/hpc-sdk/22.1/nvhpc-22-1_22.1_amd64.deb \
     https://developer.download.nvidia.com/hpc-sdk/22.1/nvhpc-2022_22.1_amd64.deb \
     && apt-get install --yes ./nvhpc-22-1_22.1_amd64.deb ./nvhpc-2022_22.1_amd64.deb

# Load NVIDIA HPC SDK modules
ENV NVARCH=Linux_x86_64
ENV NVHPC=/opt/nvidia/hpc_sdk
ENV HPCSDK_VERSION=22.1
ENV HPCSDK_RELEASE=2022
ENV CUDA_PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/cuda"
ENV MANPATH="${MANPATH}:${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/compilers/man"
ENV LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/compilers/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_RELEASE}/compilers/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/math_libs/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_RELEASE}/math_libs/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/comm_libs/mpi/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_RELEASE}/comm_libs/mpi/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/comm_libs/nccl/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_RELEASE}/comm_libs/nccl/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/comm_libs/nvshmem/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVHPC}/${NVARCH}/${HPCSDK_RELEASE}/comm_libs/nvshmem/lib:${LD_LIBRARY_PATH}"
ENV OPAL_PREFIX="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/comm_libs/mpi/"
ENV PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/compilers/bin:${PATH}"
ENV PATH="${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/comm_libs/mpi/bin:${PATH}"
ENV MANPATH="${MANPATH}:${NVHPC}/${NVARCH}/${HPCSDK_VERSION}/comm_libs/mpi/man"

# Install custom compile of boost - only necessary runtime components
RUN wget -O ~/boost_1_75_0.tar.bz2 -c 'https://sourceforge.net/projects/boost/files/boost/1.75.0/boost_1_75_0.tar.bz2/download' --max-redirect=100 \
    && apt-get -y remove --purge libboost-all-dev \
    && cd ~ \
    && tar xjf ~/boost_1_75_0.tar.bz2 \
    && cd ~/boost_1_75_0/ \
    && ./bootstrap.sh --prefix=/usr/local --with-toolset=pgi  \
    && cp ~/boost_1_75_0/tools/build/example/user-config.jam ~/user-config.jam \
    && echo "using mpi : mpic++ ; " >> ~/user-config.jam \
    && cd ~/boost_1_75_0/tools/build \
    && ./bootstrap.sh && ./b2 install \
    && cd ~/boost_1_75_0 && b2 toolset=pgi --with-mpi --with-program_options --with-serialization stage install

# Install SuperLU5
RUN cd /opt/code \ 
    && curl -O -J -L https://github.com/xiaoyeli/superlu/archive/v5.2.1.tar.gz \
    && tar xvf superlu-5.2.1.tar.gz \
    && cd superlu-5.2.1 \
    && mkdir build \
    && cd build \
    && cmake ../ -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DBUILD_SHARED_LIBS=ON \
    && make \
    && make install

# Install Armadillo
RUN cd opt/code \
    && git clone https://gitlab.com/conradsnicta/armadillo-code \
    && cd armadillo-code \
    && git checkout --track origin/9.900.x \
    && mkdir build \
    && cd build \
    && echo $PATH \
    && cmake ../ -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc \
    && make \
    && make install \
    && ldconfig

# Install PowerGrid - cnpy version writing numpy arrays
RUN  cd /opt/code \
     && git clone https://github.com/mavel101/PowerGrid \
     && cd PowerGrid \
     && git checkout python_cnpy_dev \
     && git submodule update --init --recursive \
     && mkdir build && cd build \
     && cmake ../ -DCMAKE_CXX_COMPILER=pgc++ -DCMAKE_INSTALL_PREFIX=/usr/local/PowerGrid \
     && make -j4 && make install    

# ----- Start another clean build without all of the build dependencies-----
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy packages from last stage and re-add necessary dependencies
COPY --from=mrd_converter /usr/local/. /usr/local/
COPY --from=mrd_converter /opt/nvidia /opt/nvidia 
COPY --from=mrd_converter /opt/code/PowerGrid /opt/code/PowerGrid
COPY --from=mrd_converter /opt/code/spiraltraj/SpiralTraj* /opt/code/

# Set variables for PowerGrid
ENV LD_LIBRARY_PATH="/usr/local/PowerGrid/lib:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/PowerGrid/bin:${PATH}"
# Disable file locking for PowerGrid MPI-versions
ENV HDF5_USE_FILE_LOCKING=FALSE

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

# Set Pythonpath for BART
ENV TOOLBOX_PATH="/opt/code/bart"
ENV PYTHONPATH="${PYTHONPATH}:${TOOLBOX_PATH}/python"

# copy bart python interface to pythonpath
COPY --from=mrd_converter ${TOOLBOX_PATH} ${TOOLBOX_PATH}

# Dependencies for Python MRD server
RUN  apt-get update --quiet \
     && apt-get dist-upgrade --yes \
     && apt-get install --no-install-recommends --no-install-suggests --yes \
        ca-certificates nano git gcc python3 python3-pip h5utils libxslt1.1 libfftw3-3 liblapacke libhdf5-dev openmpi-bin libatomic1 environment-modules \
     && pip3 install --no-cache-dir h5py intel-openmp scipy scikit-image dipy psutil ismrmrd

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && pip3 install SpiralTraj* \
     && rm SpiralTraj*

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone -b pulseq_dev https://github.com/pehses/python-ismrmrd-server.git

# Cleanup files not required after compiling
RUN  apt-get remove --yes gcc git python3-pip
RUN  apt-get autoremove --yes 
RUN  apt-get autoclean --yes

# add ismrmrd server start script
RUN echo "#!/bin/bash \n/opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/share/debug/python-ismrmrd-server.log" >> /usr/bin/start_server
RUN chmod +x /usr/bin/start_server

# Expose all GPU to docker
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

CMD ["/usr/bin/start_server"]
