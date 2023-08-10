FROM ubuntu:20.04

LABEL maintainer="Marten Veldmann (marten.veldmann@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --no-install-suggests --yes \
       wget ca-certificates libhdf5-serial-dev build-essential libxml2-dev libxslt1-dev\
       libfftw3-dev git liblapacke-dev libopenblas-dev curl openmpi-bin apt-utils libpugixml-dev libxml2-utils

# Install Latest CMake
RUN  apt install software-properties-common --no-install-recommends --no-install-suggests --yes \
     && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
     && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
     && apt update \
     && apt install cmake --no-install-recommends --no-install-suggests --yes

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone -b pulseq_dev https://github.com/pehses/python-ismrmrd-server.git

# ISMRMRD library
RUN cd /opt/code \
    && git clone https://github.com/ismrmrd/ismrmrd.git \
    && cd ismrmrd \
    && git checkout v1.13.6 \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j $(nproc) \
    && make install

### Dependencies for PowerGrid ###

# Download and install NVIDIA HPC SDK
RUN  wget https://developer.download.nvidia.com/hpc-sdk/22.1/nvhpc-22-1_22.1_amd64.deb \
     https://developer.download.nvidia.com/hpc-sdk/22.1/nvhpc-2022_22.1_amd64.deb \
     && apt-get install --yes ./nvhpc-22-1_22.1_amd64.deb ./nvhpc-2022_22.1_amd64.deb

RUN rm nvhpc-*

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
     && cmake ../ -DCMAKE_CXX_COMPILER=pgc++ \
     && make -j $(nproc) && make install

RUN ldconfig

# Disable File locking in hdf5 to support mpi versions of PowerGrid
ENV HDF5_USE_FILE_LOCKING=FALSE

# Expose all GPU to docker
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
