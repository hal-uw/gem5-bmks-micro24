FROM ubuntu:16.04

# Needed for add-apt-repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common
# Ubuntu 16.04 does not have a python package new enough for gem5, use a PPA
# RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
# Should be minimal needed packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    findutils \
    file \
    libunwind8 \
    libunwind-dev \
    pkg-config \
    build-essential \
    gcc-multilib \
    g++-multilib \
    git \
    ca-certificates \
    m4 \
    zlib1g \
    zlib1g-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libprotoc-dev \
    libgoogle-perftools-dev \
    python-yaml \
    wget \
    libpci3 \
    libelf1 \
    libelf-dev \
    cmake \
    openssl \
    libssl-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-dev \
    libpng12-dev \
    libffi-dev \
    tcl-dev \
    tk-dev \
    gdb

RUN wget -qO- https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tgz | tar -xzv

WORKDIR Python-3.9.10

RUN ./configure --enable-shared --with-ssl-default-suites=openssl && make && make altinstall
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/libhsakmt/lib:/usr/hsa/lib:/usr/lib

WORKDIR /
# Use python 3.9 by default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.9 1
RUN ln -s /usr/share/pyshared/lsb_release.py /usr/local/lib/python3.9/site-packages/lsb_release.py
# Setuptools is needed for cmake for ROCm build. Install using pip.
RUN pip install -U setuptools scons==3.1.2 six

RUN wget -qO- http://repo.radeon.com/rocm/archive/apt_1.6.4.tar.bz2 | tar -xjv

RUN git clone https://kroarty:551d538320901789d9490861a5a58ecf53a73019@github.com/hal-uw/ROCT-Thunk-Interface.git && \
    mkdir -p /ROCT-Thunk-Interface-build

RUN apt-get install -y libnuma-dev libpci-dev

WORKDIR /ROCT-Thunk-Interface/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc) && cpack -G DEB && dpkg -i *.deb
WORKDIR /

RUN dpkg -i apt_1.6.4/pool/main/h/hsa-ext-rocr-dev/*

RUN git clone https://kroarty:551d538320901789d9490861a5a58ecf53a73019@github.com/hal-uw/ROCR-Runtime.git && \
    mkdir -p /ROCR-Runtime/src/build

WORKDIR /ROCR-Runtime/src/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug -DCPACK_PACKAGE_VERSION=1.1.5- .. && \
    make -j$(nproc) && cpack -G DEB && dpkg -i *.deb
WORKDIR /

RUN dpkg -i apt_1.6.4/pool/main/r/rocm-utils/*

RUN git clone --recursive https://kroarty:551d538320901789d9490861a5a58ecf53a73019@github.com/hal-uw/HCC.git && \
    mkdir -p /HCC/build

# For whatever reason they don't make this directory
RUN mkdir -p /opt/rocm/include

WORKDIR /HCC/build
RUN cmake -DNUM_BUILD_THREADS=$(nproc) -DHSA_AMDGPU_GPU_TARGET="gfx801" .. && make && make package && dpkg -i *.deb && rm -r *
WORKDIR /

RUN dpkg -i apt_1.6.4/pool/main/r/rocm-opencl/*
RUN dpkg -i apt_1.6.4/pool/main/r/rocm-opencl-dev/*

RUN git clone -b update https://kroarty:551d538320901789d9490861a5a58ecf53a73019@github.com/hal-uw/HIP.git && \
    mkdir -p /HIP/build

ENV ROCM_PATH /opt/rocm
ENV HCC_HOME ${ROCM_PATH}/hcc
ENV HSA_PATH ${ROCM_PATH}/hsa
ENV HIP_PATH ${ROCM_PATH}/hip
ENV HIP_PLATFORM hcc
ENV PATH ${ROCM_PATH}/bin:${HCC_HOME}/bin:${HSA_PATH}/bin:${HIP_PATH}/bin:${PATH}
ENV HCC_AMDGPU_TARGET gfx801

WORKDIR /HIP/build
RUN apt-get install -y libelf-dev rpm
RUN cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc) && make package && dpkg -i hip_base*.deb hip_hcc*.deb
WORKDIR /

# Get ROCm libraries we need to compile from source (and ROCm-profiler)
RUN git clone --single-branch https://github.com/ROCmSoftwarePlatform/hipBLAS/ && \
    git clone --single-branch https://github.com/ROCmSoftwarePlatform/rocBLAS/ && \
    git clone --single-branch https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/ && \
    git clone --single-branch https://github.com/RadeonOpenCompute/rocm-cmake/

RUN git clone -b scheduling https://kroarty:551d538320901789d9490861a5a58ecf53a73019@github.com/hal-uw/MIOpen

ARG gem5_dist=http://dist.gem5.org/dist/v21-0

# Apply patches to various repos
RUN mkdir -p /patch && cd /patch && \
    wget ${gem5_dist}/rocm_patches/hipBLAS.patch && \
    wget ${gem5_dist}/rocm_patches/miopen-conv.patch && \
    wget ${gem5_dist}/rocm_patches/rocBLAS.patch

RUN git -C /hipBLAS/ checkout ee57787e && git -C /hipBLAS/ apply /patch/hipBLAS.patch && \
    git -C /rocBLAS/ reset --hard cbff4b4e && git -C /rocBLAS/ apply /patch/rocBLAS.patch && \
    git -C /rocm-cmake/ checkout 12670acb && \
    git -C /MIOpenGEMM/ checkout 9547fb9e

# Create build dirs for machine learning ROCm installs
RUN mkdir -p /rocBLAS/build && \
    mkdir -p /hipBLAS/build && \
    mkdir -p /rocm-cmake/build && \
    mkdir -p /MIOpenGEMM/build && \
    mkdir -p /MIOpen/build

RUN mkdir -p /opt/rocm/hsa/include

# Do the builds, empty build dir to trim image size
WORKDIR /rocm-cmake/build
RUN cmake .. && cmake --build . --target install && rm -rf *

WORKDIR /rocBLAS/build
RUN CXX=/opt/rocm/hcc/bin/hcc cmake -DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801" .. && \
    make -j$(nproc) && make package && dpkg -i rocblas*.deb

WORKDIR /hipBLAS/build
RUN CXX=/opt/rocm/hcc/bin/hcc cmake -DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801" .. && \
    make -j$(nproc) && make package && dpkg -i hipblas*.deb

WORKDIR /MIOpenGEMM/build
RUN cmake .. && make miopengemm && make install && rm -rf *

# Should link this in as a volume if at all possible
RUN mkdir -p /.cache/miopen && chmod 777 /.cache/miopen
# Un-set default c++ version for MIOpen compilation
# As MIOpen 1.7 requires c++14 or higher
RUN sed -i 's/INTERFACE_COMPILE_OPTIONS "-std=c++amp;-fPIC;-gline-tables-only"/#&/' /opt/rocm/hcc/lib/cmake/hcc/hcc-targets.cmake && \
    sed -i 's/INTERFACE_COMPILE_OPTIONS "-hc"/#&/' /opt/rocm/hcc/lib/cmake/hcc/hcc-targets.cmake
WORKDIR /MIOpen
# Half is required; This is the version that MIOpen would download
RUN wget https://github.com/pfultz2/half/archive/1.12.0.tar.gz && \
    tar -xzf 1.12.0.tar.gz
WORKDIR /MIOpen/build
RUN CXX=/opt/rocm/hcc/bin/hcc cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DMIOPEN_BACKEND=HIP \
    -DCMAKE_PREFIX_PATH="/opt/rocm/hip;/opt/rocm/hcc;/opt/rocm/rocdl;/opt/rocm/miopengemm;/opt/rocm/hsa" \
    -DMIOPEN_CACHE_DIR=/.cache/miopen \
    -DMIOPEN_AMDGCN_ASSEMBLER_PATH=/opt/rocm/opencl/bin \
    -DHALF_INCLUDE_DIR=/MIOpen/half-1.12.0/include \
    -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu -DDGPU" .. && \
    make -j$(nproc) && make install && rm -rf *

# Re-set defaults
RUN sed -i 's/#\(INTERFACE_COMPILE_OPTIONS "-std=c++amp;-fPIC;-gline-tables-only"\)/\1/' /opt/rocm/hcc/lib/cmake/hcc/hcc-targets.cmake && \
    sed -i 's/#\(INTERFACE_COMPILE_OPTIONS "-hc"\)/\1/' /opt/rocm/hcc/lib/cmake/hcc/hcc-targets.cmake

# Always use python3 and create a link to config command for gem5 to find
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/local/bin/python3.9-config /usr/bin/python3-config
RUN ln -sf /usr/local/include/python3.9 /usr/include/python3.9


