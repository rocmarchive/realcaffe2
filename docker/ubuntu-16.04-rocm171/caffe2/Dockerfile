# This Dockerfile provides a starting point for a ROCm installation of caffe2 aka rocm-caffe2.
#ARG base_image
#FROM ${base_image}

FROM rocm-base-rocm171
MAINTAINER Rohith Nallamaddi <rohith.nallamaddi@amd.com>

ENV HOME /rocm-caffe2

ENV MIOPEN_DISABLE_CACHE 1

# Required system packages to build hipcaffe on UB16.04:
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    pkg-config \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    libprotobuf-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-pip \                  
    python-numpy python-scipy python3-dev python-yaml \
    libfftw3-dev \
    libelf-dev

RUN pip install \
    future \
    numpy \
    protobuf \
    future \
    pydot \
    opencv-python \ 
    hypothesis

RUN apt-get install -y --no-install-recommends libgflags-dev

WORKDIR /data

RUN git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git

WORKDIR Thrust/thrust/system/cuda/detail

RUN rm -r cub-hip
RUN git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git
WORKDIR cub-hip
RUN git checkout hip_port_1.7.4_caffe2
ENV THRUST_ROOT=/data/Thrust

WORKDIR /rocm-caffe2

