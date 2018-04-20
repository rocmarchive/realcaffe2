#| This Dockerfile provides a starting point for a ROCm installation of math & DL libs.
FROM ubuntu:xenial-20170619
MAINTAINER Jeff Poznanovic <jeffrey.poznanovic@amd.com>

ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/debian/
#ARG DEB_ROCM_REPO=http://172.27.226.104/artifactory/list/rocm-dkms-release-1.8-ubuntu/
ARG ROCM_PATH=/opt/rocm

ENV DEBIAN_FRONTEND noninteractive
ENV HOME /root/
RUN apt update && apt install -y wget software-properties-common 

# Add rocm repository
RUN apt-get clean all
RUN wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | apt-key add -
RUN sh -c  "echo deb [arch=amd64] $DEB_ROCM_REPO xenial main > /etc/apt/sources.list.d/rocm.list"

# Install misc pkgs
RUN apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  clang-3.8 \
  clang-format-3.8 \
  clang-tidy-3.8 \
  cmake \
  cmake-qt-gui \
  ssh \
  curl \
  apt-utils \
  pkg-config \
  g++-multilib \
  git \
  libunwind-dev \
  libfftw3-dev \
  libelf-dev \
  libncurses5-dev \
  libpthread-stubs0-dev \
  vim \
  gfortran \
  libboost-program-options-dev \
  libssl-dev \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  rpm \
  libnuma-dev \
  wget && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install rocm pkgs
RUN apt-get update --allow-insecure-repositories && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    rocm-dev rocm-libs rocm-utils \
    rocfft miopen-hip miopengemm rocblas hipblas rocrand \
    rocm-profiler cxlactivitylogger && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#    rocm-libs miopen-hip miopengemm \
#    rocm-device-libs \
#    hsa-ext-rocr-dev hsakmt-roct-dev hsa-rocr-dev \
#    rocm-opencl rocm-opencl-dev \
#    rocm-utils \
#    rocm-profiler cxlactivitylogger \

ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/hip
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"
ADD ./docker/ubuntu-16.04-rocm171/rocm-base/target.lst /opt/rocm/bin
ADD ./docker/ubuntu-16.04-rocm171/rocm-base/rocm_agent_enumerator /opt/rocm/bin

# Build HCC from source, cleanup the default HCC to avoid issue
#RUN rm -rf /opt/rocm/hcc-1.0 && rm -rf /opt/rocm/lib/*.bc
#RUN cd ~ && git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git
#RUN cd ~ && mkdir -p build.hcc && cd build.hcc && cmake -DCMAKE_BUILD_TYPE=Release ../hcc
#RUN cd ~/build.hcc && make -j $(nproc) && make package && dpkg -i hcc*.deb 
#RUN ln -s /opt/rocm/hcc-1.0 /opt/rocm/hcc
 
# Build HIP from source:
#RUN cd ~ && git clone https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP.git
#RUN cd ~/HIP && mkdir -p build && cd build && cmake .. && make package -j && dpkg -i *.deb 
 
# Setup environment variables, and add those environment variables at the end of ~/.bashrc 
ARG HCC_HOME=/opt/rocm/hcc
ARG HIP_PATH=/opt/rocm/hip
ARG PATH=$HCC_HOME/bin:$HIP_PATH/bin:$PATH

# Build rocBLAS from source
#RUN cd ~ && git clone -b develop https://github.com/ROCmSoftwarePlatform/rocBLAS.git && cd rocBLAS && ./install.sh -icd
#RUN cd ~ && git clone -b develop https://github.com/ROCmSoftwarePlatform/hipBLAS.git && cd hipBLAS && ./install.sh -icd

# Add hipeigen source 
#RUN git clone -b develop https://github.com/ROCmSoftwarePlatform/hipeigen.git /opt/rocm/hipeigen

# Build MIOpen from source
#RUN cd ~ && git clone https://github.com/RadeonOpenCompute/rocm-cmake.git && cd ~/rocm-cmake && mkdir -p build && cd build && cmake .. && make package -j$(nproc) && dpkg -i ./rocm-cmake*.deb
#RUN cd ~ && git clone https://github.com/RadeonOpenCompute/clang-ocl.git && cd ~/clang-ocl && mkdir -p build && cd build && cmake .. && make package -j$(nproc) && dpkg -i ./rocm-clang-ocl*.deb
#RUN cd ~ && git clone https://github.com/ROCmSoftwarePlatform/MIOpenGEMM.git && cd ~/MIOpenGEMM && mkdir -p build && cd build && cmake .. && make package -j$(nproc) && dpkg -i ./miopengemm*.deb
#RUN cd ~ && git clone https://github.com/ROCmSoftwarePlatform/MIOpen.git
#ADD ./tmp/MLOpen /root/MIOpen
#RUN cd ~/MIOpen && cmake -P install_deps.cmake && \
#    mkdir -p build && cd build && \
#    CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" -D#CMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" .. -DMIOPEN_MAKE_BOOST_PUBLIC=ON && \
#    make -j $(nproc) && make package && dpkg -i ./MIOpen*.deb

WORKDIR $HOME

