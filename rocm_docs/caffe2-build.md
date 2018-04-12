# Caffe2 ROCm port: Building From Source

## Intro
This instruction provides a starting point for build Caffe2 ROCm port from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm

Follow steps at [Basic Installation](https://github.com/petrex/rocm_caffe2/blob/documentation/rocm_docs/caffe2-install-basic.md) and [Docker Installation](https://github.com/petrex/rocm_caffe2/blob/documentation/rocm_docs/caffe2-docker.md) to install ROCm stack and docker.

Setup environment variables, and add those environment variables at the end of ~/.bashrc 
```
export HCC_HOME=/opt/rocm/hcc
export HIP_PATH=/opt/rocm/hip
export PATH=$HCC_HOME/bin:$HIP_PATH/bin:$PATH
```
In your current working directory, please run the following steps.

## Pull the Latest ROCm_Caffe2 Src
* using https

```
git clone --recursive https://github.com/ROCmSoftwarePlatform/rocm_caffe2.git
```

* using ssh 

```
git clone --recursive git@github.com:ROCmSoftwarePlatform/rocm_caffe2.git
```
## Pull Thrust and cub-hip libraries  
Some operators in ROCm_Caffe2 utilize Thrust library to achieve parallelism. Thrust can be pulled using:

```
git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git
```
After successfully cloning Thrust, please navigate to Thrust/thrust/system/cuda/detail in your current working directory and clone the cub-hip repository there and checkout to hip_port_1.7.4_caffe2 branch. After cloning cub-hip navigate back to your working directory. This can be done using the following commands.

```
cd Thrust/thrust/system/cuda/detail
rm -r cub-hip
git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git
cd cub-hip
git checkout hip_port_1.7.4_caffe2
cd ../../../../../../
```

## Spin off the container
	
`docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $(pwd):/work petrex/rocm_caffe2`

Inside the docker create an environment variable named THRUST_ROOT that points to the Thrust repository.

```
export THRUST_ROOT=/work/Thrust
```

## Build the ROCm_Caffe2 Project from Src

* Navigate to rocm_caffe2 directory and create a directory to put Caffe2's build files in 

	`mkdir build && cd build`

* Configure Caffe2's build 
 
	`cmake ..`

* Compile, link, and install Caffe2 

	`sudo make install`
	
* Test the Caffe2 Installation 
	
	Before running the tests, make sure that the required environment variables are set:
	```
	export PYTHONPATH=/usr/local:<caffe2_home>/build:$PYTHONPATH 
	export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
	```
	Run this to see if your Caffe2 installation was successful. 
	
	`cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`
