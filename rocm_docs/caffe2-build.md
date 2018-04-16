# rocm-caffe2: Building From Source

## Intro
This instruction provides a starting point for build rocm-caffe2 (Caffe2 ROCm port) from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm

Follow steps at [Basic Installation](https://github.com/petrex/rocm_caffe2/blob/documentation/rocm_docs/caffe2-install-basic.md) and [Docker Installation](https://github.com/petrex/rocm_caffe2/blob/documentation/rocm_docs/caffe2-docker.md) to install ROCm stack and docker.

Setup environment variables, and add those environment variables at the end of ~/.bashrc 
```
export HCC_HOME=/opt/rocm/hcc
export HIP_PATH=/opt/rocm/hip
export PATH=$HCC_HOME/bin:$HIP_PATH/bin:$PATH
```

## Pull the Latest rocm-caffe2 Src
* using https

```
git clone --recursive https://github.com/ROCmSoftwarePlatform/rocm-caffe2.git
```

* using ssh 

```
git clone --recursive git@github.com:ROCmSoftwarePlatform/rocm-caffe2.git
```

## Spin off the Container
	
`docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $PWD/rocm-caffe2:/rocm-caffe2 petrex/rocaffe2:developer_preview` 

Inside the docker image, navigate to rocm-caffe2 directory

`cd /rocm-caffe2`

## Build the rocm-caffe2 Project from Src

* Create a directory to put rocm-caffe2's build files in 

	`mkdir build && cd build`

* Configure rocm-caffe2's build 
 
	`cmake ..`

* Compile, Link, and Install rocm-caffe2 

	`sudo make install`
	
* Test the rocm-caffe2 Installation 
	Before running the tests, make sure that the required environment variables are set:
	```
	export PYTHONPATH=/usr/local:<caffe2_home>/build:$PYTHONPATH 
	export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
	```

	Run this to see if your rocm-caffe2 installation was successful. 
	
	`cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`
