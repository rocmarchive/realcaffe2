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

## Pull the Latest Rocm_Caffe2 Src
* using https

```
git clone --recursive https://github.com/ROCmSoftwarePlatform/rocm_caffe2.git
```

* using ssh 

```
git clone --recursive git@github.com:ROCmSoftwarePlatform/rocm_caffe2.git
```

## Spin off the container
	
`docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $HOME/rocm_caffe2:/rocm_caffe2 petrex/rocm_caffe2`

## Build the Rocm_caff2 Project from Src

* Create a directory to put Caffe2's build files in 

	`mkdir build && cd build`

* Configure Caffe2's build 
 
	`cmake ..`

* Compile, link, and install Caffe2 

	`sudo make install`
	
* Test the Caffe2 Installation 

	Run this to see if your Caffe2 installation was successful. 
	
	`cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`
