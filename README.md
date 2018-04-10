# Caffe2

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Jenkins Build Status](https://ci.pytorch.org/jenkins/job/caffe2-master/badge/icon)](https://ci.pytorch.org/jenkins/job/caffe2-master)
[![Appveyor Build Status](https://img.shields.io/appveyor/ci/Yangqing/caffe2.svg)](https://ci.appveyor.com/project/Yangqing/caffe2)

Caffe2 is a lightweight, modular, and scalable deep learning framework. Building on the original [Caffe](http://caffe.berkeleyvision.org), Caffe2 is designed with expression, speed, and modularity in mind.

## Rocm_Caffe2
Rocm_Caffe2 is the official caffe2 port on AMD platform. The is made posible through HIP and ROCM software stack. AMD also provides native libraries for machine intelligent and deep learning workload. 

RCOM_Caffe2 has been validated on Ubuntu 16.04 LTS and AMD Vega 56/64/MI25; with rocm 1.71 amd MIOPEN 1.3 as of now.

### Prerequisites
* A ROCM enable platform. More info [here](https://rocm.github.io/install.html).

### Installation
#### Install ROCM Software Stack
* First make sure your system is up to date. 

	```
	sudo apt update  
	sudo apt dist-upgrade 
	sudo reboot  
	```
* Verify You Have ROCm Capable GPU Installed int the System

	`lspci | grep -i AMD`

* Verify You Have a Supported Version of Linux

	`uname -m && cat /etc/*release`

	You will see some thing like this for Ubuntu

	```
	x86_64 
	DISTRIB_ID=Ubuntu 
	DISTRIB_RELEASE=16.04 
	DISTRIB_CODENAME=xenial 
	DISTRIB_DESCRIPTION="Ubuntu 16.04.3 LTS"
	```

* Add the Repo Server 

	For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows:
	```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
	``` 
	The gpg key might change, so it may need to be updated when installing a new release. The current rocm.gpg.key is not avialable in a standard key ring distribution, but has the following sha1sum hash: 
	
	`f0d739836a9094004b0a39058d046349aacc1178 rocm.gpg.key`
	
* Install or Update ROCm

	Next, update the apt-get repository list and install/update the ROCm package:

	Warning: Before proceeding, make sure to completely uninstall any previous ROCm package:

	```
	sudo apt-get update
	sudo apt-get install libnuma-dev
	sudo apt-get install rocm-dkms rocm-opencl-dev
	```

* Set User Permission for ROCm

	With move to upstreaming the KFD driver and the support of DKMS, for all Console aka headless user, you will need to add all your users to the ‘video” group by setting the Unix permissions

	Ensure that your user account is a member of the “video” group prior to using the ROCm driver. You can find which groups you are a member of with the following command:

	`groups` 

	To add yourself to the video group you will need the sudo password and can use the following command:


	`sudo usermod -a -G video $LOGNAME`
 
	Once complete, reboot your system.

#### Install docker
A good refernce to docker installation on Ubuntu 16.04 can be found [here](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04).

or try these cmd to get docker community edition.

	sudo apt-get -y install apt-transport-https ca-certificates curl
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo add-apt-repository \
	   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	   $(lsb_release -cs) \
	   stable"
	sudo apt-get update
	sudo apt-get -y install docker-ce
	sudo adduser $LOGNAME docker
		

#### Pull the Official docker image 
* The fatest way to start ROCKING is through docker. You can pull the latest official docker image for rocm_caffe2 proejct:

	```
	docker pull petrex/rocm_caffe2
	```

	This docker image has all dependencies for caffe2 and ROCM software stack for AMD platform.
  
### Start Rocking
#### Pull the Latest Rocm_Caffe2 Src
* using https

	```
	git clone --recursive https://github.com/ROCmSoftwarePlatform/rocm_caffe2.git
	```

* using ssh 

	```
	git clone --recursive git@github.com:ROCmSoftwarePlatform/rocm_caffe2.git
	```

#### Spin off the container

	
`docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $HOME/rocm_caffe2:/rocm_caffe2 petrex/rocm_caffe2`


#### Build the Rocm_caff2 Project from Src

* Create a directory to put Caffe2's build files in 

	`mkdir build && cd build`

* Configure Caffe2's build 
 
	`cmake ..`

* Compile, link, and install Caffe2 

	`sudo make install`
	
* Test the Caffe2 Installation 

	Run this to see if your Caffe2 installation was successful. 
	
	`cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`


#### Running Core Tests
Before running the tests, make sure that the required environment variables are set:

```
export PYTHONPATH=/usr/local:<caffe2_home>/build:$PYTHONPATH 
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

```


Next, Navigate to <caffe2_home>/build/bin run the binaries corresponding to the tests. The test binaries are:

* blob_hip_test
* blob_test
* boolean_unmask_ops_test
* common_subexpression_elimination_test
* common_test
* context_hip_test
* conv_to_nnpack_transform_test
* conv_transpose_op_mobile_test
* cpuid_test
* elementwise_op_hip_test
* elementwise_op_test
* event_hip_test
* event_test
* fatal_signal_asan_no_sig_test
* fixed_divisor_test
* fully_connected_op_hip_test
* fully_connected_op_test
* graph_test
* init_test
* logging_test
* math_hip_test
* module_test
* mpi_test
* net_test
* observer_test
* operator_fallback_hip_test
* operator_hip_test
* operator_schema_test
* operator_test
* parallel_net_test
* pattern_net_transform_test
* predictor_test
* proto_utils_test
* registry_test
* reshape_op_hip_test
* simple_queue_test
* smart_tensor_printer_test
* stats_test
* string_ops_test
* text_file_reader_utils_test
* timer_test
* transform_test
* typeid_test
* utility_ops_hip_test
* utility_ops_test
* workspace_test

#### Running Operator Tests
* Before running the tests, make sure that the required environment variables are set:

	```
	export PYTHONPATH=/usr/local:<caffe2_home>/build:$PYTHONPATH 
	export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
	```

* To run directed operator tests: 

	`python -m caffe2.python.operator_test.<test>`

#### Running Real Workload
* Inference
* Training

## Questions and Feedback

Please use Github issues (https://github.com/caffe2/caffe2/issues) to ask questions, report bugs, and request new features.

Please participate in our survey (https://www.surveymonkey.com/r/caffe2). We will send you information about new releases and special developer events/webinars.


## License

Caffe2 is released under the [Apache 2.0 license](https://github.com/caffe2/caffe2/blob/master/LICENSE). See the [NOTICE](https://github.com/caffe2/caffe2/blob/master/NOTICE) file for details.

### Further Resources on [Caffe2.ai](http://caffe2.ai)

* [Installation](http://caffe2.ai/docs/getting-started.html)
* [Learn More](http://caffe2.ai/docs/learn-more.html)
* [Upgrading to Caffe2](http://caffe2.ai/docs/caffe-migration.html)
* [Datasets](http://caffe2.ai/docs/datasets.html)
* [Model Zoo](http://caffe2.ai/docs/zoo.html)
* [Tutorials](http://caffe2.ai/docs/tutorials.html)
* [Operators Catalogue](http://caffe2.ai/docs/operators-catalogue.html)
* [C++ API](http://caffe2.ai/doxygen-c/html/classes.html)
* [Python API](http://caffe2.ai/doxygen-python/html/namespaces.html)
