# Caffe2 ROCm port: Docker Installation

## Install docker
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
		

### Pull the Official docker image 
* The fatest way to start ROCKING is through docker. You can pull the latest official docker image for rocm_caffe2 proejct:

```
docker pull petrex/rocm_caffe2
```

This docker image has all dependencies for caffe2 and ROCM software stack for AMD platform.