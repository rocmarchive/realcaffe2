# rocm-caffe2: Docker Installation

## Install docker
A good reference to docker installation on Ubuntu 16.04 can be found [here](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04).

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
		

### Pull the docker image with all dependecies
* The fastest way to start ROCKING is through docker. You can pull the latest official docker image for rocm-caffe2 project:

```
sudo docker pull rohith612/caffe2:rocm1.8.0-miopen-develop_v2
```
This docker image has all dependencies for caffe2 and ROCM software stack for AMD platform. This docker image has rocm1.8.0 in it. 

### Pull the docker image with rocm-caffe2 pre-installed
* You can pull the docker image with rocm-caffe2 pre-installed without having to build from the source.
For rocm-1.8.0
```
sudo docker pull rocm/caffe2:rocm1.8.0-develop-v1
``` 
For rocm-1.7
```
sudo docker pull rocm/caffe2:rocm1.7-miopen-dev-v1
``` 

Launch the docker using:
```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video <name_of_docker_image>
```
This will launch docker container with inside caffe2 directory with build folder inside it.