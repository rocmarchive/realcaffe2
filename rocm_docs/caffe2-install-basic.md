# Caffe2 ROCm port: Basic installation

## Intro
This instruction provides a starting point for Caffe2 ROCm port (mostly via deb packages).
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm
```
export ROCM_PATH=/opt/rocm
export DEBIAN_FRONTEND noninteractive
sudo apt update && sudo apt install -y wget software-properties-common 
```

### Add the ROCm repository:  
```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
```

### Install or Update ROCm

Next, update the apt-get repository list and install/update the ROCm package:

Warning: Before proceeding, make sure to completely uninstall any previous ROCm package:

```
sudo apt-get update
sudo apt-get install libnuma-dev
sudo apt-get install rocm-dkms rocm-opencl-dev
```
### Add username to 'video' group and reboot:  
```
sudo adduser $LOGNAME video
sudo reboot
```



