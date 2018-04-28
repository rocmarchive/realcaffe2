#!/bin/bash

# Build the rocm-caffe2 image
image_name=$1
docker build -f ./docker/ubuntu-16.04-rocm171/Dockerfile --no-cache -t ${image_name} .
if [ $? -ne 0 ]; then { echo "ERROR: failed image build!" ; exit 1; } fi

