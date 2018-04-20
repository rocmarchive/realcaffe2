#!/bin/bash

# Build the base image
rocm_base_image=rocm-base-"$1"
docker build -f ./rocm-base/Dockerfile --no-cache -t ${rocm_base_image} .
if [ $? -ne 0 ]; then { echo "ERROR: failed base image build!" ; exit 1; } fi

# Build the caffe2 image
caffe2_image=caffe2-"$1"
docker build -f ./caffe2/Dockerfile --build-arg base_image=${rocm_base_image} --no-cache -t ${caffe2_image} .