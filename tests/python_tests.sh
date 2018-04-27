#!/bin/bash

failed_tests=()
passed_tests=()
ignore_tests=("conv_test.py",
			  "cudnn_recurrent_test.py",
			  "deform_conv_test.py",
			  "elementwise_op_broadcast_test.py",
			  "gru_test.py",
			  "piecewise_linear_transform_test.py",
			  "pooling_test.py",
			  "recurrent_net_executor_test.py",
			  "rnn_cell_test.py",
			  "sparse_lengths_sum_benchmark.py",
			  "spatial_bn_op_test.py",
			  "top_k_test.py",
			  "video_input_op_test.py")

for test in $(ls ../caffe2/python/operator_test/); do
    if [[ "${ignore_tests[*]}" =~ "$test" ]]; then
        continue
    fi
    pytest ../caffe2/python/operator_test/$test
    if [ $? -eq 0 ]; then
       passed_tests+=($test)
    else
        failed_tests+=($test)
    fi
done

echo "passed test count: ${#passed_tests[@]}"
echo "passed tests:"
echo "${passed_tests[*]}"
echo "failed test count: ${#failed_tests[@]}"
echo "failed tests:"
echo "${failed_tests[*]}"

if [ ${#failed_tests[@]} -eq 0 ]; then
	echo "All tests passed"
	exit 0
else 
	exit 1
 
