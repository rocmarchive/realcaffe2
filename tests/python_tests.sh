#!/bin/bash

op_tests=("activation_ops_test.py" "conv_test.py" "pooling_test.py" "softmax_ops_test.py")

for test in "${op_tests[@]}"; do
	python ../caffe2/python/operator_test/$test
	if [ $? -ne 0 ]; then
		echo "$test failed"
		exit 1
	fi
done

echo "All tests passed"
exit 0
