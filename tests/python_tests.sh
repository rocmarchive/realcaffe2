#!/bin/bash
rm -r ../caffe2/python/operator_test/__*

failed_tests=()
passed_tests=()
ignore_tests=("cudnn_recurrent_test.py",
			  "deform_conv_test.py",
			  "elementwise_op_broadcast_test.py",
			  "gru_test.py",
			  "piecewise_linear_transform_test.py",
			  "recurrent_net_executor_test.py",
			  "rnn_cell_test.py",
			  "sparse_lengths_sum_benchmark.py",
			  "top_k_test.py",
			  "video_input_op_test.py",
			  "matmul_op_test.py")

for test in $(ls ../caffe2/python/operator_test/); do
    if [[ "${ignore_tests[*]}" =~ "$test" ]]; then
        continue
    fi
    python -m pytest ../caffe2/python/operator_test/$test
    if [ $? -eq 0 ]; then
       passed_tests+=($test)
    else
        failed_tests+=($test)
    fi
done

if [ ${#failed_tests[@]} -eq 0 ]; then
	echo "All operator tests passed"
else 
	exit 1
fi

echo "passed test count: ${#passed_tests[@]}"
echo "passed tests:"
echo "${passed_tests[*]}"
echo "failed test count: ${#failed_tests[@]}"
echo "failed tests:"
echo "${failed_tests[*]}"

echo "running misc tests"
python -m pytest ../caffe2/python/ \
		--ignore=../caffe2/python/operator_test/ \
		--ignore=../caffe2/python/test/ \
		--ignore=../caffe2/python/predictor/ \
		--ignore=../caffe2/python/models/ \
		--ignore=../caffe2/python/modeling/ \
		--ignore=../caffe2/python/mkl/ \
		--ignore=../caffe2/python/data_parallel_model_test.py \
		--ignore=../caffe2/python/memonger_test.py \
		--ignore=../caffe2/python/optimizer_test.py \
		--ignore=../caffe2/python/hypothesis_test.py

exit "$?"




