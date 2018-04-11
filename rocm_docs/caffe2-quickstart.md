# Caffe2 ROCm port Quickstart Guide


## Running Core Tests
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

## Running Operator Tests
* Before running the tests, make sure that the required environment variables are set:

	```
	export PYTHONPATH=/usr/local:<caffe2_home>/build:$PYTHONPATH 
	export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
	```

* To run directed operator tests: 

	`python -m caffe2.python.operator_test.<test>`

## Running Real Workload
* Inference
* Training


## Known Issues / Workarounds
X freezes under load
ROCm 1.7.1 a kernel parameter `noretry` has been set to 1 to improve overall system performance. However it has been proven to bring instability to graphics driver shipped with Ubuntu. This is an ongoing issue and we are looking into it.

Before that, please try apply this change by changing `noretry` bit to 0.

`echo 0 | sudo tee /sys/module/amdkfd/parameters/noretry` 
Files under /sys won't be preserved after reboot so you'll need to do it every time.

One way to keep noretry=0 is to change `/etc/modprobe.d/amdkfd.conf` and make it be:
`options amdkfd noretry=0`
Once it's done, run sudo `update-initramfs -u`. Reboot and verify` /sys/module/amdkfd/parameters/noretry` stays as 0.

