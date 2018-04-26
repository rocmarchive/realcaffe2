# rocm-caffe2 Quickstart Guide


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
* context_test
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
* math_test
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

To run python based operator tests, pytest module can be used. If you do not have pytest module installed, install it using:
```
pip install pytest
```
* Before running the tests, make sure that the required environment variables are set:

	```
	export PYTHONPATH=/usr/local:<caffe2_home>/build:$PYTHONPATH 
	export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
	```
Please navigate to <caffe2_home/build/bin> and run the tests.

* To run directed operator tests: 

	`pytest ../caffe2/python/operator_test/<test_name>`

* To run all the tests present under caffe2/python/operator_test

	`pytest ../caffe2/python/operator_test/`
	
	To ignore runnning any tests, --ignore flag can be used
	
	`pytest ../caffe2/python/operator_test/ --ignore <name_of_test_to_ignore>`

	Multiple --ignore arguments can be passed to ingore mulitple tests. Please read pytest documentation to explore more options at https://docs.pytest.org/en/latest/usage.html
	
Available operator tests are :

* activation_ops_test
* adagrad_test
* adam_test
* apmeter_test
* assert_test
* atomic_ops_test
* batch_box_cox_test
* batch_sparse_to_dense_op_test
* blobs_queue_db_test
* boolean_mask_test
* boolean_unmask_test
* cast_op_test
* channel_shuffle_test
* checkpoint_test
* clip_op_test
* concat_split_op_test
* conditional_test
* conv_test
* conv_transpose_test
* copy_ops_test
* cosine_embedding_criterion_op_test
* counter_ops_test
* crf_test
* cross_entropy_ops_test
* dataset_ops_test
* deform_conv_test
* distance_op_test
* dropout_op_test
* duplicate_operands_test
* elementwise_linear_op_test
* elementwise_logical_ops_test
* elementwise_op_broadcast_test
* elementwise_ops_test
* emptysample_ops_test
* extend_tensor_op_test
* fc_operator_test
* filler_ops_test
* find_op_test
* flatten_op_test
* flexible_top_k_test
* gather_ops_test
* gather_ranges_op_test
* given_tensor_fill_op_test
* glu_op_test
* group_conv_test
* gru_test
* hsm_test
* im2col_col2im_test
* image_input_op_test
* index_hash_ops_test
* index_ops_test
* instance_norm_test
* layer_norm_op_test
* leaky_relu_test
* learning_rate_op_test
* lengths_tile_op_test
* lengths_top_k_ops_test
* listwise_l2r_operator_test
* load_save_test
* loss_ops_test
* lpnorm_op_test
* map_ops_test
* margin_ranking_criterion_op_test
* math_ops_test
* matmul_op_test
* merge_id_lists_op_test
* mkl_conv_op_test
* mkl_packed_fc_op_test
* mkl_speed_test
* mod_op_test
* momentum_sgd_test
* mpi_test
* negate_gradient_op_test
* normalize_op_test
* one_hot_ops_test
* pack_ops_test
* pack_rnn_sequence_op_test
* pad_test
* partition_ops_test
* piecewise_linear_transform_test
* pooling_test
* prepend_dim_test
* python_op_test
* rank_loss_operator_test
* rebatching_queue_test
* record_queue_test
* recurrent_net_executor_test
* recurrent_network_test
* reduce_ops_test
* reduction_ops_test
* relu_op_test
* reshape_ops_test
* resize_op_test
* rmac_regions_op_test
* rnn_cell_test
* segment_ops_test
* selu_op_test
* sequence_ops_test
* shape_inference_test
* sinusoid_position_encoding_op_test
* softmax_ops_test
* softplus_op_test
* sparse_gradient_checker_test
* sparse_lengths_sum_benchmark
* sparse_ops_test
* sparse_to_dense_mask_op_test
* spatial_bn_op_test
* specialized_segment_ops_test
* square_root_divide_op_test
* stats_ops_test
* string_ops_test
* text_file_reader_test
* tile_op_test
* top_k_test
* unique_uniform_fill_op_test
* utility_ops_test
* video_input_op_test
* weighted_sample_test
* weighted_sum_test

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

