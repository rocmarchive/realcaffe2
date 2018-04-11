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
	
Available operator tests are :

* AveragePool
* AveragePool1D
* AveragePool1DGradient
* AveragePool2D
* AveragePool2DGradient
* AveragePool3D
* AveragePool3DGradient
* AveragePoolGradient
* BatchGather
* BatchGatherGradient
* BatchMatMul
* BatchToSpace
* BooleanMask
* BooleanUnmask
* Broadcast
* Cast
* ChannelShuffle
* ChannelShuffleGradient
* Checkpoint
* Clip
* ClipGradient
* CloneCommonWorld
* CloseBlobsQueue
* Col2Im
* ColwiseMax
* ColwiseMaxGradient
* Concat
* ConstantFill
* Conv
* Conv1D
* Conv1DGradient
* Conv2D
* Conv2DGradient
* Conv3D
* Conv3DGradient
* ConvGradient
* ConvTranspose
* ConvTransposeGradient
* Copy
* CopyCPUToGPU
* CopyFromCPUInput
* CopyGPUToCPU
* CopyOnDeviceLike
* Cos
* CosGradient
* CosineEmbeddingCriterion
* CosineEmbeddingCriterionGradient
* CosineSimilarity
* CosineSimilarityGradient
* CountDown
* CountUp
* CreateBlobsQueue
* CreateCommonWorld
* CreateCounter
* CreateDB
* DeformConv
* DeformConvGradient
* DepthConcat
* DepthSplit
* DequeueBlobs
* DiagonalFill
* Div
* DivGradient
* Do
* DotProduct
* DotProductGradient
* Dropout
* DropoutGrad
* ElementwiseLinear
* ElementwiseLinearGradient
* Elu
* EluGradient
* EnqueueBlobs
* EnsureCPUOutput
* EnsureDense
* EQ
* Exp
* ExpandDims
* FC
* FC_Decomp
* FCGradient
* FCGradient_Decomp
* FileStoreHandlerCreate
* Find
* Flatten
* FlattenToVec
* FloatToHalf
* FP16MomentumSGDUpdate
* FP32MomentumSGDUpdate
* Free
* Gather
* GatherPadding
* GaussianFill
* GE
* GetGPUMemoryUsage
* GivenTensorBoolFill
* GivenTensorDoubleFill
* GivenTensorFill
* GivenTensorInt64Fill
* GivenTensorIntFill
* Glu
* GRUUnit
* GRUUnitGradient
* GT
* HalfToFloat
* If
* Im2Col
* ImageInput
* InstanceNorm
* InstanceNormGradient
* Iter
* L1Distance
* L1DistanceGradient
* LabelCrossEntropy
* LabelCrossEntropyGradient
* LayerNorm
* LayerNormGradient
* LE
* LeakyRelu
* LeakyReluGradient
* LearningRate
* LengthsIndicesInGradientSumGradient
* LengthsSum
* LengthsTile
* Load
* Log
* Logit
* LogitGradient
* LpPool
* LpPoolGradient
* LRN
* LRNGradient
* LSTMUnit
* LSTMUnitGradient
* LT
* MakeTwoClass
* MakeTwoClassGradient
* MarginRankingCriterion
* MarginRankingCriterionGradient
* MatMul
* Max
* MaxGradient
* MaxPool
* MaxPool
* MaxPool1D
* MaxPool1DGradient
* MaxPool2D
* MaxPool2DGradient
* MaxPool3D
* MaxPool3DGradient
* MaxPoolGradient
* MaxPoolWithIndex
* MaxPoolWithIndexGradient
* MergeDim
* Min
* MinGradient
* MomentumSGD
* MomentumSGDUpdate
* MPIAllgather
* MPIAllreduce
* MPIBroadcast
* MPICreateCommonWorld
* MPIReceiveTensor
* MPIReduce
* MPISendTensor
* MSRAFill
* Mul
* MultiClassAccuracy
* NanCheck
* NCHW2NHWC
* NegateGradient
* Negative
* NHWC2NCHW
* Normalize
* NormalizeGradient
* NormalizeL1
* Not
* OneHot
* Or
* PackSegments
* PadImage
* PadImageGradient
* Perplexity
* PiecewiseLinearTransform
* Pow
* PRelu
* PReluGradient
* PrependDim
* Print
* Range
* RangeFill
* ReceiveTensor
* RecurrentNetwork
* RecurrentNetworkBlobFetcher
* RecurrentNetworkGradient
* RedisStoreHandlerCreate
* Reduce
* ReduceBackMax
* ReduceBackMaxGradient
* ReduceBackMean
* ReduceBackMeanGradient
* ReduceBackSum
* ReduceBackSumGradient
* ReduceFrontMax
* ReduceFrontMaxGradient
* ReduceFrontMean
* ReduceFrontMeanGradient
* ReduceFrontSum
* ReduceFrontSumGradient
* Relu
* ReluFp16
* ReluFp16Gradient
* ReluGradient
* RemovePadding
* ReplaceNaN
* ResetCounter
* Reshape
* ResizeLike
* ResizeNearest
* ResizeNearestGradient
* RetrieveCount
* ReversePackedSegs
* RMACRegions
* RmsProp
* rnn_internal_accumulate_gradient_input
* rnn_internal_apply_link
* RoIPool
* RoIPoolGradient
* RowwiseMax
* RowwiseMaxGradient
* RowWiseSparseAdagrad
* SafeDequeueBlobs
* SafeEnqueueBlobs
* Save
* Scale
* ScatterAssign
* ScatterWeightedSum
* Selu
* SeluGradient
* SendTensor
* SequenceMask
* Shape
* Sigmoid
* SigmoidCrossEntropyWithLogits
* SigmoidCrossEntropyWithLogitsGradient
* SigmoidGradient
* Sign
* Sin
* SinGradient
* Size
* Slice
* SliceGradient
* Softmax
* SoftmaxGradient
* SoftmaxWithLoss
* SoftmaxWithLossGradient
* Softplus
* SoftplusGradient
* Softsign
* SoftsignGradient
* SortedSegmentRangeLogMeanExp
* SortedSegmentRangeLogMeanExpGradient
* SortedSegmentRangeMean
* SortedSegmentRangeMeanGradient
* SpaceToBatch
* SparseAdagrad
* SparseAdam
* SparseLengthsIndicesInGradientSumGradient
* SparseLengthsIndicesInGradientWeightedSumGradient
* SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
* SparseLengthsSum
* SparseLengthsWeightedSum
* SparseMomentumSGDUpdate
* SparseToDense
* SpatialBN
* SpatialBNGradient
* SpatialSoftmaxWithLoss
* SpatialSoftmaxWithLossGradient
* Split
* Sqr
* SquaredL2Distance
* SquaredL2DistanceGradient
* Squeeze
* StopGradient
* Sub
* Sum
* SumElements
* SumElementsGradient
* Summarize
* SumReduceLike
* SumSqrElements
* Swish
* SwishGradient
* Tanh
* TanhGradient
* TensorProtosDBInput
* Tile
* TileGradient
* TopK
* TopKGradient
* Transpose
* TTContraction
* TTContractionGradient
* UniformFill
* UniformIntFill
* Unique
* UnpackSegments
* UnsafeCoalesce
* UnsortedSegmentMean
* UnsortedSegmentSum
* WeightedSample
* WeightedSigmoidCrossEntropyWithLogits
* WeightedSigmoidCrossEntropyWithLogitsGradient
* WeightedSum
* While
* XavierFill
* Xor
* YellowFin
* ZeroGradient

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

