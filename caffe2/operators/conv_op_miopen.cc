/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/context_hip.h"
#include "caffe2/core/miopen_wrapper.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_miopen.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default miopen workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

class MiopenConvOpBase : public ConvPoolOpBase<HIPContext> {
 public:
  MiopenConvOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<HIPContext>(operator_def, ws),
        miopen_wrapper_(&context_),
        miopen_ws_nbytes_limit_(OperatorBase::GetSingleArgument<size_t>("ws_nbytes_limit", kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        exhaustive_search_(OperatorBase::GetSingleArgument<int>("exhaustive_search", 0)),
        miopen_state_(OperatorBase::GetSingleArgument<int>("miopen_state", 0)),
    }
//TBD PYEH
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bottom_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bias_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&weight_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_for_bias_));
    MIOPEN_ENFORCE(miopenCreateConvolutionDescriptor(&conv_desc_));
  }

  ~MiopenConvOpBase() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bottom_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bias_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(weight_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_for_bias_));
    MIOPEN_ENFORCE(miopenDestroyConvolutionDescriptor(conv_desc_));
  }

 protected:

  vector<TIndex> miopen_input_dims_;
  vector<TIndex> miopen_filter_dims_;
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t bottom_desc_;
  miopenTensorDescriptor_t bias_desc_;
  miopenTensorDescriptor_t weight_desc_;
  miopenTensorDescriptor_t top_desc_;
  miopenTensorDescriptor_t top_desc_for_bias_;
  miopenConvolutionDescriptor_t conv_desc_;
  const size_t miopen_ws_nbytes_limit_;
  size_t workSpaceSize;
  bool exhaustive_search_;
  const float alpha_;
  const float beta_;
  size_t miopen_state_;
};


class MiopenConvOp final : public MiopenConvOpBase {
 public:
  MiopenConvOp(const OperatorDef& operator_def, Workspace* ws)
      : MiopenConvOpBase(operator_def, ws)  {}

  ~MiopenConvOp() {}

  template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  miopenConvolutionFwdAlgo_t algo_;
  AlgorithmsCache<miopenConvolutionFwdAlgo_t> algo_cache_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

class MiopenConvGradientOp final : public MiopenConvOpBase {
 public:
  MiopenConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : MiopenConvOpBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }

  ~MiopenConvGradientOp() {}

  template <typename T_X, typename T_DY, typename T_W, typename T_B,
            typename MATH,
            typename T_DX, typename T_DW, typename T_DB>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  miopenConvolutionBwdWeightsAlgo_t bwd_weight_algo_;
  miopenConvolutionBwdDataAlgo_t bwd_data_algo_;
  AlgorithmsCache<miopenConvolutionBwdWeightsAlgo_t> filter_algo_cache_;
  AlgorithmsCache<miopenConvolutionBwdDataAlgo_t> data_algo_cache_;
  bool no_bias_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
bool MiopenConvOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);

  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);
  const int M = filter.dim32(0);
  ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, M);
  int N = 0, C = 0, H = 0, W = 0, D = 0, N_out = 0, C_out= 0, H_out = 0, W_out = 0, D_out = 0;

  N = X.dim32(0);
  C = X.dim32(1);
  H = X.dim32(2);
  W = X.ndim() > 3 ? X.dim32(3) : 1;
  D = X.ndim() > 4 ? X.dim32(4) : 1;

  N_out = Y->dim32(0);
  C_out = Y->dim32(1);
  H_out = Y->dim32(2);
  W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
  D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;

  cont int requestAlgoCount = 1;
  int returnedAlgoCount = 0;
  miopenConvAlgoPerf_t perf;
  //MIOPEN set output tensor
  MIOPEN_ENFORCE(miopenGetConvolutionForwardOutputDim(conv_desc_,
                                                        bottom_desc_,
                                                        weight_desc_,
                                                        &N_out,
                                                        &C_out,
                                                        &H_out,
                                                        &W_out));

  MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(top_desc_,
                                               miopenTypeWrapper<T>::type,
                                               N_out,
                                               C_out,
                                               H_out,
                                               W_out));

  //MIOPEN fwd get work space size
  MIOPEN_ENFORCE(miopenConvolutionForwardGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                            weight_desc_,
                                                            bottom_desc_,
                                                            conv_desc_,
                                                            top_desc_,
                                                            &workSpaceSize));

    std::vector<char> workspace(workSpaceSize);

  // MIOPEN fwd find conv algorithm
  MIOPEN_ENFORCE(miopenFindConvolutionForwardAlgorithm(miopen_wrapper_.inline_miopen_handle(),
                                                         bottom_desc_,
                                                         X.template data<T_X>() ,
                                                         weight_desc_,
                                                         W.template data<T_W>(),
                                                         conv_desc_,
                                                         top_desc_,
                                                         Y.template data<T_Y>(),
                                                         requestAlgoCount,
                                                         &returnedAlgoCount,
                                                         &perf,
                                                         &workspace,// state->workspace().get(workSpaceSize);
                                                         workSpaceSize,
                                                         exhaustive_search_ ));

  // Set the convolution descriptor
  MIOPEN_ENFORCE(miopenInitConvolutionDescriptor(
          conv_desc_,
          miopenConvolution,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w()));


    // MIOPEN CONV fwd
  MIOPEN_ENFORCE(miopenConvolutionForward(miopen_wrapper_.inline_miopen_handle(),
                                          &alpha_,
                                          bottom_desc_,
                                          X.template data<T_X>() ,
                                          weight_desc_,
                                          W.template data<T_W>(),
                                          conv_desc_,
                                          perf.fwd_algo,
                                          &beta_,
                                          top_desc_,
                                          Y.template data<T_Y>(),
                                          &workspace,  // state->workspace().get(workSpaceSize);
                                          workSpaceSize));

  return true;
}

bool MiopenConvOp::RunOnDevice() {

  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,      // X
                         float,      // W
                         float,      // B
                         float,      // Math
                         float>();   // Y
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,      // X
                         float16,      // W
                         float16,      // B
                         float,      // Math
                         float16>();   // Y
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "miopen convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).meta().name() << "]";
  }
  return true;
}

template <typename T_X, typename T_DY, typename T_W, typename T_B,
          typename MATH,
          typename T_DX, typename T_DW, typename T_DB>
bool MiopenConvGradientOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);

  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);

  const int M = filter.dim32(0);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = dY.dim32(1);
      W_out = dY.ndim() > 3 ? dY.dim32(2) : 1;
      D_out = dY.ndim() > 4 ? dY.dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = dY.dim32(2);
      W_out = dY.ndim() > 3 ? dY.dim32(3) : 1;
      D_out = dY.ndim() > 4 ? dY.dim32(4) : 1;
      CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      }
      group_offset_X = C / group_ * H * W * D;
      group_offset_Y = M / group_ * H_out * W_out * D_out;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  int group_offset_filter = filter.size() / group_;
  if (kernel_.size() == 1) {
    ConvPoolOpBase<HIPContext>::ComputePads({H});
  } else if (kernel_.size() == 2) {
    ConvPoolOpBase<HIPContext>::ComputePads({H, W});
  } else if (kernel_.size() == 3) {
    ConvPoolOpBase<HIPContext>::ComputePads({H, W, D});
  } else {
    CAFFE_THROW("Unsupported kernel size:", kernel_.size());
  }
  dfilter->ResizeLike(filter);

  // Set up the miopen algorithms & workspace if necessary
  bool input_changed = (X.dims() != miopen_input_dims_);
  bool filter_changed = (filter.dims() != miopen_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the miopen descriptor configurations.";
    if (input_changed) {
      miopen_input_dims_ = X.dims();
      SetTensorNdDescriptorWithGroup<T_X>(
          X.ndim(), bottom_desc_, N, C, H, W, D);
    }
    if (filter_changed) {
      miopen_filter_dims_ = filter.dims();
      if (kernel_.size() == 2) {
        MIOPEN_ENFORCE(miopenSetFilter4dDescriptor(
            filter_desc_,
            miopenTypeWrapper<T_W>::type,
            GetMiopenTensorFormat(order_),
#if MIOPEN_VERSION_MIN(7,0,0)
            M,
#else
            M / group_,
#endif
            C / group_,
            kernel_h(),
            kernel_w()));
      } else {
        vector<int> dims(filter.dims().begin(), filter.dims().end());
#if !MIOPEN_VERSION_MIN(7,0,0)
        dims[0] /= group_;
#endif
        order_ == StorageOrder::NCHW ? dims[1] /= group_
                                     : dims[filter.ndim() - 1] /= group_;
        MIOPEN_ENFORCE(miopenSetFilterNdDescriptor(
            filter_desc_,
            miopenTypeWrapper<T_W>::type,
            GetMiopenTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
      if (!no_bias_) {
        if (kernel_.size() == 2) {
          MIOPEN_ENFORCE(miopenSetTensor4dDescriptor(
              bias_desc_,
              GetMiopenTensorFormat(order_),
              miopenTypeWrapper<T_B>::type,
              1,
              M,
              1,
              1));
        } else {
          std::vector<int> bias_dims(X.ndim(), 1);
          bias_dims[1] = M;
          std::vector<int> strides = {M, 1, 1, 1, 1, 1};
          MIOPEN_ENFORCE(miopenSetTensorNdDescriptor(
              bias_desc_,
              miopenTypeWrapper<T_B>::type,
              X.ndim() > 3 ? X.ndim() : 4,
              bias_dims.data(),
              strides.data()));
        }
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_DX>(
        X.ndim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the output with descriptor useful for bias addition in one run.
    if (kernel_.size() == 2) {
      MIOPEN_ENFORCE(miopenSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetMiopenTensorFormat(order_),
          miopenTypeWrapper<T_B>::type,
          N,
          M,
          H_out,
          W_out));
    } else {
      vector<int> dims = {N, M, H_out, W_out, D_out};
      vector<int> strides = {M * H_out * W_out * D_out,
                             H_out * W_out * D_out,
                             W_out * D_out,
                             D_out,
                             1};
      MIOPEN_ENFORCE(miopenSetTensorNdDescriptor(
          top_desc_for_bias_,
          miopenTypeWrapper<T_B>::type,
          X.ndim() > 3 ? X.ndim() : 4,
          dims.data(),
          strides.data()));
    }
    // Set the convolution descriptor
#if MIOPEN_VERSION_MIN(6,0,0)
    if (kernel_.size() == 2) {
      MIOPEN_ENFORCE(miopenSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          MIOPEN_CROSS_CORRELATION,
          miopenTypeWrapper<MATH>::type));
    } else {
      MIOPEN_ENFORCE(miopenSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          dilation_.data(),
          MIOPEN_CROSS_CORRELATION,
          miopenTypeWrapper<MATH>::type));
    }
#else
    if (kernel_.size() == 2) {
      MIOPEN_ENFORCE(miopenSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          1,
          1,
          MIOPEN_CROSS_CORRELATION));
    } else {
      vector<int> ones(dilation_.size(), 1);
      MIOPEN_ENFORCE(miopenSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          ones.data(),
          MIOPEN_CROSS_CORRELATION,
          miopenTypeWrapper<MATH>::type));
    }
#endif

#if MIOPEN_VERSION_MIN(7,0,0)
    // enable TensorCore math if desired
    enable_tensor_core_ &= TensorCoreAvailable();
    if (enable_tensor_core_) {
      MIOPEN_ENFORCE(miopenSetConvolutionMathType(
            conv_desc_, MIOPEN_TENSOR_OP_MATH));
    }

    // set cuDNN groups if appropriate
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc_, group_));
#endif

    // Set the workspace
    size_t bwd_filter_ws_size, bwd_data_ws_size;

    // Choose dW algorithm
    if (force_algo_[ALGO_WGRAD] >= 0) {
      bwd_weight_algo_ = (miopenConvolutionBwdWeightsAlgo_t)force_algo_[ALGO_WGRAD];
    } else if (deterministic_) {
      bwd_weight_algo_ = MIOPEN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (exhaustive_search_) {
      bwd_weight_algo_ =
          filter_algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
            VLOG(1) << "MIOPEN Convolution bwd: doing filter exhaustive search.";
            // When we do an exhaustive search, we will ignore the workspace
            // size
            // limit and simply go for the fastest algorithm. If you happen to
            // run
            // out of memory later, you will be on your own...
            int returned_algo_count;
            // We clean up the current workspace memory so that the forward
            // algorithm is free to allocate memory.
            // Actually run the search.
            std::array<
                miopenConvolutionBwdFilterAlgoPerf_t,
                kNUM_MIOPEN_BWD_FILTER_ALGS>
                filter_perf_stat;

            miopen_wrapper_.with_miopen_state(
                miopen_state_, [&](CuDNNState* state) {
                  MIOPEN_ENFORCE(miopenFindConvolutionBackwardFilterAlgorithmEx(
                      state->miopen_handle(),
                      bottom_desc_,
                      X.template data<T_X>(),
                      top_desc_,
                      dY.template data<T_DY>(),
                      conv_desc_,
                      filter_desc_,
                      dfilter->template mutable_data<T_DW>(),
                      kNUM_MIOPEN_BWD_FILTER_ALGS,
                      &returned_algo_count,
                      filter_perf_stat.data(),
                      state->workspace().get(miopen_ws_nbytes_limit_),
                      miopen_ws_nbytes_limit_));
                });
            LogCuDNNPerfStats(filter_perf_stat, returned_algo_count);
            return filter_perf_stat[0].algo;
          });
    } else {
      // choose backward algorithm for filter
      MIOPEN_ENFORCE(miopenGetConvolutionBackwardFilterAlgorithm(
          miopen_wrapper_.inline_miopen_handle(),
          bottom_desc_,
          top_desc_,
          conv_desc_,
          filter_desc_,
          MIOPEN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          miopen_ws_nbytes_limit_,
          &bwd_weight_algo_));
    }
    // Pick dX algo if needed
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      if (force_algo_[ALGO_DGRAD] >= 0) {
        bwd_data_algo_ = (miopenConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
      } else if (deterministic_) {
        bwd_data_algo_ = MIOPEN_CONVOLUTION_BWD_DATA_ALGO_1;
      } else if (exhaustive_search_) {
        bwd_data_algo_ =
            data_algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
              VLOG(1) << "MIOPEN Convolution bwd: doing data exhaustive search.";
              int returned_algo_count;

              std::array<
                  miopenConvolutionBwdDataAlgoPerf_t,
                  kNUM_MIOPEN_BWD_DATA_ALGS>
                  data_perf_stat;
              miopen_wrapper_.with_miopen_state(
                  miopen_state_, [&](CuDNNState* state) {
                    auto* dX =
                        Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
                    dX->ResizeLike(X);
                    const T_W* filter_data = filter.template data<T_W>();
                    const T_DY* dYdata = dY.template data<T_DY>();
                    T_DX* dXdata = dX->template mutable_data<T_DX>();
                    MIOPEN_ENFORCE(miopenFindConvolutionBackwardDataAlgorithmEx(
                        state->miopen_handle(),
                        filter_desc_,
                        filter_data,
                        top_desc_,
                        dYdata,
                        conv_desc_,
                        bottom_desc_,
                        dXdata,
                        kNUM_MIOPEN_BWD_DATA_ALGS,
                        &returned_algo_count,
                        data_perf_stat.data(),
                        state->workspace().get(miopen_ws_nbytes_limit_),
                        miopen_ws_nbytes_limit_));
                  });

              LogCuDNNPerfStats(data_perf_stat, returned_algo_count);
              return data_perf_stat[0].algo;
            });
      } else {
        MIOPEN_ENFORCE(miopenGetConvolutionBackwardDataAlgorithm(
            miopen_wrapper_.inline_miopen_handle(),
            filter_desc_,
            top_desc_,
            conv_desc_,
            bottom_desc_,
            MIOPEN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            miopen_ws_nbytes_limit_,
            &bwd_data_algo_));
      }
    }

    // get workspace for backwards filter algorithm
    MIOPEN_ENFORCE(miopenGetConvolutionBackwardFilterWorkspaceSize(
        miopen_wrapper_.inline_miopen_handle(),
        bottom_desc_,
        top_desc_,
        conv_desc_,
        filter_desc_,
        bwd_weight_algo_,
        &bwd_filter_ws_size));
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // get workspace for backwards data algorithm
      MIOPEN_ENFORCE(miopenGetConvolutionBackwardDataWorkspaceSize(
          miopen_wrapper_.inline_miopen_handle(),
          filter_desc_,
          top_desc_,
          conv_desc_,
          bottom_desc_,
          bwd_data_algo_,
          &bwd_data_ws_size));
    } else {
      bwd_data_ws_size = 0;
    }
    miopen_ws_nbytes_ = std::max(bwd_filter_ws_size, bwd_data_ws_size);

    VLOG(1) << "CuDNN bwd algorithm: " << bwd_weight_algo_ << ", "
            << bwd_data_algo_;
    VLOG(1) << "CuDNN workspace size: " << miopen_ws_nbytes_;
  }

  // Now, actually run the computation.
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    dbias->Resize(M);
    MIOPEN_ENFORCE(miopenConvolutionBackwardBias(
        miopen_wrapper_.inline_miopen_handle(),
        miopenTypeWrapper<T_DY>::kOne(),
        top_desc_for_bias_,
        dY.template data<T_DY>(),
        miopenTypeWrapper<T_DB>::kZero(),
        bias_desc_,
        dbias->template mutable_data<T_DB>()));
  }

#if MIOPEN_VERSION_MIN(7,0,0)
  miopen_wrapper_.with_miopen_state(miopen_state_, [&](CuDNNState* state) {
    MIOPEN_ENFORCE(miopenConvolutionBackwardFilter(
        state->miopen_handle(),
        miopenTypeWrapper<T_X>::kOne(),
        bottom_desc_,
        X.template data<T_X>(),
        top_desc_,
        dY.template data<T_DY>(),
        conv_desc_,
        bwd_weight_algo_,
        state->workspace().get(miopen_ws_nbytes_),
        miopen_ws_nbytes_,
        miopenTypeWrapper<T_DW>::kZero(),
        filter_desc_,
        dfilter->template mutable_data<T_DW>()));
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // Compute the gradient w.r.t. the input.
      auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
      dX->ResizeLike(X);
      MIOPEN_ENFORCE(miopenConvolutionBackwardData(
          state->miopen_handle(),
          miopenTypeWrapper<T_W>::kOne(),
          filter_desc_,
          filter.template data<T_W>(),
          top_desc_,
          dY.template data<T_DY>(),
          conv_desc_,
          bwd_data_algo_,
          state->workspace().get(miopen_ws_nbytes_),
          miopen_ws_nbytes_,
          miopenTypeWrapper<T_DX>::kZero(),
          bottom_desc_,
          dX->template mutable_data<T_DX>()));
    }
  });
#else
  for (int i = 0; i < group_; ++i) {
    miopen_wrapper_.with_miopen_state(miopen_state_, [&](CuDNNState* state) {
      MIOPEN_ENFORCE(miopenConvolutionBackwardFilter(
          state->miopen_handle(),
          miopenTypeWrapper<T_X>::kOne(),
          bottom_desc_,
          X.template data<T_X>() + i * group_offset_X,
          top_desc_,
          dY.template data<T_DY>() + i * group_offset_Y,
          conv_desc_,
          bwd_weight_algo_,
          state->workspace().get(miopen_ws_nbytes_),
          miopen_ws_nbytes_,
          miopenTypeWrapper<T_DW>::kZero(),
          filter_desc_,
          dfilter->template mutable_data<T_DW>() + i * group_offset_filter));
      if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
        // Compute the gradient w.r.t. the input.
        auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
        dX->ResizeLike(X);
        MIOPEN_ENFORCE(miopenConvolutionBackwardData(
            state->miopen_handle(),
            miopenTypeWrapper<T_W>::kOne(),
            filter_desc_,
            filter.template data<T_W>() + i * group_offset_filter,
            top_desc_,
            dY.template data<T_DY>() + i * group_offset_Y,
            conv_desc_,
            bwd_data_algo_,
            state->workspace().get(miopen_ws_nbytes_),
            miopen_ws_nbytes_,
            miopenTypeWrapper<T_DX>::kZero(),
            bottom_desc_,
            dX->template mutable_data<T_DX>() + i * group_offset_X));
      }
    });
  }
#endif
  return true;
}

// TODO(Yangqing): a lot of the function contents are very similar. Consider
// consolidating them.
bool MiopenConvGradientOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,    //  X
                         float,    // dY
                         float,    //  W
                         float,    //  b
                         float,    // Math
                         float,    // dX
                         float,    // dW
                         float>(); // db
  }
  else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,    //  X
                         float16,    // dY
                         float16,    //  W
                         float16,    //  b
                         float,    // Math
                         float16,    // dX
                         float16,    // dW
                         float16>(); // db
  } else {
    LOG(FATAL) << "Unsupported input types";
  }
  return true;
}

REGISTER_MIOPEN_OPERATOR(Conv, MiopenConvOp);
REGISTER_MIOPEN_OPERATOR(ConvGradient, MiopenConvGradientOp);

REGISTER_MIOPEN_OPERATOR(Conv1D, MiopenConvOp);
REGISTER_MIOPEN_OPERATOR(Conv1DGradient, MiopenConvGradientOp);

REGISTER_MIOPEN_OPERATOR(Conv2D, MiopenConvOp);
REGISTER_MIOPEN_OPERATOR(Conv2DGradient, MiopenConvGradientOp);

REGISTER_MIOPEN_OPERATOR(Conv3D, MiopenConvOp);
REGISTER_MIOPEN_OPERATOR(Conv3DGradient, MiopenConvGradientOp);

}  // namespace caffe2
