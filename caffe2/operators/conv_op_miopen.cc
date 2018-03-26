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
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default miopen workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
    static constexpr size_t kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

    class MIOPENConvOpBase : public ConvPoolOpBase<HIPContext> {
    public:
        MIOPENConvOpBase(const OperatorDef &operator_def, Workspace *ws)
                : ConvPoolOpBase<HIPContext>(operator_def, ws),
                  miopen_wrapper_(&context_),
                  miopen_ws_nbytes_limit_(OperatorBase::GetSingleArgument<size_t>("ws_nbytes_limit",
                                                                                  kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES)),
                  alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)), //TODO get conv mode
                  beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
                  exhaustive_search_(OperatorBase::GetSingleArgument<int>("exhaustive_search", 0)){
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bottom_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bias_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&weight_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_for_bias_));
            MIOPEN_ENFORCE(miopenCreateConvolutionDescriptor(&conv_desc_));
            mode_ = miopenConvolution;

            MIOPEN_ENFORCE(miopenInitConvolutionDescriptor(
                    conv_desc_,
                    mode_,
                    pad_t(),
                    pad_l(),
                    stride_h(),
                    stride_w(),
                    dilation_h(),
                    dilation_w()));


        }

        ~MIOPENConvOpBase() {
            MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bottom_desc_));
            MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bias_desc_));
            MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(weight_desc_));
            MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_));
            MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_for_bias_));
            MIOPEN_ENFORCE(miopenDestroyConvolutionDescriptor(conv_desc_));
        }

    protected:

        MIOPENWrapper miopen_wrapper_;
        miopenTensorDescriptor_t bottom_desc_;
        miopenTensorDescriptor_t bias_desc_;
        miopenTensorDescriptor_t weight_desc_;
        miopenTensorDescriptor_t top_desc_;
        miopenTensorDescriptor_t top_desc_for_bias_;
        miopenConvolutionDescriptor_t conv_desc_;
        miopenConvolutionMode_t mode_;
        const size_t miopen_ws_nbytes_limit_;
        bool exhaustive_search_;
        const float alpha_;
        const float beta_;
    };


    class MIOPENConvOp final : public MIOPENConvOpBase {
    public:
        MIOPENConvOp(const OperatorDef &operator_def, Workspace *ws)
                : MIOPENConvOpBase(operator_def, ws),
                  requestAlgoCount_(OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
                  returnedAlgoCount_(OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
                  bestAlgoFound_(OperatorBase::GetSingleArgument<bool>("bestAlgoFound_", false))
        {

            MIOPEN_ENFORCE(miopenConvolutionForwardGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                                    weight_desc_,
                                                                    bottom_desc_,
                                                                    conv_desc_,
                                                                    top_desc_,
                                                                    &fwdConvWsSize_));
            hipFree(fwdConvWs);
            HIP_CHECK(hipMalloc(&fwdConvWs, fwdConvWsSize_));
        }

        ~MIOPENConvOp() {
            hipFree(fwdConvWs);
        }

        template<typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
        bool DoRunWithType();
        bool RunOnDevice() override;

    private:
        const int requestAlgoCount_;
        int returnedAlgoCount_;
        bool bestAlgoFound_;
        char* fwdConvWs;
        size_t fwdConvWsSize_;
        miopenConvAlgoPerf_t perf_;
        // Input: X, W, b
        // Output: Y
        INPUT_TAGS(INPUT, FILTER, BIAS);
    };

    class MIOPENConvGradientOp final : public MIOPENConvOpBase {
    public:
        MIOPENConvGradientOp(const OperatorDef &operator_def, Workspace *ws)
                : MIOPENConvOpBase(operator_def, ws),
                  no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)),
                  requestAlgoCount_(OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
                  returnedAlgoCount_(OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
                  bestDataAlgoFound_(OperatorBase::GetSingleArgument<bool>("bestAlgoFound", false)),
                  bestWeightAlgoFound_(OperatorBase::GetSingleArgument<bool>("bestAlgoFound", false)) {
            CAFFE_ENFORCE(
                    !(no_bias_ && OutputSize() == 3),
                    "If bias is not present, you should not have 3 grad output.");

            MIOPEN_ENFORCE(miopenConvolutionBackwardDataGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                                         top_desc_,
                                                                         weight_desc_,
                                                                         conv_desc_,
                                                                         bottom_desc_,
                                                                         &bwdDataWsSize_));
            hipFree(bwdDataWs);
            HIP_CHECK(hipMalloc(&bwdDataWs, bwdDataWsSize_));


            MIOPEN_ENFORCE(miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                                            top_desc_,
                                                                            bottom_desc_,
                                                                            conv_desc_,
                                                                            weight_desc_,
                                                                            &bwdWeightWsSize_));
            hipFree(bwdWeightWs);
            HIP_CHECK(hipMalloc(&bwdWeightWs, bwdWeightWsSize_));

        }

        ~MIOPENConvGradientOp() {
            hipFree(bwdDataWs);
        }

        template<typename T_X, typename T_DY, typename T_W, typename T_B,
                typename MATH,
                typename T_DX, typename T_DW, typename T_DB>
        bool DoRunWithType();
        bool RunOnDevice() override;

    private:
        bool no_bias_;
        const int requestAlgoCount_;
        int returnedAlgoCount_;
        bool bestDataAlgoFound_;
        bool bestWeightAlgoFound_;
        miopenConvAlgoPerf_t perf_;
        size_t bwdWeightWsSize_;
        size_t bwdDataWsSize_;
        char* bwdWeightWs;
        char* bwdDataWs;
        // input: X, W, dY
        // output: dW, db, and optionally dX
        INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
        OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
    };

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

    template<typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
    bool MIOPENConvOp::DoRunWithType() {
        auto &X = Input(INPUT);
        auto &Weight = Input(FILTER);
        auto *Y = Output(0);

        // Figure out the output shape
        CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
        CAFFE_ENFORCE(Weight.ndim() >= 3 && Weight.ndim() <= 5);
        const int M = Weight.dim32(0);
        ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, M);
        int N = 0, C = 0, H = 0, W = 0, D = 0, N_out = 0, C_out = 0, H_out = 0, W_out = 0, D_out = 0;

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


        MIOPEN_ENFORCE(miopenGetConvolutionForwardOutputDim(conv_desc_,
                                                            bottom_desc_,
                                                            weight_desc_,
                                                            &N_out,
                                                            &C_out,
                                                            &H_out,
                                                            &W_out));

        MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(top_desc_,
                                                   miopenTypeWrapper<T_X>::type,
                                                   N_out,
                                                   C_out,
                                                   H_out,
                                                   W_out));


        while(!bestAlgoFound_)
        {

            MIOPEN_ENFORCE(miopenFindConvolutionForwardAlgorithm(miopen_wrapper_.inline_miopen_handle(),
                                                                 bottom_desc_,
                                                                 X.template data<T_X>(),
                                                                 weight_desc_,
                                                                 Weight.template data<T_W>(),
                                                                 conv_desc_,
                                                                 top_desc_,
                                                                 Y->template mutable_data<T_Y>(),
                                                                 requestAlgoCount_,
                                                                 &returnedAlgoCount_,
                                                                 &perf_,
                                                                 fwdConvWs,
                                                                 fwdConvWsSize_,
                                                                 exhaustive_search_));

            bestAlgoFound_ = true;
        }


        MIOPEN_ENFORCE(miopenConvolutionForward(miopen_wrapper_.inline_miopen_handle(),
                                                &alpha_,
                                                bottom_desc_,
                                                X.template data<T_X>(),
                                                weight_desc_,
                                                Weight.template data<T_W>(),
                                                conv_desc_,
                                                perf_.fwd_algo,
                                                &beta_,
                                                top_desc_,
                                                Y->template mutable_data<T_Y>(),
                                                fwdConvWs,
                                                fwdConvWsSize_));

        //BIAS
        if (InputSize() == 3) {
            auto& bias = Input(BIAS);

            CAFFE_ENFORCE_EQ(bias.ndim(), 1);
            CAFFE_ENFORCE_EQ(bias.dim32(0), M);

            MIOPEN_ENFORCE(miopenConvolutionForwardBias(
                    miopen_wrapper_.inline_miopen_handle(),
                    &alpha_,
                    bias_desc_,
                    bias.template data<T_B>(),
                    &beta_,
                    top_desc_for_bias_,
                    Y->template mutable_data<T_Y>()));
        }

        return true;
    }
//TODO : enable fp16 support.
    bool MIOPENConvOp::RunOnDevice() {

        if (Input(0).IsType<float>()) {
            return DoRunWithType<float,      // X
                    float,      // W
                    float,      // B
                    float,      // Math
                    float>();   // Y
        } else {
            LOG(FATAL) << "Only float (32bit) is supported by "
                       << "miopen convolution, but input " << debug_def().input(0)
                       << " has [" << Input(0).meta().name() << "]";
        }
        return true;
    }

    template<typename T_X, typename T_DY, typename T_W, typename T_B,
            typename MATH,
            typename T_DX, typename T_DW, typename T_DB>
    bool MIOPENConvGradientOp::DoRunWithType() {
        auto &X = Input(INPUT);
        auto &Weight = Input(FILTER);
        auto &dY = Input(OUTPUT_GRAD);
        auto *dW = Output(FILTER_GRAD);
        auto *dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
        dX->ResizeLike(X);

        CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
        CAFFE_ENFORCE(Weight.ndim() >= 3 && Weight.ndim() <= 5);

        const int M = Weight.dim32(0);
        int N = 0, C = 0, H = 0, W = 0, D = 0, N_out = 0, C_out = 0, H_out = 0, W_out = 0, D_out = 0;

        N = X.dim32(0);
        C = X.dim32(1);
        H = X.dim32(2);
        W = X.ndim() > 3 ? X.dim32(3) : 1;
        D = X.ndim() > 4 ? X.dim32(4) : 1;

        N_out = dY.dim32(0);
        C_out = dY.dim32(1);
        H_out = dY.dim32(2);
        W_out = dY.ndim() > 3 ? dY.dim32(3) : 1;
        D_out = dY.ndim() > 4 ? dY.dim32(4) : 1;

        //////////// BWD DATA ////////////////////////////////////////

        while(!bestDataAlgoFound_)
        {

            MIOPEN_ENFORCE(miopenFindConvolutionBackwardDataAlgorithm
                                   (miopen_wrapper_.inline_miopen_handle(),
                                    top_desc_,
                                    dY.template data<T_DY>(),
                                    weight_desc_,
                                    Weight.template data<T_W>(),
                                    conv_desc_,
                                    bottom_desc_,
                                    dX->template data<T_DX>(),
                                    requestAlgoCount_,
                                    &returnedAlgoCount_,
                                    &perf_,
                                    bwdDataWs,// state->workspace().get(fwdConvWsSize_);
                                    bwdDataWsSize_,
                                    exhaustive_search_));

            bestDataAlgoFound_ = true;
        }


        MIOPEN_ENFORCE(miopenConvolutionBackwardData(miopen_wrapper_.inline_miopen_handle(),
                                                     &alpha_,
                                                     top_desc_,
                                                     dY.template data<T_DY>(),
                                                     weight_desc_,
                                                     Weight.template data<T_W>(),
                                                     conv_desc_,
                                                     perf_.bwd_data_algo,
                                                     &beta_,
                                                     bottom_desc_,
                                                     dX->template mutable_data<T_DX>(),
                                                     bwdDataWs,  // state->workspace().get(bwd_conv_ws_size);
                                                     bwdDataWsSize_));


        //////////////////////////////   BWD WEIGHT //////////////////////

        while(!bestWeightAlgoFound_)
        {

            MIOPEN_ENFORCE(miopenFindConvolutionBackwardWeightsAlgorithm(miopen_wrapper_.inline_miopen_handle(),
                                                                         top_desc_,
                                                                         dY.template data<T_DY>(),
                                                                         bottom_desc_,
                                                                         X.template data<T_X>(),
                                                                         conv_desc_,
                                                                         weight_desc_,
                                                                         dW->template mutable_data<T_DW>(),
                                                                         requestAlgoCount_,
                                                                         &returnedAlgoCount_,
                                                                         &perf_,
                                                                         bwdWeightWs,// state->workspace().get(bwd_conv_ws_size);
                                                                         bwdWeightWsSize_,
                                                                         exhaustive_search_));

            bestWeightAlgoFound_ = true;
        }
        MIOPEN_ENFORCE(miopenConvolutionBackwardWeights(miopen_wrapper_.inline_miopen_handle(),
                                                        &alpha_,
                                                        top_desc_,
                                                        dY.template data<T_DY>(),
                                                        bottom_desc_,
                                                        X.template data<T_X>(),
                                                        conv_desc_,
                                                        perf_.bwd_weights_algo,
                                                        &beta_,
                                                        weight_desc_,
                                                        dW->template mutable_data<T_DW>(),
                                                        bwdWeightWs,  // state->workspace().get(bwd_conv_ws_size);
                                                        bwdWeightWsSize_));

        ////////////////////////////////////// BIAS ///////////////////////////
        if (!no_bias_) {
            auto *dbias = Output(BIAS_OR_INPUT_GRAD);
            dbias->Resize(M);
            MIOPEN_ENFORCE(miopenConvolutionBackwardBias(
                    miopen_wrapper_.inline_miopen_handle(),
                    &alpha_,
                    top_desc_for_bias_,
                    dY.template data<T_DY>(),
                    &beta_,
                    bias_desc_,
                    dbias->template mutable_data<T_DB>()));
        }

        return true;
    }

    bool MIOPENConvGradientOp::RunOnDevice() {
        if (Input(0).IsType<float>()) {
            return DoRunWithType<float,    //  X
                    float,    // dY
                    float,    //  W
                    float,    //  b
                    float,    // Math
                    float,    // dX
                    float,    // dW
                    float>(); // db
        } else {
            LOG(FATAL) << "Unsupported input types";
        }
        return true;
    }

    REGISTER_MIOPEN_OPERATOR(Conv, MIOPENConvOp);
    REGISTER_MIOPEN_OPERATOR(ConvGradient, MIOPENConvGradientOp);

    REGISTER_MIOPEN_OPERATOR(Conv1D, MIOPENConvOp);
    REGISTER_MIOPEN_OPERATOR(Conv1DGradient, MIOPENConvGradientOp);

    REGISTER_MIOPEN_OPERATOR(Conv2D, MIOPENConvOp);
    REGISTER_MIOPEN_OPERATOR(Conv2DGradient, MIOPENConvGradientOp);

}  // namespace caffe2
