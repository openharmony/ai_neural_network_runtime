/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
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

#ifndef TENSORFLOW_LITE_DELEGATES_NNRT_OP_BUILDER_H
#define TENSORFLOW_LITE_DELEGATES_NNRT_OP_BUILDER_H

#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"

#include "../nnrt/nnrt_implementation.h"
#include "tensor_mapping.h"

namespace tflite {
namespace delegate {
namespace nnrt {
constexpr int32_t PADDING_SAME = 0;
constexpr int32_t PADDING_VALID = 1;

// NN API Operator Builder
class NnrtOpBuilder;

// The kernel that represents the node sub set of TF Lite being run on NN API.
struct NnrtOpMappingArgs {
    TfLiteContext* context {nullptr};
    NnrtOpBuilder* builder {nullptr};
    TfLiteNode* node {nullptr};
    int32_t nodeIndex {-1};
};

struct NnrtOpBuilderArgs {
    TfLiteContext* context {nullptr};
    OH_NNModel* nnModel {nullptr};
    TfLiteIntArray* inputTensors {nullptr};
    TensorMapping* pTensorMapping {nullptr};
    NnrtDelegate::Options delegateOptions;
};

// Abstract builder for building an op in the NN API graph. This handles
// the disparity between TFLite and NN API nnTensor types. NN API has singular
// nnTensors for both tensors and parameters, and TFLite separates the two.
class NnrtOpBuilder {
public:
    NnrtOpBuilder(const NnrtApi* nnrt, NnrtOpBuilderArgs& opBuilderArgs);
    ~NnrtOpBuilder() = default;

    // Add scalar nnTensor, the datatypes involved are bool, Int32, Int8, Int64, Float32
    TfLiteStatus AddScalarBoolTensor(bool value, OH_NN_TensorType nnTensorType)
    {
        return AddScalarTensor<bool>(value, OH_NN_BOOL, nnTensorType);
    }
    TfLiteStatus AddScalarInt32Tensor(int32_t value, OH_NN_TensorType nnTensorType)
    {
        return AddScalarTensor<int32_t>(value, OH_NN_INT32, nnTensorType);
    }
    TfLiteStatus AddScalarInt8Tensor(int32_t value, OH_NN_TensorType nnTensorType)
    {
        return AddScalarTensor<int8_t>(value, OH_NN_INT8, nnTensorType);
    }
    TfLiteStatus AddScalarInt64Tensor(int64_t value, OH_NN_TensorType nnTensorType)
    {
        return AddScalarTensor<int64_t>(value, OH_NN_INT64, nnTensorType);
    }
    TfLiteStatus AddScalarFloat32Tensor(float value, OH_NN_TensorType nnTensorType)
    {
        return AddScalarTensor<float>(value, OH_NN_FLOAT32, nnTensorType);
    }

    // Add vector nnTensor, the datatypes involved are Int32, Int64, Int16, Int8, Float32
    TfLiteStatus AddVectorInt32Tensor(const int32_t* values, uint32_t numValues, OH_NN_TensorType nnTensorType)
    {
        return AddVectorTensor<int32_t>(values, numValues, OH_NN_UINT32, nnTensorType);
    }
    TfLiteStatus AddVectorInt64Tensor(const int64_t* values, uint32_t numValues, OH_NN_TensorType nnTensorType)
    {
        return AddVectorTensor<int64_t>(values, numValues, OH_NN_INT64, nnTensorType);
    }
    TfLiteStatus AddVectorFloat32Tensor(const float* values, uint32_t numValues, OH_NN_TensorType nnTensorType)
    {
        return AddVectorTensor<float>(values, numValues, OH_NN_FLOAT32, nnTensorType);
    }

    // Add input tensor
    TfLiteStatus AddTensorInput(int32_t tensorIndex, int32_t builtinCode, int32_t tensorFlags = 0)
    {
        return AddTensor(tensorIndex, builtinCode, m_augmentedInputs, tensorFlags);
    }
    // Add output tensor
    TfLiteStatus AddTensorOutput(int32_t tensorIndex, int32_t builtinCode, int32_t tensorFlags = 0)
    {
        return AddTensor(tensorIndex, builtinCode, m_augmentedOutputs, tensorFlags);
    }

    // Finish emitting the op (of type `type`) into the NN API.
    TfLiteStatus FinalizeAddOperation(OH_NN_OperationType type, int32_t liteNodeIndex);

    void ClearInputOuputLists()
    {
        m_augmentedInputs.clear();
        m_augmentedOutputs.clear();
        m_augmentedParams.clear();
    }

    TfLiteStatus AddOpFuncParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus MapBuiltinCodeToFunc();

private:
    template<typename T>
    TfLiteStatus AddScalarTensor(T value, OH_NN_DataType nnType, OH_NN_TensorType nnTensorType)
    {
        OH_NN_Tensor tensor {
            .dataType = nnType,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .quantParam = nullptr,
            .type = nnTensorType,
        };

        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            m_nnrt->OH_NNModel_AddTensor(m_nnModel, &tensor), "adding nnTensor");

        const int32_t nnIndex = m_pTensorMapping->AddNewNonTensorTensor();
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            m_nnrt->OH_NNModel_SetTensorData(m_nnModel, nnIndex, &value, sizeof(value)),
            "setting new nnTensor value");
        m_augmentedParams.emplace_back(nnIndex);

        return kTfLiteOk;
    }

    template<typename T>
    TfLiteStatus AddVectorTensor(const T* values, int32_t numValues, OH_NN_DataType nnType,
        OH_NN_TensorType nnTensorType)
    {
        if (values == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                "[NNRT-OPBUILDER] The variable of values is nullptr when adding vector to operator.");
            return kTfLiteError;
        }
        uint32_t numBits = 8;
        double doubleScale = 0.f;
        int32_t zeroPoint = 0;
        OH_NN_QuantParam quantParam = {
            .quantCount = 1,
            .numBits = &numBits,
            .scale = &doubleScale,
            .zeroPoint = &zeroPoint
        };

        OH_NN_Tensor tensor {
            .dataType = nnType,
            .dimensionCount = 1, // For 1-dim vector, dimensionCount is one.
            .dimensions = &numValues,
            .quantParam = &quantParam,
            .type = nnTensorType,
        };

        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            m_nnrt->OH_NNModel_AddTensor(m_nnModel, &tensor), "adding nnTensor");
        const int32_t nnIndex = m_pTensorMapping->AddNewNonTensorTensor();
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            m_nnrt->OH_NNModel_SetTensorData(m_nnModel, nnIndex, values, sizeof(*(values)) * numValues),
            "settings new nnTensor value");
        m_augmentedParams.emplace_back(nnIndex);

        return kTfLiteOk;
    }

    template<typename T>
    TfLiteStatus AddActivateParamsInOperator(const NnrtOpMappingArgs& mappingArgs, T* builtinParams,
        int32_t builtinCode, OH_NN_TensorType nnTensorType)
    {
        if (builtinParams == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                "[NNRT-OPBUILDER] The builtin params is nullptr when adding activate params to operator.");
            return kTfLiteError;
        }

        if ((builtinParams->activation >= 0) &&
            (builtinParams->activation < ACTIVATE_FUSE_TYPE_LIST.size()) &&
            (ACTIVATE_FUSE_TYPE_LIST[builtinParams->activation] != OH_NN_FUSE_UNSUPPORTED)) {
            mappingArgs.builder->AddScalarInt8Tensor(ACTIVATE_FUSE_TYPE_LIST[builtinParams->activation], nnTensorType);
        } else {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                "[NNRT-OPBUILDER] unsupportted fused activation type %d for OpType %d.",
                builtinParams->activation, builtinCode);
            return kTfLiteError;
        }

        return kTfLiteOk;
    }

    template<typename T>
    TfLiteStatus AddPadParamsInOperator(const NnrtOpMappingArgs& mappingArgs, T* builtinParams, int32_t builtinCode,
        OH_NN_TensorType nnTensorType)
    {
        if (builtinParams == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                "[NNRT-OPBUILDER] The builtin params is nullptr when adding pad params to operator.");
            return kTfLiteError;
        }

        int32_t padding = 0;
        if (builtinParams->padding == kTfLitePaddingUnknown) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] unknown padding mode for OpType %d.", builtinCode);
            return kTfLiteError;
        } else {
            padding = (builtinParams->padding == kTfLitePaddingSame) ? PADDING_SAME : PADDING_VALID;
        }
        mappingArgs.builder->AddScalarInt8Tensor(padding, nnTensorType);

        return kTfLiteOk;
    }

    // NNRT requires a bias tensor, so we allocate a new tensor to fill it with zeroes.
    // It is deleted with other tensors in the context during subgraph destructor call.
    TfLiteStatus AddZerosBias(const NnrtOpMappingArgs& mappingArgs, int32_t inputId, int32_t filterId,
        int32_t channelNum);

    TfLiteStatus AddBasicComputeParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddAvgPoolingParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddMaxPoolingParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddFullConnectedParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddConv2DParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddConcatenationParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddSoftmaxParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddQuantizeParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddPackParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddPadParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddReduceMeanParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddReshapeParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddStridedSliceParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddDepthwiseConv2DParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);

    TfLiteStatus AddDefaultOpParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
    {
        return kTfLiteOk;
    }

    // Adds a new NN API tensor that shadows the TF Lite tensor `tensorIndex`.
    // This restores the NN API tensor index corresponding to the created tensor.
    // If another caller previously created a NN API tensor for `tensorIndex`
    // then the existing one is restored.
    TfLiteStatus AddTensor(int32_t tensorIndex, int32_t builtinCode, std::vector<uint32_t>& indices,
        int32_t tensorFlags = 0);

    // Adds a new NN API nnTensor to NNModel.
    // If the builtinCode is kTfLiteBuiltinDepthwiseConv2d, the weight tensor will be transposed to CHWN format.
    TfLiteStatus AddTensor(int32_t tensorIndex, int32_t builtinCode, int32_t tensorFlags, int32_t& nnTensorIndex);

    // Transpose dimension for Depth-wise Convolution Operator.
    TfLiteStatus TransposeDepthwiseTensor(int32_t tensorIndex, OH_NN_Tensor& nnTensor, std::vector<int32_t>& destDims,
        std::vector<int8_t>& tensorData);

    // Get NN nnTensor from tensor
    TfLiteStatus ConstructNNTensor(int32_t tensorIndex, int32_t builtinCode, int32_t tensorFlags,
        OH_NN_QuantParam& nnQuantParam, OH_NN_Tensor& nnTensor);

private:
    // Access to NNRT.
    const NnrtApi* const m_nnrt;

    // TfLiteContext for error handling.
    TfLiteContext* const m_context;

    // Indices of all inputs of tflite subgraph.
    std::vector<int32_t> m_inputs;

    // Tracks relationship between indices.
    TensorMapping* const m_pTensorMapping;

    // The NNRT model.
    OH_NNModel* const m_nnModel;

    // Inputs and outputs for the current op. These are augmented in the sense
    // that NN API uses nnTensors for all arguments, not just tensors, unlike
    // TensorFlow Lite.
    std::vector<uint32_t> m_augmentedInputs;
    std::vector<uint32_t> m_augmentedParams;
    std::vector<uint32_t> m_augmentedOutputs;

    // Whether to allow dynamic batch size without re-compilation.
    bool m_allowDynamicDimensions;

    // the dynamic dimension information.
    std::vector<int32_t> m_dimsUnspecified;

    // key builtInCode to OpFunc Map
    using OpFuncPtr = TfLiteStatus(NnrtOpBuilder::*)(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode);
    std::map<int32_t, OpFuncPtr> m_keyToOpFunc;
};
} // namespace nnrt
} // namespace delegate
} // namespace tflite

#endif // TENSORFLOW_LITE_DELEGATES_NNRT_OP_BUILDER_H