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
#define __STDC_WANT_LIB_EXT1__ 1

#include "nnrt_op_builder.h"

#include <cstring>

#include "neural_network_runtime.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/context_util.h"

#include "nnrt_utils.h"

namespace tflite {
namespace delegate {
namespace nnrt {
constexpr int32_t SCALAR_TENSOR_RANK = 1;
constexpr int32_t ADDZEROS_BIAS_INDEX = -1;
constexpr int32_t UNSPECIFIED_DIMENSION_VALUE = -1;
const std::vector<int32_t> DEPTHWISE_TRANSPOSE_AXISS = { 3, 1, 2, 0 };

NnrtOpBuilder::NnrtOpBuilder(const NnrtApi* nnrt, NnrtOpBuilderArgs& opBuilderArgs)
    : m_nnrt(nnrt),
      m_context(opBuilderArgs.context),
      m_pTensorMapping(opBuilderArgs.pTensorMapping),
      m_nnModel(opBuilderArgs.nnModel),
      m_allowDynamicDimensions(opBuilderArgs.delegateOptions.allowDynamicDimensions)
{
    // Map Op func pointer
    MapBuiltinCodeToFunc();

    // Get model inputs
    for (int32_t i : TfLiteIntArrayView(opBuilderArgs.inputTensors)) {
        // Constant tensors are not NNRT inputs.
        if (i != kTfLiteOptionalTensor && opBuilderArgs.context->tensors[i].allocation_type != kTfLiteMmapRo) {
            m_inputs.emplace_back(i);
        }
    }
}

TfLiteStatus NnrtOpBuilder::AddZerosBias(const NnrtOpMappingArgs& mappingArgs, int32_t inputId, int32_t filterId,
    int32_t channelNum)
{
    int32_t biasIndex = ADDZEROS_BIAS_INDEX;
    mappingArgs.context->AddTensors(mappingArgs.context, 1, &biasIndex);
    TfLiteTensor* biasTensor = &mappingArgs.context->tensors[biasIndex];
    const auto inputType = mappingArgs.context->tensors[inputId].type;

    if (inputType == kTfLiteFloat32) {
        biasTensor->type = kTfLiteFloat32;
    } else {
        biasTensor->type = kTfLiteInt32;
    }

    // Create an array with a required bias shape and resize the bias tensor.
    TfLiteIntArray* biasShape = TfLiteIntArrayCreate(1); // 1-dimension
    biasShape->data[0] = channelNum;
    biasTensor->allocation_type = kTfLiteDynamic;
    mappingArgs.context->ResizeTensor(mappingArgs.context, biasTensor, biasShape);

    // Set tensor's values to zeroes and add it using AddVector*, so that the values are copied to NNRT.
#ifdef __STDC_LIB_EXT1__
    if (inputType == kTfLiteFloat32) {
        memset_s(biasTensor->data.f, biasTensor->bytes, 0, channelNum * sizeof(float));
        TF_LITE_ENSURE_STATUS(mappingArgs.builder->AddVectorFloat32Tensor(biasTensor->data.f, channelNum,
            OH_NN_TENSOR));
    } else {
        memset_s(biasTensor->data.i32, biasTensor->bytes, 0, channelNum * sizeof(int32_t));
        const TfLiteTensor& inputTensor = mappingArgs.context->tensors[inputId];
        const TfLiteTensor& filterTensor = mappingArgs.context->tensors[filterId];

        // NNRT requires bias scale to be a product of an input scale and a filter scale.
        biasTensor->params.scale = inputTensor.params.scale * filterTensor.params.scale;
        TF_LITE_ENSURE_STATUS(mappingArgs.builder->AddVectorInt32Tensor(biasTensor->data.i32, channelNum,
            OH_NN_TENSOR));
    }
#endif

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddBasicComputeParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    if (builtinCode == kTfLiteBuiltinAdd) {
        auto builtin = reinterpret_cast<TfLiteAddParams*>(mappingArgs.node->builtin_data);
        TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLiteAddParams>(mappingArgs, builtin, builtinCode,
            OH_NN_ADD_ACTIVATIONTYPE));
    } else if (builtinCode == kTfLiteBuiltinMul) {
        auto builtin = reinterpret_cast<TfLiteMulParams*>(mappingArgs.node->builtin_data);
        TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLiteMulParams>(mappingArgs, builtin, builtinCode,
            OH_NN_MUL_ACTIVATION_TYPE));
    } else if (builtinCode == kTfLiteBuiltinSub) {
        auto builtin = reinterpret_cast<TfLiteSubParams*>(mappingArgs.node->builtin_data);
        TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLiteSubParams>(mappingArgs, builtin, builtinCode,
            OH_NN_SUB_ACTIVATIONTYPE));
    } else {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] unsupportted basic compute type %d.", builtinCode);
        return kTfLiteError;
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddAvgPoolingParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLitePoolParams*>(mappingArgs.node->builtin_data);
    std::vector<int64_t> kernel = { static_cast<int64_t>(builtin->filter_height),
        static_cast<int64_t>(builtin->filter_width) };
    std::vector<int64_t> stride = { static_cast<int64_t>(builtin->stride_height),
        static_cast<int64_t>(builtin->stride_width) };

    mappingArgs.builder->AddVectorInt64Tensor(kernel.data(), kernel.size(), OH_NN_AVG_POOL_KERNEL_SIZE);
    mappingArgs.builder->AddVectorInt64Tensor(stride.data(), stride.size(), OH_NN_AVG_POOL_STRIDE);
    TF_LITE_ENSURE_STATUS(AddPadParamsInOperator<TfLitePoolParams>(mappingArgs, builtin, builtinCode,
        OH_NN_AVG_POOL_PAD_MODE));
    TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLitePoolParams>(mappingArgs, builtin, builtinCode,
        OH_NN_AVG_POOL_ACTIVATION_TYPE));

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddMaxPoolingParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLitePoolParams*>(mappingArgs.node->builtin_data);
    std::vector<int64_t> kernel = { static_cast<int64_t>(builtin->filter_height),
        static_cast<int64_t>(builtin->filter_width) };
    std::vector<int64_t> stride = { static_cast<int64_t>(builtin->stride_height),
        static_cast<int64_t>(builtin->stride_width) };

    mappingArgs.builder->AddVectorInt64Tensor(kernel.data(), kernel.size(), OH_NN_MAX_POOL_KERNEL_SIZE);
    mappingArgs.builder->AddVectorInt64Tensor(stride.data(), stride.size(), OH_NN_MAX_POOL_STRIDE);
    TF_LITE_ENSURE_STATUS(AddPadParamsInOperator<TfLitePoolParams>(mappingArgs, builtin, builtinCode,
        OH_NN_MAX_POOL_PAD_MODE));
    TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLitePoolParams>(mappingArgs, builtin, builtinCode,
        OH_NN_MAX_POOL_ACTIVATION_TYPE));

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddFullConnectedParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    // IF bias is not presented, bias input index will be -1.
    const bool isBiasPresent =
        (mappingArgs.node->inputs->size == 3) && (mappingArgs.node->inputs->data[2] != kTfLiteOptionalTensor);

    if (!isBiasPresent) {
        const int32_t inputTensorId = mappingArgs.node->inputs->data[0];                      // kInputTensor
        const int32_t filterTensorId = mappingArgs.node->inputs->data[1];                     // kWeightsTensor
        const int32_t numUnits = mappingArgs.context->tensors[filterTensorId].dims->data[0];  // bias channel num
        TF_LITE_ENSURE_STATUS(AddZerosBias(mappingArgs, inputTensorId, filterTensorId, numUnits));
    }

    auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(mappingArgs.node->builtin_data);
    TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLiteFullyConnectedParams>(mappingArgs, builtin, builtinCode,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE));

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddConcatenationParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(mappingArgs.node->builtin_data);
    const int64_t axis = static_cast<int64_t>(builtin->axis);
    mappingArgs.builder->AddScalarInt64Tensor(axis, OH_NN_CONCAT_AXIS);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddSoftmaxParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(mappingArgs.node->builtin_data);
    const int64_t axis = static_cast<int64_t>(builtin->beta);
    mappingArgs.builder->AddScalarInt64Tensor(axis, OH_NN_SOFTMAX_AXIS);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddQuantizeParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    OH_NN_DataType nnType {OH_NN_FLOAT32};

    int32_t inputIndex = mappingArgs.node->inputs->data[0];
    m_pTensorMapping->ConvertType(m_context, inputIndex, 0, nnType);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(nnType), OH_NN_QUANT_DTYPE_CAST_SRC_T);

    int32_t outputIndex = mappingArgs.node->outputs->data[0];
    m_pTensorMapping->ConvertType(m_context, outputIndex, 0, nnType);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(nnType), OH_NN_QUANT_DTYPE_CAST_DST_T);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddPackParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLitePackParams*>(mappingArgs.node->builtin_data);
    const int64_t axis = static_cast<int64_t>(builtin->axis);
    mappingArgs.builder->AddScalarInt64Tensor(axis, OH_NN_STACK_AXIS);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddPadParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    float padValue = 0.0;
    mappingArgs.builder->AddScalarFloat32Tensor(padValue, OH_NN_PAD_CONSTANT_VALUE);
    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddReduceMeanParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLiteReducerParams*>(mappingArgs.node->builtin_data);
    const int32_t keepDims = (builtin->keep_dims);
    mappingArgs.builder->AddScalarBoolTensor(keepDims, OH_NN_REDUCE_MEAN_KEEP_DIMS);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddStridedSliceParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(mappingArgs.node->builtin_data);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(builtin->begin_mask),
        OH_NN_STRIDED_SLICE_BEGIN_MASK);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(builtin->end_mask),
        OH_NN_STRIDED_SLICE_END_MASK);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(builtin->ellipsis_mask),
        OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(builtin->new_axis_mask),
        OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    mappingArgs.builder->AddScalarInt64Tensor(static_cast<int64_t>(builtin->shrink_axis_mask),
        OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddReshapeParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    if (mappingArgs.node->inputs->size == 1) {
        auto builtin = reinterpret_cast<TfLiteReshapeParams*>(mappingArgs.node->builtin_data);
        int32_t numDimensions = builtin->num_dimensions;
        std::vector<int32_t> outputShape(numDimensions);
        for (int32_t i = 0; i < numDimensions; ++i) {
            outputShape[i] = builtin->shape[i];
        }
        mappingArgs.builder->AddVectorInt32Tensor(outputShape.data(), outputShape.size(), OH_NN_TENSOR);
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddConv2DParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLiteConvParams*>(mappingArgs.node->builtin_data);
    std::vector<int64_t> stride = { static_cast<int64_t>(builtin->stride_height),
        static_cast<int64_t>(builtin->stride_width) };
    std::vector<int64_t> dilation = { static_cast<int64_t>(builtin->dilation_height_factor),
        static_cast<int64_t>(builtin->dilation_width_factor) };
    int64_t groupNum = 1;

    mappingArgs.builder->AddVectorInt64Tensor(stride.data(), stride.size(), OH_NN_CONV2D_STRIDES);
    mappingArgs.builder->AddVectorInt64Tensor(dilation.data(), dilation.size(), OH_NN_CONV2D_DILATION);

    TF_LITE_ENSURE_STATUS(AddPadParamsInOperator<TfLiteConvParams>(mappingArgs, builtin, builtinCode,
        OH_NN_CONV2D_PAD_MODE));
    mappingArgs.builder->AddScalarInt64Tensor(groupNum, OH_NN_CONV2D_GROUP);
    TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLiteConvParams>(mappingArgs, builtin, builtinCode,
       OH_NN_CONV2D_ACTIVATION_TYPE));

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddDepthwiseConv2DParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(mappingArgs.node->builtin_data);
    std::vector<int64_t> stride = { static_cast<int64_t>(builtin->stride_height),
        static_cast<int64_t>(builtin->stride_width) };
    std::vector<int64_t> dilation = { static_cast<int64_t>(builtin->dilation_height_factor),
        static_cast<int64_t>(builtin->dilation_width_factor) };
    TF_LITE_ENSURE_STATUS(mappingArgs.builder->AddVectorInt64Tensor(stride.data(), stride.size(),
        OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES));
    TF_LITE_ENSURE_STATUS(mappingArgs.builder->AddVectorInt64Tensor(dilation.data(), dilation.size(),
        OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION));
    TF_LITE_ENSURE_STATUS(AddPadParamsInOperator<TfLiteDepthwiseConvParams>(mappingArgs, builtin, builtinCode,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD));
    TF_LITE_ENSURE_STATUS(AddActivateParamsInOperator<TfLiteDepthwiseConvParams>(mappingArgs, builtin, builtinCode,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE));
    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::FinalizeAddOperation(OH_NN_OperationType type, int32_t liteNodeIndex)
{
    // Actually add a NN API Operation
    OH_NN_UInt32Array inputIndices;
    OH_NN_UInt32Array outputIndices;
    OH_NN_UInt32Array paramIndices;
    inputIndices.data = m_augmentedInputs.data();
    inputIndices.size = static_cast<uint32_t>(m_augmentedInputs.size());
    outputIndices.data = m_augmentedOutputs.data();
    outputIndices.size = static_cast<uint32_t>(m_augmentedOutputs.size());
    paramIndices.size = static_cast<uint32_t>(m_augmentedParams.size());

    paramIndices.data = (m_augmentedParams.size() == 0) ? nullptr : m_augmentedParams.data();

    RETURN_TFLITE_ERROR_IF_NN_ERROR(m_nnrt->OH_NNModel_AddOperation(m_nnModel,
        type, &paramIndices, &inputIndices, &outputIndices), "adding operation");

    m_augmentedInputs.clear();
    m_augmentedOutputs.clear();
    m_augmentedParams.clear();

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddTensor(int32_t tensorIndex, int32_t builtinCode, std::vector<uint32_t>& indices,
    int32_t tensorFlags)
{
    int32_t nnTensorIndex = m_pTensorMapping->LiteIndexToNn(tensorIndex);
    if (nnTensorIndex != INVALID_INDEX) {
        indices.emplace_back(nnTensorIndex);
        return kTfLiteOk;
    }

    // Parameters needed for new type.
    TfLiteTensor* tensor = &(m_context->tensors[tensorIndex]);
    if (kTfLiteNoType == tensor->type) {
        indices.emplace_back(INVALID_INDEX);
        return kTfLiteOk;
    }

    TF_LITE_ENSURE_STATUS(AddTensor(tensorIndex, builtinCode, tensorFlags, nnTensorIndex));

    indices.emplace_back(nnTensorIndex);

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddTensor(int32_t tensorIndex, int32_t builtinCode, int32_t tensorFlags,
    int32_t& nnTensorIndex)
{
    TfLiteTensor* tensor = &(m_context->tensors[tensorIndex]);
    const bool scalarAsTensor = tensorFlags & NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    OH_NN_Tensor nnTensor;
    OH_NN_QuantParam nnQuantParam;
    std::vector<int32_t> weightDims;
    void* tensorData = tensor->data.data;
    std::vector<int8_t> depthwiseTensorData;
    TF_LITE_ENSURE_STATUS(ConstructNNTensor(tensorIndex, builtinCode, scalarAsTensor, nnQuantParam, nnTensor));

    // For depth-wise conv operator, we should transpose weight tensor to adapt NN tensor format.
    if ((builtinCode == kTfLiteBuiltinDepthwiseConv2d) && (tensor->allocation_type == kTfLiteMmapRo) &&
        (nnTensor.dimensionCount == DEPTHWISE_WEIGHT_DIMENSION_COUNT)) {
        size_t typeBytes = 0;
        int64_t tensorSize = 0;
        TF_LITE_ENSURE_STATUS(GetSizeOfType(m_context, tensor->type, &typeBytes));
        TF_LITE_ENSURE_STATUS(GetTensorSize(m_context, nnTensor.dimensions, nnTensor.dimensionCount, tensorSize));

        depthwiseTensorData.assign(tensorSize * typeBytes, 0);
        TfLiteStatus retCode = TransposeDepthwiseTensor(tensorIndex, nnTensor, weightDims, depthwiseTensorData);
        if (retCode != kTfLiteOk) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] Fail to transpose depthwise tensor.");
            return kTfLiteError;
        }
        tensorData = static_cast<void*>(depthwiseTensorData.data());
    }

    int32_t nnRet = m_nnrt->OH_NNModel_AddTensor(m_nnModel, &nnTensor);
    if (nnRet != OH_NN_SUCCESS) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] Fail to add nnTensor to NN model.");
        return kTfLiteError;
    }

    // Allocate a new tensor index
    nnTensorIndex = m_pTensorMapping->AddNewNnTensorIndex(tensorIndex);
    if (tensor->allocation_type == kTfLiteMmapRo) {
        nnRet = m_nnrt->OH_NNModel_SetTensorData(m_nnModel, nnTensorIndex,
            tensorData, tensor->bytes);
        if (nnRet != OH_NN_SUCCESS) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] Fail to setting new nnTensor value.");
            return kTfLiteError;
        }
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::TransposeDepthwiseTensor(int32_t tensorIndex, OH_NN_Tensor& nnTensor,
    std::vector<int32_t>& weightDims, std::vector<int8_t>& tensorData)
{
    const int32_t* tensorDims = nnTensor.dimensions;
    uint32_t tensorRank = nnTensor.dimensionCount;

    // For Depth-wise Convolution, NNRT choose to Transpose dimension with [3, 1, 2, 0]
    TF_LITE_ENSURE_STATUS(TransposeDims(m_context, tensorDims, tensorRank, DEPTHWISE_TRANSPOSE_AXISS, weightDims));
    nnTensor.dimensions = weightDims.data();

    TfLiteTensor* tensor = &(m_context->tensors[tensorIndex]);
    if (tensor->type == kTfLiteFloat32) {
        TF_LITE_ENSURE_STATUS(
            TransposeTensor(m_context, tensorIndex, tensorDims, reinterpret_cast<float*>(tensorData.data())));
    } else if (tensor->type == kTfLiteInt32) {
        TF_LITE_ENSURE_STATUS(
            TransposeTensor(m_context, tensorIndex, tensorDims, reinterpret_cast<int32_t*>(tensorData.data())));
    } else if (tensor->type == kTfLiteInt8) {
        TF_LITE_ENSURE_STATUS(
            TransposeTensor(m_context, tensorIndex, tensorDims, reinterpret_cast<int8_t*>(tensorData.data())));
    } else if (tensor->type == kTfLiteUInt8) {
        TF_LITE_ENSURE_STATUS(
            TransposeTensor(m_context, tensorIndex, tensorDims, reinterpret_cast<uint8_t*>(tensorData.data())));
    } else {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] unsupportted weight tensor type %d.", tensor->type);
        return kTfLiteError;
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::ConstructNNTensor(int32_t tensorIndex, int32_t builtinCode, int32_t tensorFlags,
    OH_NN_QuantParam& nnQuantParam, OH_NN_Tensor& nnTensor)
{
    OH_NN_DataType nnType {OH_NN_UNKNOWN};
    TF_LITE_ENSURE_STATUS(m_pTensorMapping->ConvertType(m_context, tensorIndex, tensorFlags, nnType));
    TF_LITE_ENSURE_STATUS(m_pTensorMapping->ConvertQuantParams(m_context, tensorIndex, nnQuantParam));

    TfLiteTensor* tensor = &(m_context->tensors[tensorIndex]);
    uint32_t tensorRank = static_cast<uint32_t>(tensor->dims->size);
    m_dimsUnspecified.assign(tensorRank, UNSPECIFIED_DIMENSION_VALUE);

    int32_t* tensorDims = (m_allowDynamicDimensions && (tensor->allocation_type != kTfLiteMmapRo) &&
        std::find(m_inputs.begin(), m_inputs.end(), tensorIndex) != m_inputs.end()) ?
        reinterpret_cast<int32_t*>(m_dimsUnspecified.data()) :
        tensor->dims->data;

    const bool scalarAsTensor = tensorFlags & NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    if (scalarAsTensor && tensorRank == 0) {
        tensorRank = SCALAR_TENSOR_RANK; // Use rank 1, shape {1} nnTensor for TFLite scalar tensors.
        tensorDims = const_cast<int32_t*>(&SCALAR_TENSOR_RANK);
    }

    if (tensorRank == 0) {
        // if the tensorRank is 0, the dimension ptr must be nullptr.
        tensorDims = nullptr;
    }

    nnTensor.dataType = nnType;
    nnTensor.dimensionCount = tensorRank;
    nnTensor.dimensions = tensorDims;
    nnTensor.quantParam = nnQuantParam.quantCount ? &nnQuantParam : nullptr;
    nnTensor.type = OH_NN_TENSOR;

    return kTfLiteOk;
}

TfLiteStatus NnrtOpBuilder::AddOpFuncParams(const NnrtOpMappingArgs& mappingArgs, int32_t builtinCode)
{
    if (!m_keyToOpFunc.count(builtinCode)) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-OPBUILDER] unsupportted Op builtinCode : %d.", builtinCode);
        return kTfLiteError;
    }

    OpFuncPtr pfunc = m_keyToOpFunc[builtinCode];
    return (this->*pfunc)(mappingArgs, builtinCode);
}

TfLiteStatus NnrtOpBuilder::MapBuiltinCodeToFunc()
{
    m_keyToOpFunc[kTfLiteBuiltinAdd] = &NnrtOpBuilder::AddBasicComputeParams;
    m_keyToOpFunc[kTfLiteBuiltinAveragePool2d] = &NnrtOpBuilder::AddAvgPoolingParams;
    m_keyToOpFunc[kTfLiteBuiltinConcatenation] = &NnrtOpBuilder::AddConcatenationParams;
    m_keyToOpFunc[kTfLiteBuiltinConv2d] = &NnrtOpBuilder::AddConv2DParams;
    m_keyToOpFunc[kTfLiteBuiltinDepthwiseConv2d] = &NnrtOpBuilder::AddDepthwiseConv2DParams;
    m_keyToOpFunc[kTfLiteBuiltinDequantize] = &NnrtOpBuilder::AddQuantizeParams;
    m_keyToOpFunc[kTfLiteBuiltinFullyConnected] = &NnrtOpBuilder::AddFullConnectedParams;
    m_keyToOpFunc[kTfLiteBuiltinMaxPool2d] = &NnrtOpBuilder::AddMaxPoolingParams;
    m_keyToOpFunc[kTfLiteBuiltinMul] = &NnrtOpBuilder::AddBasicComputeParams;
    m_keyToOpFunc[kTfLiteBuiltinSub] = &NnrtOpBuilder::AddBasicComputeParams;
    m_keyToOpFunc[kTfLiteBuiltinReshape] = &NnrtOpBuilder::AddReshapeParams;
    m_keyToOpFunc[kTfLiteBuiltinSoftmax] = &NnrtOpBuilder::AddSoftmaxParams;
    m_keyToOpFunc[kTfLiteBuiltinStridedSlice] = &NnrtOpBuilder::AddStridedSliceParams;
    m_keyToOpFunc[kTfLiteBuiltinPack] = &NnrtOpBuilder::AddPackParams;
    m_keyToOpFunc[kTfLiteBuiltinPad] = &NnrtOpBuilder::AddPadParams;
    m_keyToOpFunc[kTfLiteBuiltinMean] = &NnrtOpBuilder::AddReduceMeanParams;
    m_keyToOpFunc[kTfLiteBuiltinQuantize] = &NnrtOpBuilder::AddQuantizeParams;
    m_keyToOpFunc[kTfLiteBuiltinHardSwish] = &NnrtOpBuilder::AddDefaultOpParams;
    m_keyToOpFunc[kTfLiteBuiltinShape] = &NnrtOpBuilder::AddDefaultOpParams;
    m_keyToOpFunc[kTfLiteBuiltinLogistic] = &NnrtOpBuilder::AddDefaultOpParams;

    return kTfLiteOk;
}
} // namespace nnrt
} // namespace delegate
} // namespace tflite