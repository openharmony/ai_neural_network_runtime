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

#include "nnrt_delegate_kernel.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "neural_network_runtime.h"

namespace tflite {
namespace delegate {
namespace nnrt {
constexpr int32_t SCALAR_RANK = 1;

#define RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_COMPILE(code, callDesc)                                                   \
    do {                                                                                                              \
        if ( (code) != OH_NN_SUCCESS) {                                                                               \
            const auto errorDesc = NnrtErrorDescription((code));                                                      \
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "NN API returned error %s at line %d while %s.\n", errorDesc.c_str(),   \
                __LINE__, (callDesc));                                                                                \
            m_nnrt->OH_NNCompilation_Destroy(&m_pNnCompilation);                                                      \
            return kTfLiteError;                                                                                      \
        }                                                                                                             \
    } while (0)

bool NnrtDelegateKernel::Validate(const int32_t builtinCode)
{
    if (TFLITE_TYPE_TO_NNRT_TYPE.count(builtinCode) &&
        TFLITE_TYPE_TO_NNRT_TYPE.at(builtinCode) != OH_NN_UNSUPPORT_OPS) {
        return true;
    }

    return false;
}

TfLiteStatus NnrtDelegateKernel::Init(TfLiteContext* context, const TfLiteDelegateParams* params)
{
    TF_LITE_ENSURE_EQ(context, params != nullptr, true);

    if (m_initialised) {
        TFLITE_LOG_PROD(TFLITE_LOG_INFO,
            "[NNRT-DELEGATE_KERNEL] NnrtDelegateKernel has completed initialization, no need init again.");
        return kTfLiteOk;
    }

    for (auto nodeIndex : TfLiteIntArrayView(params->nodes_to_replace)) {
        m_delegateNodes.emplace_back(nodeIndex);
    }

    NnrtDelegate::Options delegateOptions;
    TF_LITE_ENSURE_STATUS(NnrtDelegate::GetOptions(params->delegate, delegateOptions));
    TF_LITE_ENSURE_STATUS(tflite::GetTargetDevice(context, params->delegate, m_nnrt, m_nnrtDevice));
    if (m_nnModel == nullptr) {
        m_nnModel = m_nnrt->OH_NNModel_Construct();
        if (m_nnModel == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE_KERNEL] Fail to create ONNRT model.");
            return kTfLiteError;
        }
        TF_LITE_ENSURE_STATUS(BuildGraph(context, delegateOptions, params->input_tensors, params->output_tensors));
    }

    m_initialised = true;

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node)
{
    TF_LITE_ENSURE_EQ(context, node != nullptr, true);

    if (!m_initialised) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-DELEGATE_KERNEL] NnrtDelegateKernel Prepare failed, not Init yet.");
        return kTfLiteError;
    }

    if (m_compiled) {
        return kTfLiteOk; // If model has completed compilation, no need compile again.
    }

    // Create OH_NNCompilation
    m_pNnCompilation = m_nnrt->OH_NNCompilation_Construct(m_nnModel);
    if (m_pNnCompilation == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE_KERNEL] Fail to create OH_NNCompilation instance.");
        return kTfLiteError;
    }

    NnrtDelegate::Options delegateOptions;
    TF_LITE_ENSURE_STATUS(NnrtDelegate::GetOptions(node->delegate, delegateOptions));

    TF_LITE_ENSURE_STATUS(SetNnOptions(context, delegateOptions));
    RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_COMPILE(m_nnrt->OH_NNCompilation_Build(m_pNnCompilation),
        "completing NNRT compilation");

    m_compiled = true;
    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::Invoke(TfLiteContext* context, TfLiteNode* node)
{
    if (!m_compiled) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-DELEGATE_KERNEL] NnrtDelegateKernel Invoke failed, not compile yet.");
        return kTfLiteError;
    }

    // Create OH_NNExecutor_Construct
    OH_NNExecutor* pNnExecution {nullptr};
    pNnExecution = m_nnrt->OH_NNExecutor_Construct(m_pNnCompilation);
    if (pNnExecution == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE_KERNEL] Fail to create OH_NNExecutor instance.");
        return kTfLiteError;
    }

    // Set the input tensor buffers.
    OH_NN_Tensor inputNnTensor;
    TF_LITE_ENSURE_STATUS(SetInputTensors(context, node, pNnExecution, inputNnTensor));

    // Get the output tensor buffers.
    TF_LITE_ENSURE_STATUS(SetOutputTensors(context, node, pNnExecution));

    // Invoke delegated subgraph.
    RETURN_TFLITE_ERROR_IF_NN_ERROR(m_nnrt->OH_NNExecutor_Run(pNnExecution), "running computation");

    m_nnrt->OH_NNExecutor_Destroy(&pNnExecution);
    pNnExecution = nullptr;
    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::Map(const int32_t builtinCode, const NnrtOpMappingArgs& mappingArgs,
    int32_t& nnOpType) const
{
    if (TFLITE_TYPE_TO_NNRT_TYPE.count(builtinCode) == 0) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-DELEGATE_KERNEL] Not support current TF-Lite Operator, builtCode: %d.", builtinCode);
        return kTfLiteError;
    }

    TfLiteStatus retValue = mappingArgs.builder->AddOpFuncParams(mappingArgs, builtinCode);
    if (retValue != kTfLiteOk) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE_KERNEL] Failed to add params to these operations.");
        return retValue;
    }
    nnOpType = TFLITE_TYPE_TO_NNRT_TYPE.at(builtinCode);

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::BuildGraph(TfLiteContext* context, const NnrtDelegate::Options& delegateOptions,
    const TfLiteIntArray* inputTensors, const TfLiteIntArray* outputTensors)
{
    if (context == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE_KERNEL] The context is nullptr when building the graph.");
        return kTfLiteError;
    }

    TF_LITE_ENSURE_EQ(context, inputTensors != nullptr, true);
    TF_LITE_ENSURE_EQ(context, outputTensors != nullptr, true);

    // Build the ops and tensors.
    TF_LITE_ENSURE_STATUS(AddOpsAndTensors(context, inputTensors, delegateOptions));
    // Map input and output tensor indices to NN
    // Make the TensorFlow Lite inputs and outputs to nn_indices.
    OH_NN_UInt32Array inputIndices;
    OH_NN_UInt32Array outputIndices;
    std::vector<uint32_t> inputsData;
    for (auto i : TfLiteIntArrayView(inputTensors)) {
        // Constant tensors are not NNRT inputs.
        if ((i != kTfLiteOptionalTensor) && (context->tensors[i].allocation_type != kTfLiteMmapRo) &&
            // The delegate might not have mapped this input (this can
            // happen if one tensor is split in several ones)
            (m_tensorMapping.LiteIndexToNn(i) != INVALID_INDEX)) {
            const int32_t inputTensorNnIndex = m_tensorMapping.LiteIndexToNn(i);
            inputsData.emplace_back(inputTensorNnIndex);
        }
    }

    std::vector<uint32_t> outputsData;
    for (auto i : TfLiteIntArrayView(outputTensors)) {
        const int32_t outputTensorNnIndex = m_tensorMapping.LiteIndexToNn(i);
        // Unmapped outputs are not added
        if (outputTensorNnIndex != INVALID_INDEX) {
            outputsData.emplace_back(outputTensorNnIndex);
        }
    }

    inputIndices.data = inputsData.data();
    outputIndices.data = outputsData.data();
    inputIndices.size = inputsData.size();
    outputIndices.size = outputsData.size();
    // Tell NN to declare inputs/outputs
    RETURN_TFLITE_ERROR_IF_NN_ERROR(m_nnrt->OH_NNModel_SpecifyInputsAndOutputs(m_nnModel, &inputIndices,
        &outputIndices), "identifying model inputs and outputs");

    RETURN_TFLITE_ERROR_IF_NN_ERROR(m_nnrt->OH_NNModel_Finish(m_nnModel), "finalizing the model");
    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::AddOpsAndTensors(TfLiteContext* context, const TfLiteIntArray* inputTensors,
    const NnrtDelegate::Options& delegateOptions)
{
    // The tensor builder allows creating a single op. It is created outside
    // the for loop to avoid reallocating the vectors.
    NnrtOpBuilderArgs opBuilderArgs = {
        .context = context,
        .nnModel = m_nnModel,
        .inputTensors = const_cast<TfLiteIntArray*>(inputTensors),
        .pTensorMapping = &m_tensorMapping,
        .delegateOptions = delegateOptions
    };
    NnrtOpBuilder builder(m_nnrt, opBuilderArgs);

    // Clear the input and output lists.
    builder.ClearInputOuputLists();

    // Add other tensors.
    TfLiteNode* node = nullptr;
    TfLiteRegistration* reg = nullptr;
    for (int32_t nodeIndex : m_delegateNodes) {
        node = nullptr;
        reg = nullptr;
        TF_LITE_ENSURE_STATUS(
            context->GetNodeAndRegistration(context, nodeIndex, &node, &reg)); // Obtain the op and registration.
        if ((node == nullptr) || (reg == nullptr)) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE_KERNEL] Get node and registration failed.");
            return kTfLiteError;
        }

        const bool scalarAsTensor = IsScalarInputSupported(reg->builtin_code);
        int32_t inputTensorFlags = 0;
        if (scalarAsTensor) {
            inputTensorFlags |= NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
        }

        // Get op type and tensors, fails if the Validate function failed.
        int32_t nnOpType;
        NnrtOpMappingArgs opMappingArgs = { context, &builder, node, nodeIndex };
        TF_LITE_ENSURE_STATUS(Map(reg->builtin_code, opMappingArgs, nnOpType));

        for (int32_t inputPos = 0; inputPos < node->inputs->size; ++inputPos) {
            if ((reg->builtin_code == kTfLiteBuiltinFullyConnected) &&
                (node->inputs->data[inputPos] == kTfLiteOptionalTensor)) {
                continue; // skip optional bias and handle it during mapping.
            }
            const auto inputIndex = node->inputs->data[inputPos];
            TF_LITE_ENSURE_STATUS(builder.AddTensorInput(inputIndex, reg->builtin_code, inputTensorFlags));
        }
        // Map outputs to NN API tensor indices.
        int32_t outputTensorFlags = 0;
        for (int32_t outputPos = 0; outputPos < node->outputs->size; ++outputPos) {
            auto outputIndex = node->outputs->data[outputPos];
            TF_LITE_ENSURE_STATUS(builder.AddTensorOutput(outputIndex, reg->builtin_code, outputTensorFlags));
        }
        TF_LITE_ENSURE_STATUS(builder.FinalizeAddOperation(static_cast<OH_NN_OperationType>(nnOpType), nodeIndex));
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::ConvertTensorTypeToNn(TfLiteContext* context,
    const std::pair<int32_t, int32_t>& indexPair, OH_NN_QuantParam* nnQuantParam, OH_NN_Tensor& nnTensor)
{
    TF_LITE_ENSURE_EQ(context, context->tensors_size > indexPair.first, true);
    TfLiteTensor* tensor = &(context->tensors[indexPair.first]);
    TF_LITE_ENSURE_EQ(context, tensor != nullptr, true);

    OH_NN_DataType nnType {OH_NN_UNKNOWN};
    TF_LITE_ENSURE_STATUS(m_tensorMapping.ConvertType(context, indexPair.first, 0, nnType));

    uint32_t tensorRank = static_cast<uint32_t>(tensor->dims->size);
    int32_t* tensorDims = reinterpret_cast<int32_t*>(tensor->dims->data);
    if (tensorDims == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-DELEGATE_KERNEL] The tensorDims is nullptr when converting the type of tensors to nnrt.");
        return kTfLiteError;
    }

    // treat scalar input as single cell tensor in NNRT.
    if (tensorRank == 0) {
        tensorRank = SCALAR_RANK;
        *tensorDims = SCALAR_RANK;
    }

    nnTensor.dataType = nnType;
    nnTensor.dimensionCount = tensorRank;
    nnTensor.dimensions = tensorDims;
    nnTensor.quantParam = nnQuantParam;
    nnTensor.type = OH_NN_TENSOR;

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::SetInputTensors(TfLiteContext* context, TfLiteNode* node,
    OH_NNExecutor* pNnExecution, OH_NN_Tensor& nnTensor)
{
    TF_LITE_ENSURE_EQ(context, node != nullptr, true);
    TF_LITE_ENSURE_EQ(context, pNnExecution != nullptr, true);

    // Note: we access tflite tensors using
    // absolute indices but NN api indices inputs by relative indices.
    int32_t relativeIndex = 0;
    OH_NN_QuantParam* nnQuantParam = nullptr;
    TfLiteIntArray* tensors = node->inputs;
    TF_LITE_ENSURE_EQ(context, tensors != nullptr, true);

    for (auto absoluteIndex : TfLiteIntArrayView(tensors)) {
        if (absoluteIndex == kTfLiteOptionalTensor) {
            continue;
        }

        std::pair<int32_t, int32_t> indexPair = std::make_pair(absoluteIndex, relativeIndex);
        ConvertTensorTypeToNn(context, indexPair, nnQuantParam, nnTensor);

        TfLiteTensor* tensor = &context->tensors[absoluteIndex];
        TF_LITE_ENSURE_EQ(context, tensor != nullptr, true);

        if (tensor->allocation_type != kTfLiteMmapRo) {
            RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(m_nnrt->OH_NNExecutor_SetInput(pNnExecution, relativeIndex,
                &nnTensor, tensor->data.raw, tensor->bytes),
                "associating NNRT execution output with a memory object", tensor);
            ++relativeIndex;
        } else {
            continue;
        }
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::SetOutputTensors(TfLiteContext* context, TfLiteNode* node,
    OH_NNExecutor* pNnExecution)
{
    TF_LITE_ENSURE_EQ(context, node != nullptr, true);
    TF_LITE_ENSURE_EQ(context, pNnExecution != nullptr, true);

    // Note: we access tflite tensors using
    // absolute indices but NN api indices inputs by relative indices.
    int32_t relativeIndex = 0;
    TfLiteIntArray* tensors = node->outputs;
    TF_LITE_ENSURE_EQ(context, tensors != nullptr, true);
    for (auto absoluteIndex : TfLiteIntArrayView(tensors)) {
        if (m_tensorMapping.LiteIndexToNn(absoluteIndex) == INVALID_INDEX) {
            continue;
        }

        TfLiteTensor* tensor = &context->tensors[absoluteIndex];
        TF_LITE_ENSURE_EQ(context, tensor != nullptr, true);
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            m_nnrt->OH_NNExecutor_SetOutput(pNnExecution, relativeIndex, tensor->data.raw, tensor->bytes),
            "associating NNRT execution output to a memory object", tensor);
        ++relativeIndex;
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegateKernel::SetNnOptions(TfLiteContext* context, const NnrtDelegate::Options& delegateOptions)
{
    if (context == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-DELEGATE_KERNEL] The context is nullptr when setting nnrt options.");
        return kTfLiteError;
    }

    RETURN_TFLITE_ERROR_IF_NN_ERROR(m_nnrt->OH_NNCompilation_SetDevice(m_pNnCompilation, m_nnrtDevice),
        "creating NNRT compilation");

    auto performance = delegateOptions.executionPerformance;
    if (performance != OH_NN_PERFORMANCE_NONE) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_COMPILE(
            m_nnrt->OH_NNCompilation_SetPerformanceMode(m_pNnCompilation, performance),
                "setting compilation performance");
    }

    // Set cacahe, if cacheDir & modelToken & device is valid.
    std::string cacheDir = delegateOptions.cacheDir;
    std::string modelToken = delegateOptions.modelToken;
    uint32_t version = delegateOptions.version;
    if (!cacheDir.empty() && (!IsUseTargetDevice(delegateOptions) ||
        (delegateOptions.acceleratorName == NNRT_REFERENCE_DEVICE))) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_COMPILE(
            m_nnrt->OH_NNCompilation_SetCache(m_pNnCompilation, cacheDir.c_str(), version),
            "setting compilation cache");
    } else if (cacheDir.empty()) {
        TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "The cacheDir is empty, will not load or save cache.");
    }
    return kTfLiteOk;
}
} // namespace nnrt
} // namespace delegate
} // tflite
