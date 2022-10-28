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

#include "pooling_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int NUM_ELEMENT_PAD_MODE = 1;
static const int NUM_ELEMENT_PAD_LIST = 4;
static const int ACTIVATION_LENGTH = 1;

OH_NN_ReturnCode PoolingBuilder::PoolingBuild(const std::vector<uint32_t>& paramsIndex,
                                              const std::vector<uint32_t>& inputsIndex,
                                              const std::vector<uint32_t>& outputsIndex,
                                              const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[PoolingBuilder] PoolingBuild failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    // Set input and output
    OH_NN_ReturnCode returnCode = SetInputAndOutput(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[PoolingBuilder] PoolingBuild failed, the SetInputAndOutput failed.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_AVG_POOL_KERNEL_SIZE:
            case OH_NN_MAX_POOL_KERNEL_SIZE:
                returnCode = SetKernel(tensor);
                break;
            case OH_NN_AVG_POOL_STRIDE:
            case OH_NN_MAX_POOL_STRIDE:
                returnCode = SetStrides(tensor);
                break;
            case OH_NN_AVG_POOL_PAD_MODE:
            case OH_NN_MAX_POOL_PAD_MODE:
            case OH_NN_MAX_POOL_PAD:
            case OH_NN_AVG_POOL_PAD:
                returnCode = SetPadModeOrPaddings(tensor);
                break;
            case OH_NN_AVG_POOL_ACTIVATION_TYPE:
            case OH_NN_MAX_POOL_ACTIVATION_TYPE:
                returnCode = SetActivation(tensor);
                break;
            default:
                LOGE("[PoolingBuilder] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[PoolingBuilder] PoolingBuild failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PoolingBuilder::SetInputAndOutput(const std::vector<uint32_t>& inputsIndex,
                                                   const std::vector<uint32_t>& outputsIndex,
                                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[PoolingBuilder] SetInputAndOutput failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PoolingBuilder::SetKernel(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set kernelSize
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[PoolingBuilder] SetKernel failed, the KernelSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[PoolingBuilder] SetKernel GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    const int64_t* pKernelSize = reinterpret_cast<const int64_t*>(buffer);
    int kernelSize = tensor->GetElementCount();
    m_kernelSize.assign(pKernelSize, pKernelSize + kernelSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PoolingBuilder::SetStrides(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Strides
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[PoolingBuilder] SetStrides failed, the Strides should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[PoolingBuilder] SetStrides GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    const int64_t* pStrides = reinterpret_cast<const int64_t*>(buffer);
    int strideslSize = tensor->GetElementCount();
    m_strides.assign(pStrides, pStrides + strideslSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PoolingBuilder::SetPadModeOrPaddings(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[PoolingBuilder] SetPadModeOrPaddings GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    size_t tensorElementCount = tensor->GetElementCount();
    // Set PadMode or PadList
    if (tensorElementCount == NUM_ELEMENT_PAD_MODE) {
        // PadMode
        if (tensor->GetDataType() != OH_NN_INT8) {
            LOGE("[PoolingBuilder] SetPadModeOrPaddings failed, the type of padMode should be OH_NN_INT8.");
            return OH_NN_INVALID_PARAMETER;
        }

        int8_t* pPadMode = static_cast<int8_t*>(buffer);
        if (!OHOS::NeuralNetworkRuntime::Validation::ValidatePadMode(*pPadMode)) {
            LOGE("[PoolingBuilder] SetPadModeOrPaddings failed, invalid pad mode.");
            return OH_NN_INVALID_PARAMETER;
        }
        m_padMode = NNToMS::TransformPadModeValue(*pPadMode);
    } else if (tensorElementCount == NUM_ELEMENT_PAD_LIST) {
        if (tensor->GetDataType() != OH_NN_INT64) {
            LOGE("[PoolingBuilder] SetPadModeOrPaddings failed, the type of padList should be OH_NN_INT64.");
            return OH_NN_INVALID_PARAMETER;
        }

        int64_t* pPad = static_cast<int64_t*>(buffer);
        // PadList
        m_pad.clear();
        for (int i = 0; i < NUM_ELEMENT_PAD_LIST; i++) {
            m_pad.emplace_back(static_cast<int64_t>(pPad[i]));
        }
    } else {
        LOGE("[PoolingBuilder] SetPadModeOrPaddings failed, invalid element size of padMode or padList,"
            "padMode should be single value, and padList should be 4.");
        return OH_NN_INVALID_PARAMETER;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PoolingBuilder::SetActivation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set ActivationType
    if (tensor->GetElementCount() != ACTIVATION_LENGTH) {
        LOGE("[PoolingBuilder] SetActivation failed, the Activation shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[PoolingBuilder] SetActivation failed, the ActivationType should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[PoolingBuilder] SetActivation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    int8_t* pFuseData = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*pFuseData))) {
        LOGE("[PoolingBuilder] SetActivation failed, activation input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    auto fuseType = (OH_NN_FuseType)(*pFuseData);
    m_activationType = NNToMS::TransfromFusionType(fuseType);

    return OH_NN_SUCCESS;
}
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
