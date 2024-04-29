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

#include "fullconnection_builder.h"

#include "transform.h"
#include "validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static constexpr int OUTPUT_NUM = 1;
static constexpr int PARAM_MAX_NUM = 4;
static constexpr int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "FullConnection";

FullConnectionBuilder::FullConnectionBuilder() {}

FullConnectionBuilder::~FullConnectionBuilder() {}

OH_NN_ReturnCode FullConnectionBuilder::SetFullConnectionInput(const std::vector<uint32_t>& inputsIndex,
                                                               const std::vector<uint32_t>& outputsIndex,
                                                               const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (outputsIndex.size() != OUTPUT_NUM) {
        LOGE("[FullConnection] SetFullConnectionInput failed, the index of outputs don't equal to %d.", OUTPUT_NUM);
        return OH_NN_INVALID_PARAMETER;
    }

    size_t allTensorsSize = allTensors.size();
    bool isOverTensorSize = std::any_of(inputsIndex.begin(), inputsIndex.end(), [allTensorsSize](uint32_t index) {
        return index >= allTensorsSize;
    });
    if (isOverTensorSize) {
        LOGE("[FullConnection] SetFullConnectionInput failed, the index of inputs is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode FullConnectionBuilder::SetHasBias(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[FullConnection] The hasBias should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[FullConnection] The hasBias should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[FullConnection] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_hasBias = *(static_cast<bool*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode FullConnectionBuilder::SetUseAxis(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[FullConnection] The useAxis should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[FullConnection] The useAxis should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[FullConnection] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    bool useAxis = *(static_cast<bool*>(buffer));
    if (m_axisIsSet && !useAxis) {
        LOGE("[FullConnection] m_useAxis is not allowed to be set to false when m_axis is already set.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_useAxis = useAxis;
    m_useAxisIsSet = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode FullConnectionBuilder::SetFullConnectionActivation(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    // Set Activation
    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[FullConnection] SetFullConnectionActivation failed, the Activation shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[FullConnection] SetFullConnectionActivation failed, the Activation should have type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[FullConnection] SetFullConnectionActivation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    int8_t* pFuseData = static_cast<int8_t*>(tensor->GetBuffer());
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*pFuseData))) {
        LOGE("[FullConnection] SetFullConnectionActivation failed, activation input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_activationType = NNToMS::TransfromFusionType((OH_NN_FuseType)(*pFuseData));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode FullConnectionBuilder::SetAxis(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[FullConnection] SetAxis failed, the axis shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[FullConnection] SetAxis failed, the Axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[FullConnection] SetAxis GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_useAxisIsSet && !m_useAxis) {
        LOGE("[FullConnection] m_useAxis has been set to false, axis is not allowed.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_axis = *static_cast<int64_t*>(buffer);
    m_useAxis = true;
    m_axisIsSet = true;
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode FullConnectionBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                              const std::vector<uint32_t>& inputsIndex,
                                              const std::vector<uint32_t>& outputsIndex,
                                              const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[FullConnection] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = SetFullConnectionInput(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[FullConnection] Build failed, SetFullConnectionInput failed.");
        return returnCode;
    }

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[FullConnection] Build failed, passed invalid param index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i]; // 参数 tensor
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[FullConnection] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[FullConnection] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr FullConnectionBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[FullConnection] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_FullConnection_CreatePrimitive(m_hasBias, m_useAxis,
        m_axis, m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(FullConnectionBuilder, OH_NN_OPS_FULL_CONNECTION);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS