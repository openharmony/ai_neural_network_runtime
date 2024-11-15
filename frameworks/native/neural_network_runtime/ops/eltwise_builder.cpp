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

#include "eltwise_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 1;
static constexpr int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Eltwise";

EltwiseBuilder::EltwiseBuilder() {}

EltwiseBuilder::~EltwiseBuilder() {}

OH_NN_ReturnCode EltwiseBuilder::SetMode(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[Eltwise] SetMode failed, the EltwiseMode should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Eltwise] SetMode failed, the eltwiseMode shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Eltwise] SetMode GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    int8_t eltwiseMode = *static_cast<int8_t*>(buffer);
    if (eltwiseMode < mindspore::lite::ELTWISE_MODE_PROD ||
        eltwiseMode > mindspore::lite::ELTWISE_MODE_UNKNOWN) {
        LOGE("[Eltwise] SetMode failed, passed invalid eltwiseMode, received %d", eltwiseMode);
        return OH_NN_INVALID_PARAMETER;
    }
    m_mode = (mindspore::lite::EltwiseMode)eltwiseMode;

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode EltwiseBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                       const std::vector<uint32_t>& inputsIndex,
                                       const std::vector<uint32_t>& outputsIndex,
                                       const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Eltwise] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Eltwise] Build failed, passed invalid input index or output indices.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Eltwise] Build failed, passed invalid input param indices.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[Eltwise] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Eltwise] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr EltwiseBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Eltwise] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_Eltwise_CreatePrimitive(m_mode);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(EltwiseBuilder, OH_NN_OPS_ELTWISE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS