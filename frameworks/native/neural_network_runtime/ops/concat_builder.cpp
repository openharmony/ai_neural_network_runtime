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

#include "concat_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static constexpr int MINIMUM_INTPUT = 2;
static constexpr int OUTPUT_NUM = 1;
static constexpr int AXIS_LENGTH = 1;
static const std::string OP_NAME = "Concat";

ConcatBuilder::ConcatBuilder() {}

ConcatBuilder::~ConcatBuilder() {}

OH_NN_ReturnCode ConcatBuilder::SetAxis(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetElementCount() != AXIS_LENGTH) {
        LOGE("[Concat] SetAxis failed, the Activation shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Concat] SetAxis failed, the axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Concat] SetAxis GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_axis = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ConcatBuilder::Build(const std::vector<uint32_t>& paramsIndex,
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Concat] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (inputsIndex.size() < MINIMUM_INTPUT) {
        LOGE("[Concat] Build failed, Concat need more than one inputs.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (outputsIndex.size() != OUTPUT_NUM) {
        LOGE("[Concat] Build failed, The number of index of outputs not equal to 1.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode returnCode = SetInputsAndOutputs(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Concat] Build failed, set inputs or outputs failed.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_CONCAT_AXIS:
                returnCode = SetAxis(tensor);
                break;
            default:
                LOGE("[Concat] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Concat] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ConcatBuilder::SetInputsAndOutputs(const std::vector<uint32_t>& inputsIndex,
                                                    const std::vector<uint32_t>& outputsIndex,
                                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    size_t allTensorsSize = allTensors.size();
    for (auto index : inputsIndex) {
        if (index >= allTensorsSize) {
            LOGE("[Concat] Invalid input index, it is out of range %zu.", allTensorsSize);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    for (auto index : outputsIndex) {
        if (index >= allTensorsSize) {
            LOGE("[Concat] Invalid output index, it is out of range %zu.", allTensorsSize);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    m_inputsIndex.clear();
    m_inputsIndex = inputsIndex;

    m_outputsIndex.clear();
    m_outputsIndex = outputsIndex;

    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr ConcatBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Concat] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_Concat_CreatePrimitive(m_axis);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(ConcatBuilder, OH_NN_OPS_CONCAT);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS