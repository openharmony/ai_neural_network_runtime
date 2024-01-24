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

#include "stack_builder.h"

#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_MIN_NUM = 2;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "Stack";

StackBuilder::StackBuilder() {}

StackBuilder::~StackBuilder() {}

OH_NN_ReturnCode StackBuilder::SetAxis(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[StackBuilder] The last input axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != 1) {
        LOGE("[StackBuilder] The last input axis should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[StackBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_axis = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode StackBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                     const std::vector<uint32_t>& inputsIndex,
                                     const std::vector<uint32_t>& outputsIndex,
                                     const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[StackBuilder] Stack operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (inputsIndex.size() < INPUT_MIN_NUM) {
        LOGE("[StackBuilder] The number of index of inputs don't larger than %d.", INPUT_MIN_NUM);
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputsIndex.size() != OUTPUT_NUM) {
        LOGE("[StackBuilder] The number of index of outputs don't equal to %d.", OUTPUT_NUM);
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    OH_NN_ReturnCode returnCode;
    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_STACK_AXIS:
                returnCode = SetAxis(tensor);
                break;
            default:
                LOGE("[StackBuilder] Parameter Type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[StackBuilder] Passed invalid param.");
            return returnCode;
        }
    }

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphTensorPtr StackBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[StackBuilder] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_Stack_CreatePrimitive(m_axis);
    if (primitive == nullptr) {
        LOGE("[StackBuilder] MindIR_Stack_CreatePrimitive failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphTensorPtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(StackBuilder, OH_NN_OPS_STACK);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
