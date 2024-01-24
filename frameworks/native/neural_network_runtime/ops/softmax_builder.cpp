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

#include "softmax_builder.h"

#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "Softmax";

SoftmaxBuilder::SoftmaxBuilder() {}

SoftmaxBuilder::~SoftmaxBuilder() {}

OH_NN_ReturnCode SoftmaxBuilder::SetAxis(std::shared_ptr<NNTensor> tensor)
{
    // Set Axis
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[SoftmaxBuilder] The 2nd input axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != 1) {
        LOGE("[SoftmaxBuilder] The 2nd input axis should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_axis.clear();

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[SoftmaxBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_axis.emplace_back(*(static_cast<const int64_t*>(buffer)));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SoftmaxBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                       const std::vector<uint32_t>& inputsIndex,
                                       const std::vector<uint32_t>& outputsIndex,
                                       const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[SoftmaxBuilder] Softmax operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[SoftmaxBuilder] Passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_SOFTMAX_AXIS:
                returnCode = SetAxis(tensor);
                break;
            default:
                LOGE("[SoftmaxBuilder] Parameter Type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[SoftmaxBuilder] Passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphTensorPtr SoftmaxBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[SoftmaxBuilder] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_Softmax_CreatePrimitive(m_axis);
    if (primitive == nullptr) {
        LOGE("[SoftmaxBuilder] Create primitive of Softmax failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphTensorPtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(SoftmaxBuilder, OH_NN_OPS_SOFTMAX);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS