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

#include "top_k_builder.h"

#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const std::string OP_NAME = "TopK";
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 2;

TopKBuilder::TopKBuilder() {}

TopKBuilder::~TopKBuilder() {}

OH_NN_ReturnCode TopKBuilder::SetSorted(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[TopK] The sorted should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[TopK] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_sorted = *(static_cast<const bool *>(buffer));

    return OH_NN_SUCCESS;
}

/**
 * Build method.
 * 1.build primitive of ops.
 * 2.build inputIndex of ops.
 * 3.build outputIndex of ops.
 */
OH_NN_ReturnCode TopKBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                    const std::vector<uint32_t>& inputsIndex,
                                    const std::vector<uint32_t>& outputsIndex,
                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[TopK] Build operation has been completed, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[TopK] Passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_TOP_K_SORTED:
                returnCode = SetSorted(tensor);
                break;
            default:
                LOGE("[TopK] Parameter Type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[TopK] Passed invalid param.");
            return returnCode;
        }
    }

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr TopKBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[TopK] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    int64_t axis = 0;
    auto primitive = mindspore::lite::MindIR_TopKFusion_CreatePrimitive(m_sorted, axis);
    if (primitive == nullptr) {
        LOGE("[TopK] MindIR_TopKFusion_CreatePrimitive failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(TopKBuilder, OH_NN_OPS_TOP_K);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
