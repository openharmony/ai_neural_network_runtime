/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "unstack_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Unstack";

UnstackBuilder::UnstackBuilder() {}

UnstackBuilder::~UnstackBuilder() {}

OH_NN_ReturnCode UnstackBuilder::SetAxis(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Unstack] The axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Unstack] The axis should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Unstack] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_axis = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode UnstackBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                       const std::vector<uint32_t>& inputsIndex,
                                       const std::vector<uint32_t>& outputsIndex,
                                       const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Unstack] Build failed, the Unstack operation has been build. cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Unstack] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;
    
    OH_NN_ReturnCode returnCode;
    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_UNSTACK_AXIS:
                returnCode = SetAxis(tensor);
                break;
            default:
                LOGE("[Unstack] Build failed, param invalid, type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Unstack] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr UnstackBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Unstack] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_Unstack_CreatePrimitive(m_axis);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(UnstackBuilder, OH_NN_OPS_UNSTACK);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
