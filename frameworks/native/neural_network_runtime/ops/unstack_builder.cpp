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
static const int OUTPUT_MIN_NUM = 1;
static const int PARAM_MAX_NUM = 1;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Unstack";

UnstackBuilder::UnstackBuilder() {}

UnstackBuilder::~UnstackBuilder() {}

OH_NN_ReturnCode UnstackBuilder::SetAxis(const std::shared_ptr<NNTensor>& tensor)
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

    if (inputsIndex.size() != INPUT_NUM) {
        LOGE("[Unstack] The number of index of inputs don't equal to %d.", INPUT_NUM);
        return OH_NN_INVALID_PARAMETER;
    }
    
    if (outputsIndex.size() < OUTPUT_MIN_NUM) {
        LOGE("[Unstack] The number of index of outputs don't larger than %d.", OUTPUT_MIN_NUM);
        return OH_NN_INVALID_PARAMETER;
    }
    
    size_t allTensorsSize = allTensors.size();
    bool isOverTensorSize = std::any_of(inputsIndex.begin(), inputsIndex.end(), [allTensorsSize](uint32_t index) {
        return index >= allTensorsSize;
    });
    if (isOverTensorSize) {
        LOGE("The index of inputs is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    isOverTensorSize = std::any_of(inputsIndex.begin(), inputsIndex.end(), [allTensorsSize](uint32_t index) {
        return index >= allTensorsSize;
    });
    if (isOverTensorSize) {
        LOGE("The index of outputs is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    auto returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Unstack] Passed invalid param index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
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
