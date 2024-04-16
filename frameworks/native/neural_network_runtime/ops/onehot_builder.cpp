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

#include "onehot_builder.h"

#include "mindir.h"

#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 4;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 1;
static const std::string OP_NAME = "Onehot";

OnehotBuilder::OnehotBuilder() {}

OnehotBuilder::~OnehotBuilder() {}

OH_NN_ReturnCode OnehotBuilder::SetAxis(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Onehot] Onehot SetAxis failed. The axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Onehot] SetAxis failed, the axis passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_axis = *static_cast<int64_t*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OnehotBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                      const std::vector<uint32_t>& inputsIndex,
                                      const std::vector<uint32_t>& outputsIndex,
                                      const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Onehot] Onehot build failed. operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Onehot] Onehot build failed. Passed invalid input or output index of Onehot operation index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Onehot] Onehot build failed. Passed invalid param index of Onehot operation index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[Onehot] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Onehot] Onehot Build failed. Passed invalid param.");
            return returnCode;
        }
    }

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr OnehotBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Onehot] Onehot GetPrimitive failed. Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_OneHot_CreatePrimitive(m_axis);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(OnehotBuilder, OH_NN_OPS_ONE_HOT);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // OHOS