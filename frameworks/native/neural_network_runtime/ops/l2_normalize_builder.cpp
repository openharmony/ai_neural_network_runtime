/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#include "l2_normalize_builder.h"

#include "transform.h"
#include "validation.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 3;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "L2Normalize";

L2NormalizeBuilder::L2NormalizeBuilder() {}

L2NormalizeBuilder::~L2NormalizeBuilder() {}

OH_NN_ReturnCode L2NormalizeBuilder::SetAxis(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[L2Normalize] The axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_axis.clear();

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[L2Normalize] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int64_t* pAxis = static_cast<int64_t*>(buffer);

    uint32_t elementCount = tensor->GetElementCount();
    for (uint32_t i = 0; i < elementCount; ++i) {
        m_axis.emplace_back(*pAxis);
        ++pAxis;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode L2NormalizeBuilder::SetEpsilon(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[L2Normalize] The epsilon should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[L2Normalize] The epsilon should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[L2Normalize] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_epsilon = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode L2NormalizeBuilder::SetActivationType(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[L2Normalize] SetActivationType failed, the activationType should have type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[L2Normalize] SetActivationType failed, the activationType shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[L2Normalize] SetActivationType GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    int8_t* pActivationType = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*pActivationType))) {
        LOGE("[L2Normalize] SetActivationType failed, activationType input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_activationType = NNToMS::TransfromFusionType((OH_NN_FuseType)(*pActivationType));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode L2NormalizeBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                           const std::vector<uint32_t>& inputsIndex,
                                           const std::vector<uint32_t>& outputsIndex,
                                           const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[L2Normalize] Build failed, the depthToSpace operation has been build. cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[L2Normalize] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    ret = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[L2Normalize] Build failed, passed invalid param index.");
        return ret;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            ret = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[L2Normalize] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (ret != OH_NN_SUCCESS) {
            LOGE("[L2Normalize] Build failed, passed invalid param.");
            return ret;
        }
    }
    
    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr L2NormalizeBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[L2Normalize] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_L2NormalizeFusion_CreatePrimitive(m_axis, m_epsilon, m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(L2NormalizeBuilder, OH_NN_OPS_L2_NORMALIZE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS