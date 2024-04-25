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

#include "exp_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 3;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Exp";

ExpBuilder::ExpBuilder() {}

ExpBuilder::~ExpBuilder() {}

OH_NN_ReturnCode ExpBuilder::SetBase(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[Exp] The base should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Exp] The base should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Exp] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_base = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ExpBuilder::SetScale(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[Exp] The scale should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Exp] The scale should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Exp] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_scale = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ExpBuilder::SetShift(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[Exp] The shift should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Exp] The shift should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Exp] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_shift = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ExpBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                   const std::vector<uint32_t>& inputsIndex,
                                   const std::vector<uint32_t>& outputsIndex,
                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Exp] Build failed, the exp operation has been build. cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Exp] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    ret = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Exp] Build failed, passed invalid param index.");
        return ret;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            ret = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[Exp] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (ret != OH_NN_SUCCESS) {
            LOGE("[Exp] Build failed, passed invalid param.");
            return ret;
        }
    }

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr ExpBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Exp] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_ExpFusion_CreatePrimitive(m_base, m_scale, m_shift);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(ExpBuilder, OH_NN_OPS_EXP);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS