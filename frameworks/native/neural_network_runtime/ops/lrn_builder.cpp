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

#include "lrn_builder.h"

#include "mindir.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 5;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "LRN";
static const std::unordered_map<int, std::string> normRegionList = {{0, "ACROSS_CHANNELS"}};

LRNBuilder::LRNBuilder() {}

LRNBuilder::~LRNBuilder() {}

OH_NN_ReturnCode LRNBuilder::SetDepthRadius(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LRN] The depthRadius should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LRN] The depthRadius should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LRN] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_depthRadius = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LRNBuilder::SetBias(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LRN] The bias should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LRN] The bias should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LRN] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_bias = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LRNBuilder::SetAlpha(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LRN] The alpha should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LRN] The alpha should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LRN] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_alpha = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LRNBuilder::SetBeta(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LRN] The beta should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LRN] The beta should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LRN] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_beta = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LRNBuilder::SetNormRegion(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT32) {
        LOGE("[LRN] The normRegion should be type OH_NN_INT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LRN] The beta should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LRN] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    int normRegionKey = *(static_cast<int*>(buffer));
    auto it = normRegionList.find(normRegionKey);
    if (it != normRegionList.end()) {
        m_normRegion = it->second;
    } else {
        LOGE("[LRN] The normRegion should between [0, 0], but get %d.", normRegionKey);
        LOGE("[LRN] normRegion value: 0-ACROSS_CHANNELS");
        return OH_NN_INVALID_PARAMETER;
    }
    
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LRNBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                   const std::vector<uint32_t>& inputsIndex,
                                   const std::vector<uint32_t>& outputsIndex,
                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[LRN] Build failed, the LRN operation has been build. cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[LRN] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    ret = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[LRN] Build failed, passed invalid input or output index.");
        return ret;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            ret = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[LRN] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (ret != OH_NN_SUCCESS) {
            LOGE("[LRN] Build failed, passed invalid param.");
            return ret;
        }
    }
    
    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr LRNBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[LRN] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_LRN_CreatePrimitive(m_depthRadius, m_bias, m_alpha,
                                                                  m_beta, m_normRegion);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(LRNBuilder, OH_NN_OPS_LRN);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS