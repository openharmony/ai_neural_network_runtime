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

#include "layernorm_builder.h"

#include "mindir.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 3;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 3;
static const int INPUT_X = 0;
static const int INPUT_GAMMA = 1;
static const int INPUT_BETA = 2;
static const std::string OP_NAME = "LayerNorm";

LayerNormBuilder::LayerNormBuilder() {}

LayerNormBuilder::~LayerNormBuilder() {}

OH_NN_ReturnCode LayerNormBuilder::SetBeginNormAxis(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LayerNormBuilder] SetBeginNormAxis failed. The has_bias should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!tensor->IsScalar()) {
        LOGE("[LayerNormBuilder] SetBeginNormAxis failed. The beginNormAxis should be a scalar value.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LayerNormBuilder] SetBeginNormAxis failed, the beginNormAxis passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_beginNormAxis = *static_cast<int64_t*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LayerNormBuilder::SetEpsilon(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LayerNormBuilder] SetEpsilon failed. The epsilon should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!tensor->IsScalar()) {
        LOGE("[LayerNormBuilder] SetEpsilon failed. The epsilon should be a scalar value.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LayerNormBuilder] SetEpsilon failed, the epsilon passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_epsilon = *static_cast<float*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LayerNormBuilder::SetBeginParamsAxis(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LayerNormBuilder] SetBeginParamsAxis failed. The has_bias should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!tensor->IsScalar()) {
        LOGE("[LayerNormBuilder] SetBeginParamsAxis failed. The beginNormAxis should be a scalar value.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LayerNormBuilder] SetBeginParamsAxis failed, the beginParamsAxis passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_beginParamsAxis = *static_cast<int64_t*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LayerNormBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                         const std::vector<uint32_t>& inputsIndex,
                                         const std::vector<uint32_t>& outputsIndex,
                                         const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[LayerNormBuilder] Build failed. LayerNorm operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[LayerNormBuilder] Build failed. Passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[LayerNormBuilder] Build failed. Passed invalid param index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[LayerNormBuilder] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[LayerNormBuilder] Build failed. Passed invalid param.");
            return returnCode;
        }
    }

    auto inputShape = allTensors[inputsIndex[INPUT_X]]->GetDimensions();
    int64_t inputShapeSize = static_cast<int64_t>(inputShape.size());
    // beginNormAxis must great than 1, because normal shape cannot equal input shape.
    if (m_beginNormAxis >= inputShapeSize || m_beginNormAxis < 1) {
        LOGE("[LayerNormBuilder] Build failed, invalid beginNormAxis value, it should be [1, rank(input)).");
        return OH_NN_INVALID_PARAMETER;
    }
    // validate gamma and beta shape
    returnCode = ValidateGammaAndBetaShape(inputsIndex, m_beginNormAxis, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        return returnCode;
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr LayerNormBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[LayerNormBuilder] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_LayerNormFusion_CreatePrimitive(m_beginNormAxis,
        m_epsilon, m_elementwiseAffine, m_beginParamsAxis);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

OH_NN_ReturnCode LayerNormBuilder::ValidateGammaAndBetaShape(const std::vector<uint32_t>& inputsIndex,
    int64_t beginAxis, const std::vector<std::shared_ptr<NNTensor>>& allTensors) const
{
    auto inputShape = allTensors[inputsIndex[INPUT_X]]->GetDimensions();
    auto gammaShape = allTensors[inputsIndex[INPUT_GAMMA]]->GetDimensions();
    auto betaShape = allTensors[inputsIndex[INPUT_BETA]]->GetDimensions();
    int64_t inputShapeSize = static_cast<int64_t>(inputShape.size());
    if (gammaShape.size() != static_cast<size_t>(inputShapeSize - beginAxis)) {
        LOGE("[LayerNormBuilder] Invalid gamma dimension, gamma dimension should be equal to normalized dimension.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (betaShape.size() != static_cast<size_t>(inputShapeSize - beginAxis)) {
        LOGE("[LayerNormBuilder] Invalid beta dimension, beta dimension should be equal to normalized dimension.");
        return OH_NN_INVALID_PARAMETER;
    }

    for (auto i = beginAxis; i < inputShapeSize; i++) {
        if (gammaShape[i - beginAxis] != inputShape[i]) {
            LOGE("[LayerNormBuilder] Invalid gamma shape, gamma shape should equal to normalized shape.");
            return OH_NN_INVALID_PARAMETER;
        }
        if (betaShape[i - beginAxis] != inputShape[i]) {
            LOGE("[LayerNormBuilder] Invalid beta shape, bata shape should equal to normalized shape.");
            return OH_NN_INVALID_PARAMETER;
        }
    }

    return OH_NN_SUCCESS;
}

REGISTER_OPS(LayerNormBuilder, OH_NN_OPS_LAYER_NORM);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS