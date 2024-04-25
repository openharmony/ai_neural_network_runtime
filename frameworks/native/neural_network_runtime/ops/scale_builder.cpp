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

#include "scale_builder.h"

#include "ops_registry.h"
#include "validation.h"
#include "transform.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 3;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 2;
static const int SCALE_LENGTH = 1;
static const std::string OP_NAME = "Scale";

ScaleBuilder::ScaleBuilder() {}

ScaleBuilder::~ScaleBuilder() {}

OH_NN_ReturnCode ScaleBuilder::SetAxis(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[ScaleBuilder] SetAxis failed, the axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ScaleBuilder] SetAxis failed, the axis dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ScaleBuilder] SetAxis failed, the axis passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_axis = static_cast<uint64_t*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ScaleBuilder::SetActivationType(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[ScaleBuilder] SetActivationType failed, the activation should be type OH_NN_INT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ScaleBuilder] SetActivationType failed, the activation dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ScaleBuilder] SetActivationType failed, the activation passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    const int8_t* fuseData = static_cast<const int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*fuseData))) {
        LOGE("[ScaleBuilder] SetActivationType failed, the activation input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto fuseType = (OH_NN_FuseType)(*fuseData);
    m_activationType = NNToMS::TransfromFusionType(fuseType);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ScaleBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                     const std::vector<uint32_t>& inputsIndex,
                                     const std::vector<uint32_t>& outputsIndex,
                                     const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[ScaleBuilder] Build failed, the scale operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[ScaleBuilder] Build failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[ScaleBuilder] Build failed, passed invaid param index.");
        return returnCode;
    }

    for (uint32_t i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[ScaleBuilder] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[ScaleBuilder] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr ScaleBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[ScaleBuilder] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_ScaleFusion_CreatePrimitive(*m_axis, m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(ScaleBuilder, OH_NN_OPS_SCALE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS