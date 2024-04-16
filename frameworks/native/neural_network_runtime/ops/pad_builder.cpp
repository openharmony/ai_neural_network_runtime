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

#include "pad_builder.h"

#include "transform.h"
#include "validation.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 2;
static const int SCALE_LENGTH = 1;
static const std::string OP_NAME = "Pad";
static const std::unordered_map<int, mindspore::lite::PaddingMode> paddingList = {
    {0, mindspore::lite::PADDING_MODE_CONSTANT},
    {1, mindspore::lite::PADDING_MODE_REFLECT},
    {2, mindspore::lite::PADDING_MODE_SYMMETRIC},
    {3, mindspore::lite::PADDING_MODE_RESERVED}};

PadBuilder::PadBuilder() {}

PadBuilder::~PadBuilder() {}

OH_NN_ReturnCode PadBuilder::SetPaddingMode(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[Pad] SetPaddingMode failed, the paddingMode shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT32) {
        LOGE("[Pad] SetPaddingMode failed, the paddingMode should be type OH_NN_INT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Pad] SetPaddingMode GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    int paddingModeKey = *(static_cast<int*>(buffer));
    auto it = paddingList.find(paddingModeKey);
    if (it != paddingList.end()) {
        m_paddingMode = it->second;
    } else {
        LOGE("[DepthToSpace] The padding mode value should between [0, 3], but get %d.", paddingModeKey);
        LOGE("[DepthToSpace] paddingMode value:");
        LOGE(" 0-PADDING_MODE_CONSTANT, 1-PADDING_MODE_REFLECT, 2-PADDING_MODE_SYMMETRIC, 3-PADDING_MODE_RESERVED");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PadBuilder::SetConstantValue(const std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[Pad] Pad SetConstantValue failed. The constant_value should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[Pad] Pad SetConstantValue failed. The constant_value should be type OH_NN_FLOAT32");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Pad] SetConstantValue failed, the constantValue passed an empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_constantValue = *static_cast<float*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode PadBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                   const std::vector<uint32_t>& inputsIndex,
                                   const std::vector<uint32_t>& outputsIndex,
                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Pad] Pad Build failed. operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Pad] Pad Build failed. Passed invalid input or output index of Pad operation index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Pad] Pad Build failed. Passed invalid param index of Pad operation index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[Pad] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Pad] Pad Build failed. Passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}
LiteGraphPrimitvePtr PadBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Pad] GetPrimitive failed. Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    std::vector<std::vector<int64_t>> paddings;

    void* primitive = MindIR_PadFusion_CreatePrimitive(paddings, m_paddingMode, m_constantValue);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(PadBuilder, OH_NN_OPS_PAD);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespcae OHOS