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

#include "mindir.h"

#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const int SCALE_LENGTH = 1;
static const std::string OP_NAME = "Pad";

PadBuilder::PadBuilder() {}

PadBuilder::~PadBuilder() {}

OH_NN_ReturnCode PadBuilder::SetConstantValue(std::shared_ptr<NNTensor> tensor)
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

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_PAD_CONSTANT_VALUE:
                returnCode = SetConstantValue(tensor);
                break;
            default:
                LOGE("[Pad] Parameter Type is invalid, type=%d", tensor->GetType());
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

    mindspore::lite::PaddingMode padding_mode = mindspore::lite::PADDING_MODE_CONSTANT;
    void* primitive = MindIR_PadFusion_CreatePrimitive(paddings, padding_mode, m_constantValue);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(PadBuilder, OH_NN_OPS_PAD);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespcae OHOS