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

#include "div_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static constexpr int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Div";

DivBuilder::DivBuilder() {}

DivBuilder::~DivBuilder() {}

OH_NN_ReturnCode DivBuilder::SetActicationType(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Div] SetActicationType failed, the Activation shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[Div] SetActicationType failed, the activation should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Div] SetActivation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int8_t* fuseData = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*fuseData))) {
        LOGE("[Div] SetActicationType failed, fuse activation type is invalid");
        return OH_NN_INVALID_PARAMETER;
    }
    auto fuseType = (OH_NN_FuseType)(*fuseData);
    m_activationType = NNToMS::TransfromFusionType(fuseType);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DivBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                   const std::vector<uint32_t>& inputsIndex,
                                   const std::vector<uint32_t>& outputsIndex,
                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Div] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Div] Build failed, passed invalid input or output index.");
        return returnCode;
    }
    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_DIV_ACTIVATIONTYPE:
                returnCode = SetActicationType(tensor);
                break;
            default:
                LOGE("[Div] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Div] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr DivBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Div] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_DivFusion_CreatePrimitive(m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(DivBuilder, OH_NN_OPS_DIV);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
