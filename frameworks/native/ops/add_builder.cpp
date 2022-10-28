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

#include "add_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "Add";

AddBuilder::AddBuilder() {}

AddBuilder::~AddBuilder() {}

OH_NN_ReturnCode AddBuilder::SetActivation(std::shared_ptr<NNTensor>& tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[Add] SetActivation failed, the activationType should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Add] SetActivation GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    int8_t* fuseData = static_cast<int8_t*>(buffer);
    if (!Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*fuseData))) {
        LOGE("[Add] SetActivation failed, fuse activation type is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_activationType = NNToMS::TransfromFusionType(static_cast<OH_NN_FuseType>(*fuseData));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode AddBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                   const std::vector<uint32_t>& inputsIndex,
                                   const std::vector<uint32_t>& outputsIndex,
                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Add] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Add] Build failed, the input or output index of Add operation is invalid.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (uint32_t i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_ADD_ACTIVATIONTYPE:
                ret = SetActivation(tensor);
                break;
            default:
                LOGE("[Add] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (ret != OH_NN_SUCCESS) {
            LOGE("[Add] Build failed, passed invalid param.");
            return ret;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr AddBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Add] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_AddFusion_CreatePrimitive(m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(AddBuilder, OH_NN_OPS_ADD);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
