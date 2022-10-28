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

#include "hswish_builder.h"

#include "mindir.h"

#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUMS = 1;
static const int OUTPUT_NUMS = 1;
static const std::string OP_NAME = "Hswish";

HswishBuilder::HswishBuilder() {}

HswishBuilder::~HswishBuilder() {}

OH_NN_ReturnCode HswishBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                      const std::vector<uint32_t>& inputsIndex,
                                      const std::vector<uint32_t>& outputsIndex,
                                      const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Hswish] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUMS, OUTPUT_NUMS);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Hswish] Build failed, passed invalid input or output indices.");
        return returnCode;
    }

    if (!paramsIndex.empty()) {
        LOGW("[Hswish] Build failed, hswish expects no parameters, but receive %zu", paramsIndex.size());
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr HswishBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Hswish] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    mindspore::lite::ActivationType activationType = mindspore::lite::ACTIVATION_TYPE_HSWISH;
    float alpha = 0.0f;
    float minVal = 0.0f;
    float maxVal = 0.0f;
    bool approximate = false;
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType,
        alpha, minVal, maxVal, approximate);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);

    return graphPrimitivePtr;
}

REGISTER_OPS(HswishBuilder, OH_NN_OPS_HSWISH);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS